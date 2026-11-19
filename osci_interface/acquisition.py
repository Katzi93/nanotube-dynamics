"""Preview and measurement scheduling on a dedicated Qt thread."""

from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
import logging
from time import monotonic
from typing import NamedTuple

import numpy as np
from PyQt5 import QtCore

from scope import WaveformMetadata, open_scope


@dataclass(frozen=True)
class CaptureSettings:
    reads: int
    delay_s: float
    interval_s: float | None


class ChannelStatistics(NamedTuple):
    unit: str
    minimum: float
    mean: float
    maximum: float
    sd: float | None
    sem: float | None


@dataclass(frozen=True)
class Measurement:
    timestamp: datetime
    elapsed_s: float
    settings: CaptureSettings
    scope_identity: str
    channels: dict[str, ChannelStatistics]
    waveform_metadata: dict[str, WaveformMetadata]


def summarize(reads, unit, samples_per_waveform):
    minima, means, maxima, variances = np.asarray(reads).T
    mean = means.mean()
    total_samples = samples_per_waveform * len(means)
    # Combine variation within each waveform with variation between waveform means.
    sum_squares = samples_per_waveform * np.sum(variances + (means - mean) ** 2)
    sd = float(np.sqrt(sum_squares / (total_samples - 1))) if total_samples > 1 else None
    sem = float(sd / np.sqrt(total_samples)) if sd is not None else None
    return ChannelStatistics(unit, float(minima.min()), float(mean), float(maxima.max()), sd, sem)


class Recording:
    def __init__(self, settings, start_s, scope_identity):
        self.settings = settings
        self.start_s = start_s
        self.scope_identity = scope_identity
        self.next_start_s = start_s + settings.delay_s
        self.units = None
        self.samples = {}
        self.batch_metadata = {}
        self.count = 0

    def add_waveforms(self, waveforms, read_time_s, timestamp, metadata):
        if self.count == 0:
            self.batch_start_s = read_time_s
            self.batch_timestamp = timestamp
            self.batch_metadata = metadata
            self.samples = {channel: [] for channel in self.units}
        for channel, waveform in waveforms.items():
            values = waveform.values
            self.samples[channel].append((values.min(), values.mean(), values.max(), values.var()))
        self.count += 1

    def finish_batch(self):
        measurement = Measurement(
            self.batch_timestamp,
            self.batch_start_s - self.start_s,
            self.settings,
            self.scope_identity,
            {
                channel: summarize(reads, self.units[channel], self.batch_metadata[channel].sample_count)
                for channel, reads in self.samples.items()
            },
            self.batch_metadata,
        )
        if self.settings.interval_s is not None:
            self.next_start_s = self.batch_start_s + self.settings.interval_s
        self.count = 0
        return measurement


class ScopeWorker(QtCore.QObject):
    connected = QtCore.pyqtSignal(str)
    disconnected = QtCore.pyqtSignal(str)
    waveforms_ready = QtCore.pyqtSignal(dict)
    measurement_ready = QtCore.pyqtSignal(object)
    progress = QtCore.pyqtSignal(str)
    recording_finished = QtCore.pyqtSignal(str)

    def __init__(self, poll_interval_ms):
        super().__init__()
        self.resources = ExitStack()
        self.scope = None
        self.recording = None
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(poll_interval_ms)
        self.timer.timeout.connect(self.poll)

    @QtCore.pyqtSlot(str)
    def connect_scope(self, address):
        try:
            self.scope = self.resources.enter_context(open_scope(address))
        except Exception as error:
            logging.exception("Could not connect to the scope")
            self.disconnect_scope(f"Connection failed: {error}")
            return
        self.connected.emit(self.scope.identity)
        self.timer.start()
        self.poll()

    @QtCore.pyqtSlot()
    def disconnect_scope(self, message="Disconnected"):
        self.timer.stop()
        self.recording = None
        self.scope = None
        try:
            self.resources.close()
        except Exception as error:
            logging.exception("Could not close the scope connection")
            message = f"{message}. Closing VISA failed: {error}"
        self.disconnected.emit(message)

    @QtCore.pyqtSlot(object)
    def start_recording(self, settings):
        if self.scope is None:
            self.stop_recording("Connect to the oscilloscope before recording.")
            return
        self.recording = Recording(settings, monotonic(), self.scope.identity)
        self.progress.emit("Settling…" if settings.delay_s else "Reading waveforms…")

    @QtCore.pyqtSlot()
    def stop_recording(self, message="Stopped"):
        self.recording = None
        self.recording_finished.emit(message)

    @QtCore.pyqtSlot()
    def poll(self):
        read_time_s = monotonic()
        timestamp = datetime.now().astimezone()
        try:
            waveforms = self.scope.read_waveforms()
        except Exception as error:
            logging.exception("Scope read failed")
            self.disconnect_scope(f"Scope read failed: {error}")
            return

        self.waveforms_ready.emit(waveforms)
        recording = self.recording
        if recording is None or read_time_s < recording.next_start_s:
            return
        units = {channel: waveform.unit for channel, waveform in waveforms.items()}
        if not units:
            self.stop_recording("Recording stopped: no channels are enabled on the scope.")
            return
        if recording.units is None:
            recording.units = units
        if units != recording.units:
            self.stop_recording("Recording stopped: enabled channels or units changed on the scope.")
            return

        metadata = {channel: waveform.metadata for channel, waveform in waveforms.items()}
        if recording.count and metadata != recording.batch_metadata:
            self.stop_recording("Recording stopped: scope settings changed during waveform averaging.")
            return
        recording.add_waveforms(waveforms, read_time_s, timestamp, metadata)
        self.progress.emit(f"Reading waveforms: {recording.count}/{recording.settings.reads}")
        if recording.count == recording.settings.reads:
            self.measurement_ready.emit(recording.finish_batch())
            if recording.settings.interval_s is None:
                self.stop_recording("Point recorded")
            else:
                self.progress.emit("Recording · waiting for the next measurement")
