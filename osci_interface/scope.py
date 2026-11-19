"""USB waveform transfers for the Tektronix TBS1000 series."""

from contextlib import ExitStack, closing, contextmanager
from typing import NamedTuple

import numpy as np
import pyvisa


CHANNELS = ("CH1", "CH2", "CH3", "CH4")
TEKTRONIX_USB_ID = 0x0699
UNIT_LABELS = {"v": "V", "volts": "V", "a": "A", "amps": "A", "amperes": "A"}


class WaveformMetadata(NamedTuple):
    sample_count: int
    sample_interval_s: float
    description: str


class Waveform(NamedTuple):
    time_s: np.ndarray
    values: np.ndarray
    unit: str
    vertical_range: tuple[float, float]
    metadata: WaveformMetadata


def response_fields(response):
    """Read the named fields in a Tektronix response with headers enabled."""
    fields = {}
    for part in response.strip().strip(";").split(";"):
        header, value = part.split(maxsplit=1)
        fields[header.rsplit(":", 1)[-1].upper()] = value.strip('"')
    return fields


class TektronixScope:
    def __init__(self, instrument):
        self.instrument = instrument
        self.identity = instrument.query("*IDN?").strip()
        if self.identity.split(",")[0].upper() != "TEKTRONIX":
            raise ValueError(f"Expected a Tektronix scope, received {self.identity!r}.")

    def read_waveforms(self):
        selection = response_fields(self.instrument.query("SELECT?"))
        if "CH1" not in selection:
            raise ValueError("The scope did not report its channel display state.")
        channels = [channel for channel in CHANNELS if selection.get(channel) in ("1", "ON")]
        return {channel: self.read_waveform(channel) for channel in channels}

    def read_waveform(self, channel):
        self.instrument.write(f"DATA:SOURCE {channel}")
        fields = response_fields(self.instrument.query("WFMPRE?"))
        if fields["PT_FMT"] != "Y":
            raise ValueError(f"{channel} is returning an envelope. Use Sample or Average acquisition mode.")
        if fields["XUNIT"].lower() not in ("s", "seconds"):
            raise ValueError(f"{channel} is not reporting waveform time in seconds.")
        seconds_per_sample = float(fields["XINCR"])
        units_per_code = float(fields["YMULT"])
        code_offset = float(fields["YOFF"])
        signal_offset = float(fields["YZERO"])
        scaling = (seconds_per_sample, units_per_code, code_offset, signal_offset)
        if not np.isfinite(scaling).all() or seconds_per_sample <= 0 or units_per_code == 0:
            raise ValueError(f"{channel} reports invalid waveform scaling.")

        codes = self.instrument.query_binary_values("CURVE?", datatype="b", container=np.array)
        if not codes.size or codes.size != int(fields["NR_PT"]):
            raise ValueError(f"{channel} returned {codes.size} samples, expected {fields['NR_PT']}.")
        time_s = np.arange(codes.size) * seconds_per_sample
        values = (codes - code_offset) * units_per_code + signal_offset
        unit = UNIT_LABELS.get(fields["YUNIT"].lower(), fields["YUNIT"])
        # Scale the full signed-byte range with the same conversion as the trace.
        code_limits = np.iinfo(np.int8)
        low = (code_limits.min - code_offset) * units_per_code + signal_offset
        high = (code_limits.max - code_offset) * units_per_code + signal_offset
        metadata = WaveformMetadata(codes.size, seconds_per_sample, fields.get("WFID", ""))
        return Waveform(time_s, values, unit, (min(low, high), max(low, high)), metadata)


@contextmanager
def open_scope(address, timeout_ms=2000):
    with ExitStack() as session:
        manager = session.enter_context(closing(pyvisa.ResourceManager()))
        if not address:
            addresses = [
                resource for resource in manager.list_resources("USB?*::INSTR")
                if int(resource.split("::")[1], 0) == TEKTRONIX_USB_ID
            ]
            if not addresses:
                raise RuntimeError("No Tektronix USB scope found. Check the scope and USB connection.")
            if len(addresses) > 1:
                raise RuntimeError(f"Enter a VISA address to choose a scope: {', '.join(addresses)}")
            address = addresses[0]
        instrument = session.enter_context(closing(manager.open_resource(address, open_timeout=timeout_ms)))
        instrument.timeout = timeout_ms
        scope = TektronixScope(instrument)
        header = instrument.query("HEADER?").split()[-1]
        verbose = instrument.query("VERBOSE?").split()[-1]
        # Read the response before closing USB so the final settings change is processed.
        session.callback(instrument.query, f"VERBOSE {verbose};:HEADER {header};:HEADER?")
        instrument.write("HEADER ON;:VERBOSE ON")
        session.callback(instrument.write, instrument.query("DATA?").strip())
        # The TBS1000 transfers up to 2500 samples per waveform.
        instrument.write("DATA:ENCDG RIBINARY;WIDTH 1;START 1;STOP 2500")
        yield scope
