"""Desktop oscilloscope recorder."""

import csv
from io import StringIO
import sys

from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

from acquisition import CaptureSettings, ScopeWorker
from scope import CHANNELS


CHANNEL_COLORS = dict(zip(CHANNELS, ("#ffc766", "#70bcff", "#69deaa", "#ff8d9e")))
HORIZONTAL_DIVISIONS = 10
VERTICAL_DIVISIONS = 8


class MeasurementsTable(QtWidgets.QTableWidget):
    def __init__(self, continuous=False):
        super().__init__()
        self.continuous = continuous
        labels = ["Meas No", "Time", "Channel", "Min", "Mean", "Max", "SD", "SEM", "N Waves", "Note", "Unit"]
        if continuous:
            labels.extend(("Run", "Elapsed (s)"))
        timing_label = "Requested interval (s)" if continuous else "Settle delay (s)"
        labels.extend(("Scope ID", "Samples per waveform", "Sample interval (s)", "Waveform duration (s)",
                       "Scope waveform settings", timing_label, "N Samples"))
        self.setColumnCount(len(labels))
        self.setHorizontalHeaderLabels(labels)
        header = self.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        visible_widths = {"Time": 230, "Meas No": 72, "Channel": 72, "Mean": 130,
                          "Min": 120, "Max": 120, "SD": 120, "SEM": 120, "Unit": 60, "N Waves": 80}
        for column, label in enumerate(labels):
            if label in visible_widths:
                self.setColumnWidth(column, max(visible_widths[label], header.sectionSizeHint(column)))
            else:
                self.hideColumn(column)
        # Reorder the display while keeping the CSV column indices stable.
        header.moveSection(header.visualIndex(labels.index("Min")), header.visualIndex(labels.index("Mean")))
        header.moveSection(header.visualIndex(labels.index("Time")), 0)
        header.moveSection(header.visualIndex(labels.index("N Waves")), header.visualIndex(labels.index("Unit")))
        self.note_column = labels.index("Note")
        self.note_display_column = labels.index("Channel")
        self.verticalHeader().hide()
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.cellDoubleClicked.connect(self.edit_note)
        self.setAlternatingRowColors(True)
        self.clear_measurements()

    def edit_note(self, row, _column):
        if self.columnSpan(row, self.note_display_column) > 1:
            self.editItem(self.item(row, self.note_display_column))

    def clear_measurements(self):
        self.clearSpans()
        self.setRowCount(0)
        self.measurement_count = 0

    def append(self, measurement, run_number):
        self.measurement_count += 1
        timestamp = measurement.timestamp.isoformat(sep=" ", timespec="milliseconds")
        first_row = self.rowCount()
        note_row = first_row + len(measurement.channels)
        self.setRowCount(note_row + 1)
        for row, (channel, stats) in enumerate(measurement.channels.items(), start=first_row):
            values = [str(self.measurement_count), timestamp, channel]
            values.extend(stats[1:])
            settings = measurement.settings
            values.extend((str(settings.reads), "", stats.unit))
            if self.continuous:
                values.extend((str(run_number), f"{measurement.elapsed_s:.3f}"))
            metadata = measurement.waveform_metadata[channel]
            timing_s = settings.interval_s if self.continuous else settings.delay_s
            duration_s = (metadata.sample_count - 1) * metadata.sample_interval_s
            values.extend((measurement.scope_identity, str(metadata.sample_count),
                           f"{metadata.sample_interval_s:.12g}", f"{duration_s:.12g}",
                           metadata.description, f"{timing_s:g}", str(settings.reads * metadata.sample_count)))
            for column, value in enumerate(values):
                if isinstance(value, str) or value is None:
                    item = QtWidgets.QTableWidgetItem(value or "")
                else:
                    recorded_value = f"{value:.12g}"
                    item = QtWidgets.QTableWidgetItem(f"{value:.6g}")
                    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, recorded_value)
                    item.setToolTip(recorded_value)
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                self.setItem(row, column, item)
        for column, text in enumerate((str(self.measurement_count), "Note")):
            item = QtWidgets.QTableWidgetItem(text)
            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.setItem(note_row, column, item)
        note = QtWidgets.QTableWidgetItem("")
        note.setToolTip("Double-click to edit the measurement note")
        self.setItem(note_row, self.note_display_column, note)
        self.setSpan(note_row, self.note_display_column, 1, self.columnCount() - self.note_display_column)
        if self.state() != QtWidgets.QAbstractItemView.State.EditingState:
            self.scrollToBottom()

    def write_csv(self, path):
        contents = StringIO(newline="")
        writer = csv.writer(contents)
        writer.writerow([])
        columns = range(self.columnCount())
        writer.writerow(self.horizontalHeaderItem(column).text() for column in columns)
        for row in range(self.rowCount()):
            measurement_number = self.item(row, 0)
            if self.columnSpan(row, self.note_display_column) > 1:
                fields = [""] * self.columnCount()
                fields[0] = measurement_number.text()
                fields[self.note_column] = self.item(row, self.note_display_column).text()
                writer.writerow(fields)
            else:
                items = (self.item(row, column) for column in columns)
                writer.writerow(item.data(QtCore.Qt.ItemDataRole.UserRole) or item.text() for item in items)

        output = QtCore.QSaveFile(path)
        if not output.open(QtCore.QIODevice.OpenModeFlag.WriteOnly):
            raise OSError(output.errorString())
        encoded = contents.getvalue().encode("utf-8-sig")
        if output.write(encoded) != len(encoded):
            output.cancelWriting()
            raise OSError(output.errorString())
        if not output.commit():
            raise OSError(output.errorString())


class GraticuleAxis(pg.AxisItem):
    def __init__(self, orientation, divisions):
        super().__init__(orientation=orientation)
        self.divisions = divisions

    def tickValues(self, minVal, maxVal, size):
        spacing = (maxVal - minVal) / self.divisions
        if spacing <= 0:
            return []
        ticks = [minVal + index * spacing for index in range(self.divisions)]
        return [(spacing, [*ticks, maxVal])]


def signal_plot(x_label, y_label, show_samples=False):
    axes = {
        "bottom": GraticuleAxis("bottom", HORIZONTAL_DIVISIONS),
        "left": GraticuleAxis("left", VERTICAL_DIVISIONS),
    }
    widget = pg.PlotWidget(background="#14202d", axisItems=axes)
    widget.setLabel("bottom", x_label, units="s")
    widget.setLabel("left", y_label)
    widget.showGrid(x=True, y=True, alpha=0.25)
    curve = widget.plot(symbol="o" if show_samples else None, symbolSize=4, symbolPen=None)
    return widget, curve


class GraticulePlot(QtWidgets.QWidget):
    def __init__(self, plot):
        super().__init__()
        self.plot = plot
        self.plot.setParent(self)

    def resizeEvent(self, event):
        width, height = event.size().width(), event.size().height()
        division_size = min(width // HORIZONTAL_DIVISIONS, height // VERTICAL_DIVISIONS)
        plot_width = division_size * HORIZONTAL_DIVISIONS
        plot_height = division_size * VERTICAL_DIVISIONS
        self.plot.setGeometry((width - plot_width) // 2, (height - plot_height) // 2, plot_width, plot_height)
        super().resizeEvent(event)


class WaveformPanel(QtWidgets.QWidget):
    def __init__(self, initial_channel):
        super().__init__()
        self.channel = QtWidgets.QComboBox()
        self.channel.addItems(CHANNELS)
        self.channel.setCurrentText(initial_channel)
        self.plot, self.curve = signal_plot("Time from first sample", "Signal")
        self.plot.setMouseEnabled(x=False, y=False)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        channel_row = QtWidgets.QHBoxLayout()
        channel_row.addWidget(QtWidgets.QLabel("Channel"))
        channel_row.addWidget(self.channel)
        channel_row.addStretch()
        layout.addLayout(channel_row)
        layout.addWidget(GraticulePlot(self.plot), 1)
        self.channel.currentTextChanged.connect(self.set_channel_color)
        self.set_channel_color(initial_channel)

    def set_channel_color(self, channel):
        self.curve.setPen(CHANNEL_COLORS[channel], width=1.5)

    def show_waveform(self, waveform):
        unit = waveform.unit if waveform else ""
        if self.plot.getAxis("left").labelUnits != unit:
            self.plot.setLabel("left", "Signal", units=unit)
        if waveform:
            self.plot.setYRange(*waveform.vertical_range, padding=0)
            if waveform.time_s.size > 1:
                self.plot.setXRange(waveform.time_s[0], waveform.time_s[-1], padding=0)
            self.curve.setData(waveform.time_s, waveform.values)
        else:
            self.curve.clear()


class ScopeWindow(QtWidgets.QWidget):
    connect_requested = QtCore.pyqtSignal(str)
    disconnect_requested = QtCore.pyqtSignal()
    record_requested = QtCore.pyqtSignal(object)
    stop_requested = QtCore.pyqtSignal()

    def __init__(self, poll_interval_ms):
        super().__init__()
        self.setWindowTitle("Oscilloscope Recorder")
        screen = self.screen().availableGeometry()
        title_height = self.style().pixelMetric(QtWidgets.QStyle.PixelMetric.PM_TitleBarHeight, None, self)
        self.resize(screen.width() // 3, screen.height() - title_height)
        self.connected = False
        self.connection_pending = False
        self.recording = False
        self.closing = False
        self.latest_waveforms = {}
        self.reads_by_mode = [1, 1]
        self.run_number = 0
        self.trend_time = []
        self.trend_means = {channel: [] for channel in CHANNELS}
        self.trend_units = {}
        self.build_ui()

        self.thread = QtCore.QThread(self)
        self.worker = ScopeWorker(poll_interval_ms)
        self.worker.moveToThread(self.thread)
        self.connect_requested.connect(self.worker.connect_scope)
        self.disconnect_requested.connect(self.worker.disconnect_scope)
        self.record_requested.connect(self.worker.start_recording)
        self.stop_requested.connect(self.worker.stop_recording)
        self.worker.connected.connect(self.scope_connected)
        self.worker.disconnected.connect(self.scope_disconnected)
        self.worker.waveforms_ready.connect(self.show_waveforms)
        self.worker.measurement_ready.connect(self.show_measurement)
        self.worker.progress.connect(self.status.setText)
        self.worker.recording_finished.connect(self.recording_finished)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.close)
        self.thread.start()
        self.refresh_controls()
        QtCore.QTimer.singleShot(0, self.size_plot_rows)

    def build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(("Point Measurement", "Continuous Recording"))
        self.mode.setMinimumSize(300, 44)
        mode_font = self.mode.font()
        mode_font.setPointSize(13)
        self.mode.setFont(mode_font)
        layout.addWidget(self.mode, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)

        connection_row = QtWidgets.QHBoxLayout()
        self.address = QtWidgets.QLineEdit()
        self.address.setPlaceholderText("Automatically detect a Tektronix USB scope")
        self.connect_button = QtWidgets.QPushButton("Connect")
        self.identity_label = QtWidgets.QLabel()
        self.identity_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        connection_row.addWidget(QtWidgets.QLabel("VISA address"))
        connection_row.addWidget(self.address, 1)
        connection_row.addWidget(self.connect_button)
        connection_row.addWidget(self.identity_label)
        layout.addLayout(connection_row)

        settings_row = QtWidgets.QHBoxLayout()
        self.reads = QtWidgets.QSpinBox()
        self.reads.setRange(1, 1000)
        self.reads.setValue(self.reads_by_mode[0])
        self.delay = QtWidgets.QDoubleSpinBox()
        self.delay.setRange(0, 3600)
        self.delay.setSuffix(" s")
        self.delay_label = QtWidgets.QLabel("Settle delay")
        self.interval = QtWidgets.QDoubleSpinBox()
        self.interval.setRange(0.1, 3600)
        self.interval.setValue(5)
        self.interval.setSuffix(" s")
        self.interval_label = QtWidgets.QLabel("Interval")
        self.filename = QtWidgets.QLineEdit()
        self.filename.setPlaceholderText("measurements.csv")
        self.filename.setMaximumWidth(240)
        filename_label = QtWidgets.QLabel("CSV filename")
        filename_label.setBuddy(self.filename)
        self.filename.setAccessibleName("CSV filename")
        self.record_button = QtWidgets.QPushButton()
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.save_button = QtWidgets.QPushButton("Save CSV")
        self.clear_button = QtWidgets.QPushButton("Clear Results")
        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(self.record_button)
        button_row.addWidget(self.stop_button)
        button_row.addWidget(self.save_button)
        button_row.addWidget(self.clear_button)
        button_row.addStretch()
        settings_row.addWidget(QtWidgets.QLabel("Waveform reads"))
        settings_row.addWidget(self.reads)
        settings_row.addWidget(self.delay_label)
        settings_row.addWidget(self.delay)
        settings_row.addWidget(self.interval_label)
        settings_row.addWidget(self.interval)
        settings_row.addWidget(filename_label)
        settings_row.addWidget(self.filename, 1)
        settings_row.addStretch()

        plots = QtWidgets.QWidget()
        plot_layout = QtWidgets.QHBoxLayout(plots)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self.waveform_panels = (WaveformPanel("CH1"), WaveformPanel("CH2"))
        for panel in self.waveform_panels:
            plot_layout.addWidget(panel, 1)
            panel.channel.currentTextChanged.connect(self.update_waveform_plots)
        self.trend_row = QtWidgets.QWidget()
        trend_layout = QtWidgets.QHBoxLayout(self.trend_row)
        trend_layout.setContentsMargins(0, 0, 0, 0)
        self.trend_plots = [
            signal_plot("Elapsed recording time", "Mean signal", show_samples=True)
            for _ in self.waveform_panels
        ]
        for plot, _ in self.trend_plots:
            trend_layout.addWidget(GraticulePlot(plot), 1)

        self.tables = QtWidgets.QStackedWidget()
        self.tables.addWidget(MeasurementsTable())
        self.tables.addWidget(MeasurementsTable(continuous=True))

        self.status = QtWidgets.QLabel("Disconnected")
        self.status.setWordWrap(True)
        self.status.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        results = QtWidgets.QWidget()
        results_layout = QtWidgets.QVBoxLayout(results)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.addLayout(button_row)
        results_layout.addLayout(settings_row)
        results_layout.addWidget(self.status)
        results_layout.addWidget(self.tables, 1)

        self.plot_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.plot_splitter.addWidget(plots)
        self.plot_splitter.addWidget(self.trend_row)
        self.plot_splitter.addWidget(results)
        layout.addWidget(self.plot_splitter, 1)

        self.connect_button.clicked.connect(self.toggle_connection)
        for panel in self.waveform_panels:
            panel.channel.currentTextChanged.connect(self.change_trend_channels)
        self.change_trend_channels()
        self.mode.currentIndexChanged.connect(self.change_mode)
        self.reads.valueChanged.connect(self.remember_read_count)
        self.record_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_recording)
        self.save_button.clicked.connect(self.save_results)
        self.clear_button.clicked.connect(self.clear_results)

    def refresh_controls(self):
        continuous = self.mode.currentIndex() == 1
        table = self.tables.widget(self.mode.currentIndex())
        self.tables.setCurrentWidget(table)
        self.trend_row.setVisible(continuous)
        self.delay_label.setVisible(not continuous)
        self.delay.setVisible(not continuous)
        self.interval_label.setVisible(continuous)
        self.interval.setVisible(continuous)
        self.connect_button.setText("Disconnect" if self.connected else "Connect")
        self.connect_button.setEnabled(not self.connection_pending)
        self.address.setEnabled(not self.connected and not self.connection_pending)
        self.record_button.setText("Start Recording" if continuous else "Record Point")
        self.record_button.setEnabled(
            self.connected and bool(self.latest_waveforms) and not self.recording and not self.connection_pending
        )
        self.stop_button.setEnabled(self.recording and not self.connection_pending)
        for widget in (self.mode, self.reads, self.delay, self.interval):
            widget.setEnabled(not self.recording)
        self.save_button.setEnabled(table.rowCount() > 0)
        self.clear_button.setEnabled(table.rowCount() > 0 and not self.recording)

    def change_mode(self, index):
        self.reads.setValue(self.reads_by_mode[index])
        self.refresh_controls()
        QtCore.QTimer.singleShot(0, self.size_plot_rows)

    def size_plot_rows(self):
        continuous = not self.trend_row.isHidden()
        plot_rows = 2 if continuous else 1
        splitter = self.plot_splitter
        height = splitter.height() - plot_rows * splitter.handleWidth()
        plot_width = (splitter.width() - splitter.widget(0).layout().spacing()) // len(self.waveform_panels)
        panel_layout = self.waveform_panels[0].layout()
        channel_row_height = panel_layout.itemAt(0).sizeHint().height() + panel_layout.spacing()
        results_height = max(height // 3, splitter.widget(2).minimumSizeHint().height())
        maximum_plot_height = (height - results_height - channel_row_height) // plot_rows
        division_size = min(plot_width // HORIZONTAL_DIVISIONS, maximum_plot_height // VERTICAL_DIVISIONS)
        plot_height = max(1, division_size) * VERTICAL_DIVISIONS
        live_height = plot_height + channel_row_height
        trend_height = plot_height if continuous else 0
        splitter.setSizes([live_height, trend_height, height - live_height - trend_height])

    def remember_read_count(self, reads):
        self.reads_by_mode[self.mode.currentIndex()] = reads

    def toggle_connection(self):
        self.connection_pending = True
        self.refresh_controls()
        if self.connected:
            self.status.setText("Disconnecting…")
            self.disconnect_requested.emit()
        else:
            self.status.setText("Connecting…")
            self.connect_requested.emit(self.address.text().strip())

    def scope_connected(self, identity):
        self.connected = True
        self.connection_pending = False
        self.identity_label.setText(" ".join(identity.split(",")[:2]))
        self.status.setText("Waiting for waveforms…")
        self.refresh_controls()

    def scope_disconnected(self, message):
        self.connected = False
        self.connection_pending = False
        self.recording = False
        self.latest_waveforms = {}
        self.identity_label.clear()
        self.update_waveform_plots()
        self.status.setText(message)
        self.refresh_controls()
        if self.closing:
            self.thread.quit()

    def show_waveforms(self, waveforms):
        channels_changed = tuple(waveforms) != tuple(self.latest_waveforms)
        self.latest_waveforms = waveforms
        if channels_changed or self.status.text() == "Waiting for waveforms…":
            shown = {panel.channel.currentText() for panel in self.waveform_panels}
            for panel in self.waveform_panels:
                if panel.channel.currentText() not in waveforms:
                    replacement = next((channel for channel in waveforms if channel not in shown), None)
                    if replacement is not None:
                        panel.channel.setCurrentText(replacement)
                        shown.add(replacement)
            if not self.recording:
                self.status.setText("Ready" if waveforms else "Enable a channel on the scope.")
            self.refresh_controls()
        self.update_waveform_plots()

    def change_trend_channels(self):
        for panel, (plot, curve) in zip(self.waveform_panels, self.trend_plots):
            channel = panel.channel.currentText()
            color = CHANNEL_COLORS[channel]
            plot.setTitle(f"{channel} recording")
            curve.setPen(color, width=1.5)
            curve.setSymbolBrush(color)
        self.update_trend_plots()

    def update_waveform_plots(self):
        for panel in self.waveform_panels:
            panel.show_waveform(self.latest_waveforms.get(panel.channel.currentText()))

    def update_trend_plots(self):
        for panel, (plot, curve) in zip(self.waveform_panels, self.trend_plots):
            channel = panel.channel.currentText()
            unit = self.trend_units.get(channel, "")
            if plot.getAxis("left").labelUnits != unit:
                plot.setLabel("left", "Mean signal", units=unit)
            if self.trend_means[channel]:
                curve.setData(self.trend_time, self.trend_means[channel])
            else:
                curve.clear()

    def start_recording(self):
        continuous = self.mode.currentIndex() == 1
        if continuous:
            self.run_number += 1
            self.clear_trend()
        self.recording = True
        self.refresh_controls()
        interval_s = self.interval.value() if continuous else None
        delay_s = 0 if continuous else self.delay.value()
        settings = CaptureSettings(self.reads.value(), delay_s, interval_s)
        self.record_requested.emit(settings)

    def stop_recording(self):
        self.stop_button.setEnabled(False)
        self.stop_requested.emit()

    def recording_finished(self, message):
        self.recording = False
        self.status.setText(message)
        self.refresh_controls()

    def show_measurement(self, measurement):
        table = self.tables.currentWidget()
        table.append(measurement, self.run_number)
        if table.continuous:
            self.trend_time.append(measurement.elapsed_s)
            for channel, stats in measurement.channels.items():
                self.trend_units[channel] = stats.unit
                self.trend_means[channel].append(stats.mean)
            self.update_trend_plots()
        self.save_button.setEnabled(True)
        self.clear_button.setEnabled(not self.recording)

    def clear_trend(self):
        self.trend_time.clear()
        self.trend_units.clear()
        for means in self.trend_means.values():
            means.clear()
        self.update_trend_plots()

    def save_results(self):
        table = self.tables.currentWidget()
        filename = self.filename.text().strip() or "measurements.csv"
        if not filename.lower().endswith(".csv"):
            filename += ".csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save measurements", filename, "CSV (*.csv)")
        if path:
            try:
                table.write_csv(path)
            except OSError as error:
                QtWidgets.QMessageBox.critical(self, "Could not save CSV", str(error))
            else:
                self.status.setText(f"Saved {path}")

    def clear_results(self):
        answer = QtWidgets.QMessageBox.question(self, "Clear results", "Clear all results for this mode?")
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            table = self.tables.currentWidget()
            table.clear_measurements()
            if table.continuous:
                self.clear_trend()
                self.run_number = 0
            self.refresh_controls()

    def closeEvent(self, event):
        if self.thread.isRunning():
            event.ignore()
            if not self.closing:
                self.closing = True
                self.setEnabled(False)
                self.disconnect_requested.emit()
        else:
            event.accept()


if __name__ == "__main__":
    preview_interval_ms = 150  # Time between live waveform reads, including during recording.

    application = QtWidgets.QApplication(sys.argv)
    window = ScopeWindow(preview_interval_ms)
    window.show()
    sys.exit(application.exec_())
