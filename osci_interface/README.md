# Oscilloscope recorder

Run from the repository root with a Python environment that has the packages in
`osci_interface/requirements.txt` and a working VISA driver such as NI-VISA.

```powershell
python .\run_oszi.py
```

Click **Connect** with the VISA address blank to detect a single Tektronix USB
scope. Enable the channels you want to record on the scope. Each plot has its own channel selector. Each measurement includes all enabled
channels, including their units as reported by the scope.

- **Point Measurement** records one group of readings after the settling delay.
  It defaults to one waveform read per channel.
- **Continuous Recording** starts on the next available capture and then
  records groups at the requested interval. It defaults to one waveform read per channel
  every five seconds. Change the applied voltage at the bench and use the
  elapsed-time plots to follow the response. If collecting a group takes longer
  than the interval, the next group starts as soon as the recorder is ready.
  Press **Stop** to finish the run and keep completed groups.

Each mode has its own results table and **Save CSV** button. Each measurement
uses one row per enabled channel, followed by an editable note row. The continuous table includes
the run number and elapsed time since that run started. The elapsed-time plots
show the latest run for the channels selected above them. Save your results before closing the app. Each CSV row
also includes the scope ID, sample count, sample interval, and duration from
the first to the last sample. It includes the scope's waveform settings and
the requested recording interval or point settling delay. The CSV starts
with a blank line before the header.

Mean, min, max, SD, and SEM use all samples captured for each channel in a
measurement. SEM is the sample SD divided by the square root of the sample
count. SD and SEM require at least two samples. The app reads the scope's
displayed waveform, so keep the scope running
and triggering to collect updated waveforms. Use Sample or Average acquisition
and avoid the scope's slow scan mode for quantitative measurements.

[Tektronix TBS1000 series programmer manual](https://download.tek.com/manual/TBS1000-B-EDU-TDS2000-B-C-TDS1000-B-C-EDU-TDS200-TPS2000-Programmer.pdf)
