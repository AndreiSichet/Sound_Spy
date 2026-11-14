# Visualization components

Live and analysis visualizations go here:

- **Canvas**: live waveform, scrolling spectrogram, level meters (step 03).
- **D3**: polar source map for the Spatial and Sources tabs (modules 3-5).
- **Recharts**: metric charts, e.g. SI-SDR improvement per run (step 06+).

Components receive data from the store (fed by `/ws/session`); they never open sockets themselves.
