"""Recording storage.

WAV at native sample rate with all channels + JSON sidecar (device, profile, channels,
sample rate, timestamp, label, notes) + a ``Recording`` row in the database (written via
the API layer). Raw recordings are immutable. Also the custom-dataset manifest.

TODO(step-02): implement.
TODO(step-04): implement.
"""
