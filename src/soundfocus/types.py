"""Core audio container used everywhere in SoundFocus.

``AudioBuffer`` holds ``samples`` as a float32 NumPy array of shape ``(frames, channels)``
(always 2-D, also for mono), the ``sample_rate`` in Hz and a ``metadata`` dict (device,
profile, channel indices, timestamp, label, ...). No code may assume a channel count.

TODO(step-01): implement.
"""
