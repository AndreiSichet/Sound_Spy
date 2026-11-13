"""``AudioSource`` interface.

Abstract base for every audio producer: ``read(seconds) -> AudioBuffer`` for
record-then-process and ``stream() -> Iterator[AudioBuffer]`` yielding blocks for the
later real-time mode. ``MicSource``, ``FileSource`` and ``SimulatedArraySource`` are
interchangeable through this interface.

TODO(step-01): implement.
"""
