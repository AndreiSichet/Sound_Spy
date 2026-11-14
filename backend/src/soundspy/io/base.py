"""``AudioSource`` and ``AudioSink`` interfaces.

``AudioSource``: ``read(seconds) -> AudioBuffer`` and ``stream() -> Iterator[AudioBuffer]``;
``MicSource``, ``FileSource`` and ``SimulatedArraySource`` are interchangeable.
``AudioSink``: ``write(AudioBuffer)`` / ``close()``; ``SpeakerSink`` and ``FileSink``.

TODO(step-02): implement.
"""
