"""SoundFocus: identify, localize and enhance a selected sound source.

Package layout (see ``docs/architecture.md``):

- ``io``             module 1, audio acquisition (sources, devices, storage)
- ``dsp``            shared signal processing (channel reduction, STFT, features, ...)
- ``data``           module 2, datasets, label mapping, augmentation, mixtures
- ``classification`` module 2, sound event classification
- ``enhancement``    module 2, class-based enhancement
- ``localization``   module 3 (placeholder)
- ``pipeline``       end-to-end processing chain
- ``cli``            the ``soundfocus`` command
"""

__version__ = "0.1.0"
