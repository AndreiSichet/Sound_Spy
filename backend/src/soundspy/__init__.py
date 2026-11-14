"""Sound Spy: listen to the environment and choose what to hear.

The library is the core; ``api`` is a thin FastAPI layer on top of it and the only package
allowed to import FastAPI / SQLAlchemy. See ``docs/architecture.md``.

- ``core``        contracts: AudioBuffer, TargetSpec, events, LabelSet
- ``engine``      streaming engine: Processor / SideProcessor, ProcessorChain, modes, sessions
- ``io``          audio sources and sinks, devices, recording storage
- ``dsp``         shared signal processing (channel reduction, levels, STFT, features)
- ``analysis``    live visual data (levels, spectrogram)
- ``detection``   which classes are present
- ``extraction``  keep the target classes, attenuate the rest
- ``spatial``     localization, beamforming, source tracking (tabs 2-3)
- ``data``        ontology, dataset collectors, splits, mixtures
- ``training``    experiments, training and evaluation loops
- ``api``         FastAPI app, WebSocket, database
- ``cli``         the ``soundspy`` command
"""

__version__ = "0.1.0"
