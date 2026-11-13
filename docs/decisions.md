# Decision log

| Date | Decision | Reason |
|---|---|---|
| 2026-11-13 | Record-then-process first; real-time streaming later. | Simpler to develop, debug and evaluate offline; the `AudioSource.stream()` interface keeps real-time possible later. |
| 2026-11-13 | Laptop built-in mic for modules 1-2; ReSpeaker XVF3800 4-mic USB array from module 3 onward. | Classification and class-based enhancement need no array; localization and beamforming do. Device profiles make the switch a config change. |
| 2026-11-13 | Fine-tune a pretrained audio model instead of training from scratch. | Limited labelled data and compute; pretrained audio representations transfer well to sound event classification. |
| 2026-11-13 | Enhancement: classical classifier-guided spectral masking baseline first, neural separation later. | A transparent baseline with known behaviour gives reference metrics before adding neural complexity. |
| 2026-11-13 | Python + PyTorch; Windows 11 + NVIDIA GPU. | Available hardware and the de-facto ecosystem for audio deep learning. |
