# Literature and notes

## Reference project

- **Semantic Hearing: Programming Acoustic Scenes with Binaural Hearables.** B. Veluri,
  M. Itani, J. Chan, T. Yoshioka, S. Gollakota. UIST 2023.
  Code (MIT): https://github.com/vb000/SemanticHearing
  Ideas adopted: label-conditioned target sound extraction (Waveformer: dilated causal conv
  encoder + causal transformer decoder, label embedding multiplied into the features); chunked
  streaming inference with state buffers; classes mapped to the AudioSet ontology with
  foreground/background sets; synthetic training mixtures from FSD50K / ESC-50 / MUSDB18 / noise
  with spatial simulation; experiment folders with config + checkpoints.
