# Experiments

One folder per training run (detection or extraction model):

```
experiments/<name>/
├── config.json      # tracked: everything needed to reproduce the run
├── checkpoints/     # git-ignored: model weights
└── logs/            # git-ignored: training logs / metrics
```

Created and read by `soundspy.training.experiment` (step 05); started with `soundspy train`.
