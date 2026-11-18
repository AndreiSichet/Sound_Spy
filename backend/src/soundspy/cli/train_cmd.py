"""``soundspy train``: train a model.

    soundspy train detection  [--name det_mn10_v1]       [--epochs N] [--smoke] [--resume]
    soundspy train extraction [--name tse_waveformer_v1] [--epochs N] [--smoke] [--resume]

Writes ``experiments/<name>/`` (config.json, log.csv, checkpoints/, metrics.json, plots/;
detection also thresholds.json). A run stopped with Ctrl+C (or a closed window) continues with
``--resume``.

* **detection** (step 05): ``--smoke`` = 2 epochs on 500 train / 200 val recipes.
* **extraction** (step 07): fine-tunes the Waveformer extractor on the pre-rendered cache
  (``soundspy data cache --task extraction`` first). ``--smoke`` = 1 epoch on 256 cached recipes.
  The batch size is found automatically (``--batch-size`` skips the search).
"""

from __future__ import annotations

import argparse

NAME = "train"
HELP = "train a model (detection, extraction)"


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(NAME, help=HELP, description=HELP)
    parser.add_argument("task", choices=("detection", "extraction"), help="what to train")
    parser.add_argument("--name", default=None, help="experiment name (default from config)")
    parser.add_argument("--epochs", type=int, default=None, help="override the epoch count")
    parser.add_argument("--smoke", action="store_true",
                        help="quick end-to-end check on a few hundred recipes")
    parser.add_argument("--resume", action="store_true", help="continue from checkpoints/last.pt")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="extraction: use this batch size instead of searching for one")
    parser.add_argument("--time-budget-h", type=float, default=None,
                        help="extraction: stop when this many hours are used (default: "
                             "training.time_budget_h)")
    parser.set_defaults(handler=run)


def run_extraction(args: argparse.Namespace) -> int:
    from soundspy.config import ConfigError, load_config
    from soundspy.data.cache import CacheError
    from soundspy.data.mixtures import RecipeError
    from soundspy.data.prepare import PrepareError
    from soundspy.extraction.neural_model import NeuralModelError
    from soundspy.training.datasets import LoaderStalled
    from soundspy.training.train import TrainingError
    from soundspy.training.train_extraction import ExtractionTrainOptions, train_extraction

    name = args.name
    try:
        config = load_config()
        if name is None and args.smoke:
            name = config.extraction.neural_tse.training.name + "_smoke"
        opts = ExtractionTrainOptions(
            name=name, epochs=args.epochs, smoke=args.smoke, resume=args.resume,
            device=args.device, allow_cpu=args.device == "cpu", workers=args.workers,
            batch_size=args.batch_size, time_budget_h=args.time_budget_h)
        result = train_extraction(config, opts, log=lambda m: print(m, flush=True))
    except (ConfigError, TrainingError, CacheError, NeuralModelError, LoaderStalled,
            PrepareError, RecipeError, ValueError) as exc:
        print(f"error: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted: continue with the same command plus --resume")
        return 130
    print(f"best epoch {result.best_epoch}: val SI-SDRi {result.best_si_sdri:+.2f} dB "
          f"(batch size {result.batch_size} x {result.accumulate}) -> {result.paths.root}")
    return 0


def run(args: argparse.Namespace) -> int:
    if args.task == "extraction":
        return run_extraction(args)

    from soundspy.config import ConfigError, load_config
    from soundspy.data.mixtures import RecipeError
    from soundspy.data.prepare import PrepareError
    from soundspy.training.datasets import LoaderStalled
    from soundspy.training.evaluate import EvaluationError
    from soundspy.training.train import TrainingError, TrainOptions, train_detection

    name = args.name
    try:
        config = load_config()
        if name is None and args.smoke:
            name = config.detection.training.name + "_smoke"
        opts = TrainOptions(name=name, epochs=args.epochs, smoke=args.smoke, resume=args.resume,
                            device=args.device, allow_cpu=args.device == "cpu",
                            workers=args.workers)
        result = train_detection(config, opts, log=lambda m: print(m, flush=True))
    except (ConfigError, TrainingError, EvaluationError, PrepareError, RecipeError,
            LoaderStalled, ValueError) as exc:
        print(f"error: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted: continue with the same command plus --resume")
        return 130
    print(f"best epoch {result.best_epoch}: val mAP {result.best_map:.4f} "
          f"(batch size {result.batch_size}) -> {result.paths.root}")
    return 0
