"""``soundspy train``: train a model.

    soundspy train detection [--name det_mn10_v1] [--epochs N] [--smoke] [--resume]

Writes ``experiments/<name>/`` (config.json, log.csv, checkpoints/, thresholds.json,
metrics.json, plots/). ``--smoke`` = 2 epochs on 500 train / 200 val recipes. A run stopped with
Ctrl+C (or a closed window) continues with ``--resume``. Extraction training comes in step 07.
"""

from __future__ import annotations

import argparse

NAME = "train"
HELP = "train a model (detection)"


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(NAME, help=HELP, description=HELP)
    parser.add_argument("task", choices=("detection",), help="what to train")
    parser.add_argument("--name", default=None, help="experiment name (default from config)")
    parser.add_argument("--epochs", type=int, default=None, help="override the epoch count")
    parser.add_argument("--smoke", action="store_true",
                        help="quick end-to-end check: 2 epochs on 500 train / 200 val recipes")
    parser.add_argument("--resume", action="store_true", help="continue from checkpoints/last.pt")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--workers", type=int, default=None, help="DataLoader workers")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
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
