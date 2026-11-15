"""``soundspy evaluate``: score a trained model against the zero-shot baseline.

    soundspy evaluate detection [--experiment det_mn10_v1] [--split test|val]

Writes the results into ``experiments/<name>/metrics.json`` and prints a per-class table.
Without a trained checkpoint only the zero-shot baseline is scored.
"""

from __future__ import annotations

import argparse
import math

NAME = "evaluate"
HELP = "evaluate a detection model (fine-tuned vs zero-shot)"


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(NAME, help=HELP, description=HELP)
    parser.add_argument("task", choices=("detection",))
    parser.add_argument("--experiment", default=None, help="experiment name (default from config)")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="only the first N recipes")
    parser.set_defaults(handler=run)


def _f(x: float) -> str:
    return "  -  " if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def format_table(result: dict, labels: list[str]) -> str:
    models = [m for m in ("fine_tuned", "zero_shot") if m in result]
    head = f"{'class':<13}" + "".join(f"{m + ' AP':>15}{'F1':>7}{'P':>7}{'R':>7}" for m in models)
    lines = [head, "-" * len(head)]
    for lab in labels:
        cells = ""
        for m in models:
            c = result[m]["per_class"][lab]
            cells += (f"{_f(c['ap']):>15}{_f(c.get('f1')):>7}{_f(c.get('precision')):>7}"
                      f"{_f(c.get('recall')):>7}")
        pos = result[models[0]]["per_class"][lab]["positives"]
        lines.append(f"{lab:<13}{cells}   ({pos} positive windows)")
    cells = "".join(f"{_f(result[m]['map']):>15}{_f(result[m].get('f1')):>7}"
                    f"{_f(result[m].get('precision')):>7}{_f(result[m].get('recall')):>7}"
                    for m in models)
    lines.append(f"{'macro / mAP':<13}{cells}")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    from soundspy.config import ConfigError, load_config
    from soundspy.data.mixtures import RecipeError
    from soundspy.data.prepare import PrepareError
    from soundspy.training.datasets import LoaderStalled
    from soundspy.training.evaluate import EvaluationError, evaluate_detection

    try:
        config = load_config()
        name = args.experiment or config.detection.training.name
        result = evaluate_detection(config, name, args.split, device=args.device,
                                    workers=args.workers, limit=args.limit,
                                    progress=lambda m: print(m, flush=True))
    except (ConfigError, EvaluationError, PrepareError, RecipeError, LoaderStalled,
            ValueError) as exc:
        print(f"error: {exc}")
        return 1
    print(f"\n{args.split}: {result['windows']} windows from {result['recipes']} recipes")
    print(format_table(result, list(config.labels)))
    return 0
