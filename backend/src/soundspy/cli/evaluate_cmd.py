"""``soundspy evaluate``: score the detection model or an extraction method.

    soundspy evaluate detection [--experiment det_mn10_v1] [--split test|val]
    soundspy evaluate extraction --method mask_baseline [--split test] [--n 500]
                                 [--sample-rate 48000]

**detection** scores the trained model against the zero-shot baseline, writes
``experiments/<name>/metrics.json`` and prints a per-class table (without a trained checkpoint
only the zero-shot baseline is scored).

**extraction** (step 06) runs five systems over the same seeded subset of recipes (no processing,
the baseline with the offline / live / true activity, and the ideal-ratio-mask upper bound),
writes ``experiments/ext_<method>/`` (``metrics.json``, ``metrics.md``, ``per_recipe.csv``) and
renders listening examples to ``outputs/eval/ext_<method>/``.
"""

from __future__ import annotations

import argparse
import math

NAME = "evaluate"
HELP = "evaluate the detection model or an extraction method"


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(NAME, help=HELP, description=HELP)
    parser.add_argument("task", choices=("detection", "extraction"))
    parser.add_argument("--experiment", default=None, help="experiment name (default from config)")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="only the first N recipes")
    parser.add_argument("--method", choices=("mask_baseline", "neural_tse"),
                        default="mask_baseline",
                        help="extraction: mask_baseline = the five step-06 systems; neural_tse = "
                             "those plus the neural extractor, also on a reverberant copy")
    parser.add_argument("--n", type=int, default=None,
                        help="extraction: number of recipes (default from config)")
    parser.add_argument("--sample-rate", type=int, default=None,
                        help="extraction: processing rate (default from config)")
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


def run_extraction(args: argparse.Namespace) -> int:
    from soundspy.config import ConfigError, load_config
    from soundspy.data.mixtures import RecipeError
    from soundspy.data.prepare import PrepareError
    from soundspy.training.evaluate_extraction import (
        ExtractionEvalError,
        evaluate_extraction,
        format_markdown,
    )

    try:
        config = load_config()
        result = evaluate_extraction(config, args.method, args.split, n=args.n,
                                     sample_rate=args.sample_rate, device=args.device,
                                     progress=lambda m: print(m, flush=True))
    except (ConfigError, ExtractionEvalError, PrepareError, RecipeError, ValueError) as exc:
        print(f"error: {exc}")
        return 1
    print()
    print(format_markdown(result, list(config.labels)))
    print(f"-> {result['experiment_dir']}")
    if result["examples_dir"]:
        print(f"listening examples -> {result['examples_dir']}")
    return 0


def run(args: argparse.Namespace) -> int:
    if args.task == "extraction":
        return run_extraction(args)

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
