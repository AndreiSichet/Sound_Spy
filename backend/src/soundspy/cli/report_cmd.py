"""``soundspy report``: regenerate ``docs/evaluation.md`` and ``docs/figures/`` (step 08).

    soundspy report [--rerun] [--no-figures] [--out FOLDER]

Builds the evaluation document for the university report from the stored result files (see
``soundspy/report.py``). ``--rerun`` first re-runs the long evaluations (the synthetic test set
with the current hybrid, the speech-route check: about two hours, the graphics card busy).
"""

from __future__ import annotations

import argparse
from pathlib import Path

NAME = "report"
HELP = "regenerate docs/evaluation.md and its figures from the stored results"


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(NAME, help=HELP, description=HELP)
    parser.add_argument("--rerun", action="store_true",
                        help="re-run the long evaluations first (about two hours)")
    parser.add_argument("--no-figures", action="store_true", help="tables only")
    parser.add_argument("--out", type=Path, default=None,
                        help="folder for evaluation.md and figures/ (default: docs/)")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    from soundspy.config import load_config
    from soundspy.report import build, rerun

    config = load_config()
    if args.rerun:
        rerun(config)
    result = build(config, out=args.out, with_figures=not args.no_figures)
    print(f"-> {result['path']}")
    for figure in result["figures"]:
        print(f"   {figure}")
    if result["missing"]:
        print("not available yet (sections say so): " + ", ".join(result["missing"]))
    return 0
