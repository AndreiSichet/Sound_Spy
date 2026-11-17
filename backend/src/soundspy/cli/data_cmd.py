"""``soundspy data``: training data on drive D: (downloads, clip index, mixture recipes).

Subcommands::

    soundspy data paths                      resolved data folders and free space
    soundspy data download ontology esc50    download + verify + extract (resumable)
    soundspy data download fsd50k [--yes]
    soundspy data cleanup-archives [--yes]   delete downloaded archives of finished datasets
    soundspy data prepare                    clip index + activity maps
    soundspy data stats                      coverage table (class x split x dataset)
    soundspy data mix [--split all] [--count N]   mixture recipes (JSONL)
    soundspy data preview [--split val] [--n 5]   render a few recipes to outputs/previews
    soundspy data cache --task extraction [--sample-rate 44100] [--split all]
                                             render every recipe once into arrays on D:
                                             (training then reads them at memory speed)
"""

from __future__ import annotations

import argparse
import time

from soundspy.config import AppConfig, ConfigError, load_config

NAME = "data"
HELP = "training data: download datasets, index clips, build mixture recipes"

PATH_KEYS = ("raw", "custom", "external", "processed", "mixtures", "model_cache")


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(NAME, help=HELP, description=HELP)
    sub = parser.add_subparsers(dest="data_command", metavar="<action>", required=True)

    p = sub.add_parser("paths", help="print the resolved data folders and free space")
    p.set_defaults(handler=run_paths)

    p = sub.add_parser("download", help="download, verify and extract datasets (resumable)")
    p.add_argument("datasets", nargs="+", choices=("ontology", "esc50", "fsd50k", "all"),
                   help="what to download")
    p.add_argument("--yes", action="store_true", help="do not ask before large downloads")
    p.add_argument("--no-progress", action="store_true", help="no progress bars")
    p.set_defaults(handler=run_download)

    p = sub.add_parser("cleanup-archives",
                       help="delete the downloaded archives of fully extracted datasets")
    p.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    p.set_defaults(handler=run_cleanup)

    p = sub.add_parser("prepare", help="build the clip index and the activity maps")
    p.add_argument("--workers", type=int, default=None,
                   help="parallel processes for the activity maps (default: CPU count)")
    p.set_defaults(handler=run_prepare)

    p = sub.add_parser("stats", help="coverage table: clips and minutes per class/split/dataset")
    p.set_defaults(handler=run_stats)

    p = sub.add_parser("mix", help="generate mixture recipes (JSONL, deterministic)")
    p.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    p.add_argument("--count", type=int, default=None,
                   help="recipes per split (default: mixtures.counts from the data config)")
    p.set_defaults(handler=run_mix)

    p = sub.add_parser("preview", help="render a few recipes to outputs/previews (44.1 kHz)")
    p.add_argument("--split", choices=("train", "val", "test"), default="val")
    p.add_argument("--n", type=int, default=5)
    p.set_defaults(handler=run_preview)

    p = sub.add_parser("cache", help="pre-render the mixtures for training (resumable)")
    p.add_argument("--task", choices=("extraction",), required=True)
    p.add_argument("--sample-rate", type=int, default=None,
                   help="default: the extraction model's rate (44100)")
    p.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    p.add_argument("--workers", type=int, default=None,
                   help="parallel processes (default: CPU count - 2)")
    p.add_argument("--limit", type=int, default=None, help="only the first N recipes per split")
    p.add_argument("--force", action="store_true",
                   help="rebuild a cache that was made with other settings")
    p.set_defaults(handler=run_cache)


def _config() -> AppConfig:
    return load_config()


def _fail(message: str) -> int:
    print(f"error: {message}")
    return 1


def _confirm(message: str) -> bool:
    try:
        return input(f"{message} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def run_paths(args: argparse.Namespace) -> int:
    from soundspy.config import CACHE_ENV_VARS
    from soundspy.data.download import GB, free_bytes

    try:
        config = _config()
    except ConfigError as exc:
        return _fail(str(exc))
    width = max(len(k) for k in (*PATH_KEYS, "XDG_CACHE_HOME"))
    print(f"{'project root':<{width}}  {config.project_root}")
    for key in PATH_KEYS:
        path = config.paths[key]
        state = "exists" if path.exists() else "created on first use"
        print(f"{key:<{width}}  {path}   ({state}, {free_bytes(path) / GB:.1f} GB free)")
    import os

    for var in CACHE_ENV_VARS:
        print(f"{var:<{width}}  {os.environ.get(var, '(not set)')}")
    return 0


def run_download(args: argparse.Namespace) -> int:
    from soundspy.data.download import DownloadContext, DownloadError, download

    try:
        config = _config()
    except ConfigError as exc:
        return _fail(str(exc))
    ctx = DownloadContext(
        external=config.paths["external"],
        progress=not args.no_progress,
        confirm=(lambda _m: True) if args.yes else _confirm,
    )
    start = time.monotonic()
    try:
        results = download(args.datasets, ctx)
    except DownloadError as exc:
        return _fail(str(exc))
    except KeyboardInterrupt:
        print("\ninterrupted: run the same command again to resume.")
        return 130
    for res in results:
        took = f" in {res.seconds / 60:.1f} min" if res.status == "done" else ""
        print(f"{res.dataset}: {res.status}{took} -> {res.root}")
    print(f"total {(time.monotonic() - start) / 60:.1f} min")
    return 0


def run_cleanup(args: argparse.Namespace) -> int:
    from soundspy.data.download import GB, archive_dirs, cleanup_archives

    try:
        config = _config()
    except ConfigError as exc:
        return _fail(str(exc))
    dirs = archive_dirs(config.paths["external"])
    if not dirs:
        print("No archives to delete (only datasets whose extraction finished are listed).")
        return 0
    for item in dirs:
        print(f"{item.dataset:<9} {item.files:>3} files  {item.size / GB:7.2f} GB  {item.path}")
    total = sum(d.size for d in dirs)
    if not args.yes and not _confirm(f"Delete these archives ({total / GB:.2f} GB)?"):
        print("Nothing deleted.")
        return 0
    freed = cleanup_archives(dirs)
    print(f"Deleted, {freed / GB:.2f} GB freed.")
    return 0


def run_prepare(args: argparse.Namespace) -> int:
    from soundspy.data.prepare import PrepareError, prepare

    try:
        config = _config()
        start = time.monotonic()
        summary = prepare(config, workers=args.workers)
    except (ConfigError, PrepareError) as exc:
        return _fail(str(exc))
    print(summary.text())
    print(f"prepare took {(time.monotonic() - start) / 60:.1f} min")
    return 0


def run_stats(args: argparse.Namespace) -> int:
    from soundspy.data.prepare import PrepareError, load_clip_index
    from soundspy.data.stats import coverage_report

    try:
        config = _config()
        index = load_clip_index(config.paths["processed"])
    except (ConfigError, PrepareError) as exc:
        return _fail(str(exc))
    print(coverage_report(index, config))
    return 0


def run_mix(args: argparse.Namespace) -> int:
    from soundspy.data.mixtures import RecipeError, write_recipes
    from soundspy.data.prepare import PrepareError, load_activity, load_clip_index

    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    try:
        config = _config()
        index = load_clip_index(config.paths["processed"])
        activity = load_activity(config.paths["processed"])
        for split in splits:
            start = time.monotonic()
            path, count = write_recipes(index, activity, config, split, count=args.count)
            print(f"{split}: {count} recipes -> {path} ({time.monotonic() - start:.1f} s)")
    except (ConfigError, PrepareError, RecipeError) as exc:
        return _fail(str(exc))
    return 0


def run_preview(args: argparse.Namespace) -> int:
    from soundspy.data.mixtures import RecipeError, write_previews
    from soundspy.data.prepare import PrepareError

    try:
        config = _config()
        out_dir, ids = write_previews(config, args.split, args.n)
    except (ConfigError, PrepareError, RecipeError) as exc:
        return _fail(str(exc))
    for rid in ids:
        print(f"{rid}: {rid}_mix.wav, {rid}_target.wav, {rid}.json")
    print(f"-> {out_dir}")
    return 0


def run_cache(args: argparse.Namespace) -> int:
    from soundspy.data.cache import CacheError, build_extraction_cache
    from soundspy.data.mixtures import RecipeError
    from soundspy.data.prepare import PrepareError
    from soundspy.data.reverb import ReverbUnavailable

    try:
        config = _config()
    except ConfigError as exc:
        return _fail(str(exc))
    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    started = time.perf_counter()
    total = 0
    try:
        for split in splits:
            summary = build_extraction_cache(
                config, split, sample_rate=args.sample_rate, workers=args.workers,
                limit=args.limit, force=args.force, progress=lambda m: print(m, flush=True))
            total += summary.bytes
            state = (f"rendered {summary.built} shard(s) in {summary.seconds:.0f} s"
                     if summary.built else "already complete")
            print(f"{split}: {summary.recipes} recipes in {summary.shards} shard(s), "
                  f"{summary.bytes / 1e9:.2f} GB, {state} -> {summary.directory}", flush=True)
    except (CacheError, RecipeError, PrepareError, ReverbUnavailable) as exc:
        return _fail(str(exc))
    print(f"cache: {total / 1e9:.2f} GB in {time.perf_counter() - started:.0f} s")
    return 0
