"""Step 04 must work without PyTorch: importing the data pipeline must not import torch."""

from __future__ import annotations

import subprocess
import sys

MODULES = [
    "soundspy.data.download", "soundspy.data.ontology", "soundspy.data.roles",
    "soundspy.data.splits", "soundspy.data.activity", "soundspy.data.prepare",
    "soundspy.data.stats", "soundspy.data.mixtures", "soundspy.data.collectors.esc50",
    "soundspy.data.collectors.fsd50k", "soundspy.data.collectors.custom",
    "soundspy.dsp.resample", "soundspy.cli.data_cmd", "soundspy.cli.record_dataset_cmd",
]


def test_data_pipeline_does_not_import_torch() -> None:
    code = (
        "import importlib, sys\n"
        f"for m in {MODULES!r}: importlib.import_module(m)\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'torchaudio'))\n"
        "print(bad)\n"
        "sys.exit(1 if bad else 0)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            check=False)
    assert result.returncode == 0, result.stdout + result.stderr
