"""``soundspy report`` (step 08): the evaluation document and its figures from fixture result
files; missing results leave notes instead of failing."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from soundspy.report import build


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _system(si: float, sup: float = -20.0) -> dict:
    return {"si_sdr_i": si, "si_sdr_i_median": si + 0.1, "suppression_db": sup, "worse": 7,
            "n": 483, "n_empty": 17}


def _fixtures(exp: Path) -> None:
    per_class = {c: {"ap": 0.5 + 0.05 * i, "f1": 0.4} for i, c in enumerate(("speech", "music"))}
    det = {"per_class": per_class, "map": 0.55, "f1": 0.45}
    _write(exp / "det_mn10_v1" / "metrics.json",
           {s: {"fine_tuned": det, "zero_shot": {**det, "map": 0.41}} for s in ("val", "test")})
    overall = {"mixture": _system(0.0, 0.0), "timeline": _system(2.19), "live": _system(-4.85),
               "oracle": _system(8.1), "irm": _system(24.5)}
    by_class = {"speech": {"timeline": {"si_sdr_i": 1.0}}}
    _write(exp / "ext_mask_baseline" / "metrics.json",
           {"test": {"attenuation_db": 30, "overall": overall, "by_target_class": by_class}})
    hybrid = {"hybrid_offline": _system(3.2), "hybrid_live": _system(2.06), "wf": _system(3.28),
              "timeline": _system(2.19), "live": _system(-4.85)}
    bench = {"pair_a": {"overall": {"hybrid_offline": {"si_sdr_i": 7.0}}}}
    _write(exp / "ext_hybrid" / "metrics.json", {"test": {"overall": hybrid, "benchmarks": bench}})
    cells = {"dfn": {"pause_short_db": -29.5, "pause_long_db": -31.6},
             "hybrid_live": {"pause_short_db": -33.0, "pause_long_db": -42.5}}
    summary = {"40": {"mixes music": cells, "music_only": {"hybrid_live": {"suppression_db": -46}},
                      "speech_only": {"hybrid_live": {"interruption": 0.003}}},
               "30": {"mix +0 dB": {"hybrid_live": {"si_sdr_i": 9.9}}}}
    _write(exp / "ext_hybrid_07d" / "real_room.json",
           {"sess": {"role": "held-out", "ratios_db": [0.0], "summary": summary}})
    cell = {"speech": {"si_sdr_i": 6.97}, "pair_a": {"si_sdr_i": 6.27},
            "pair_absent": {"suppression_db": -24.3}, "librispeech": {"si_sdr_i": 12.4}}
    _write(exp / "ext_hybrid_07d" / "synthetic_test.json",
           {"split": "test", "attenuations_db": [30.0, 40.0],
            "rows": {"dfn": {"centred": {"30": cell}},
                     "07d": {"centred": {"30": cell}, "live": {"30": cell}}}})
    _write(exp / "realtime" / "latency_latest.json",
           {"time": "2026-11-20T10:00", "modes": {"speech": {
               "title": "Hybrid, speech route", "median_ms": 118.0, "min_ms": 116.0,
               "max_ms": 121.0, "found": 10, "estimate_ms": 124.0}}})
    parts = {"input_ms": 22.0, "chunk_ms": 20.0, "chain_ms": 40.0, "queue_ms": 20.0,
             "output_ms": 22.0}
    _write(exp / "realtime" / "latency_screening.json",
           {"A | speech": {"summary": {"underruns": 0, "rtf_p95_max": 0.34,
                                       "latency_ms_median": 124.0, "extractor_ms": 39.98,
                                       "latency_parts_mean": parts}}})
    summary = {"underruns": 0, "input_dropped": 0, "output_dropped": 0, "rtf_p95_max": 0.37,
               "rtf_p95_median": 0.30, "latency_ms_median": 134.0, "ram_mb_first": 1253.6,
               "ram_mb_last": 1189.2, "vram_mb_first": 58.0, "vram_mb_last": 58.0,
               "latency_parts_mean": parts}
    samples = [{"t_s": 60.0 * i, "underruns": 0, "rtf_p95": 0.3, "ram_mb": 1200.0,
                "vram_reserved_mb": 58.0} for i in range(31)]
    _write(exp / "realtime" / "soak_20261120-000522.json",
           {"time": "2026-11-20T00:05:22", "minutes": 30.0, "route": "speech_enhancer",
            "method": "hybrid", "attenuation_db": 40.0, "overrides": [],
            "thread_priority": "mmcss", "summary": summary, "samples": samples})
    _write(exp / "realtime" / "soak_20261120-001529.json",  # a --set run: not in the tables
           {"time": "2026-11-20T00:15:29", "minutes": 3.0, "route": "waveformer",
            "overrides": ["output.prebuffer_chunks=1"], "summary": summary, "samples": []})


def test_report_from_fixture_results(config, tmp_path: Path) -> None:
    exp = tmp_path / "experiments"
    _fixtures(exp)
    cfg = dataclasses.replace(config, paths={**config.paths, "experiments": exp},
                              project_root=tmp_path)
    result = build(cfg, out=tmp_path / "docs")
    text = result["path"].read_text(encoding="utf-8")
    for heading in ("## 1. Setup", "## 2. Detection", "## 3. Extraction", "## 4. Benchmarks",
                    "## 5. Real room", "## 6. Real time", "## 7. Negative results",
                    "## 9. Limitations"):
        assert heading in text
    assert "+3.20" in text and "−4.85" in text  # extraction rows, typographic minus
    assert "Held-out session" in text and "−42.5" in text
    assert "118.0" in text and "soundspy measure-latency --all" in text
    assert text.count("*Reproduce:*") >= 5  # every table names its command
    assert "**Long runs**" in text and "mmcss" in text and "1253.6 -> 1189.2" in text
    assert "output.prebuffer_chunks" not in text
    assert "07d speech-route gate (offline)" in text and "+12.40" in text
    names = {p.name for p in result["figures"]}
    assert names == {"detection_ap.png", "extraction_si_sdri.png", "pause_suppression.png",
                     "latency_breakdown.png"}
    assert all(p.stat().st_size > 1000 for p in result["figures"])


def test_report_with_nothing_measured_yet(config, tmp_path: Path) -> None:
    cfg = dataclasses.replace(config, paths={**config.paths, "experiments": tmp_path / "none"},
                              project_root=tmp_path)
    result = build(cfg, out=tmp_path / "docs", with_figures=False)
    text = result["path"].read_text(encoding="utf-8")
    assert "no detection results" in text and "not measured yet" in text
    assert "detection" in result["missing"]
