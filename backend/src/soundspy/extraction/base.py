"""``Extractor``: the processor that keeps the target classes and attenuates the rest.

An extractor is a ``Processor`` that reads ``TargetSpec.labels`` and ``attenuation_db``. Two
things every extractor reports, so the session and the UI can show them:

* ``latency_samples`` (from ``Processor``): its algorithmic delay at the configured rate;
* ``status_info()``: ``{"method", "latency_ms", "gating", "target_active"}``.

Implementations: ``MaskBaselineExtractor`` (step 06, detector-gated Wiener filter) and the neural
target-sound extractor of step 07. ``build_extractor`` picks one from ``extraction.method``.
"""

from __future__ import annotations

from typing import Any

from soundspy.engine.processor import Processor


class Extractor(Processor):
    method: str = "extractor"

    def status_info(self) -> dict[str, Any]:
        rate = self.sample_rate or 0
        return {
            "method": self.method,
            "latency_ms": round(1000.0 * self.latency_samples / rate, 2) if rate else 0.0,
            "gating": False,
            "target_active": False,
        }
