"""Configuration loading.

Reads ``backend/configs/default.yaml``, the device profiles in ``backend/configs/devices/``,
``classes.yaml`` (-> ``LabelSet``) and the detection / extraction configs. Relative paths in the
configs are resolved against the **project root** (the folder that contains ``backend/`` and
``frontend/``), so ``data/``, ``models/``, ``outputs/`` and ``experiments/`` live at the root.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from soundspy.core.labels import LabelSet

BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIGS_DIR = BACKEND_DIR / "configs"
DEFAULT_PROJECT_ROOT = BACKEND_DIR.parent


class ConfigError(ValueError):
    """A configuration file is missing a field or holds an invalid value."""


@dataclass(frozen=True)
class DeviceProfile:
    """Microphone description from ``configs/devices/<name>.yaml``.

    - ``device``: ``None`` = system default input, an ``int`` device index, or a
      case-insensitive substring of the device name.
    - ``hostapi``: ``None`` = host API preference order (WASAPI first on Windows), or a
      case-insensitive substring of a host API name.
    - ``sample_rate``: ``None`` = the device's native rate.
    - ``raw_channels``: ``None`` = every channel, or the 0-based device channels to keep.
    - ``geometry``: mic coordinates in metres, one ``(x, y, z)`` per kept channel.
    """

    name: str
    device: str | int | None = None
    hostapi: str | None = None
    sample_rate: int | None = None
    raw_channels: tuple[int, ...] | None = None
    geometry: tuple[tuple[float, ...], ...] | None = None

    @property
    def has_geometry(self) -> bool:
        return bool(self.geometry)


DEVICE_PROFILE_KEYS = ("device", "hostapi", "sample_rate", "raw_channels", "geometry")


def parse_device_profile(name: str, data: dict[str, Any], source: str = "") -> DeviceProfile:
    """Validate a device-profile mapping and build a ``DeviceProfile``."""
    where = f"device profile {name!r}" + (f" ({source})" if source else "")

    def fail(message: str) -> ConfigError:
        return ConfigError(f"{where}: {message}")

    unknown = sorted(set(data) - set(DEVICE_PROFILE_KEYS))
    if unknown:
        raise fail(f"unknown field(s) {unknown}; allowed: {list(DEVICE_PROFILE_KEYS)}")

    device = data.get("device")
    if isinstance(device, bool) or not (device is None or isinstance(device, int | str)):
        raise fail(f"'device' must be null, an index or a name, got {device!r}")
    if isinstance(device, int) and device < 0:
        raise fail(f"'device' index must be >= 0, got {device}")
    if isinstance(device, str) and not device.strip():
        raise fail("'device' name must not be empty")

    hostapi = data.get("hostapi")
    if hostapi is not None and (not isinstance(hostapi, str) or not hostapi.strip()):
        raise fail(f"'hostapi' must be null or a host API name, got {hostapi!r}")

    sample_rate = data.get("sample_rate")
    if sample_rate is not None and (
        isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0
    ):
        raise fail(f"'sample_rate' must be null or a positive integer, got {sample_rate!r}")

    raw = data.get("raw_channels")
    raw_channels: tuple[int, ...] | None = None
    if raw is not None:
        if (
            not isinstance(raw, list)
            or not raw
            or not all(isinstance(c, int) and not isinstance(c, bool) and c >= 0 for c in raw)
        ):
            raise fail(
                f"'raw_channels' must be null or a non-empty list of indices >= 0, got {raw!r}"
            )
        if len(set(raw)) != len(raw):
            raise fail(f"'raw_channels' contains duplicates: {raw}")
        raw_channels = tuple(raw)

    geo = data.get("geometry")
    geometry: tuple[tuple[float, ...], ...] | None = None
    if geo is not None:
        if not isinstance(geo, list) or not geo:
            raise fail("'geometry' must be null or a list of [x, y, z] positions in metres")
        for mic in geo:
            if (
                not isinstance(mic, list)
                or len(mic) != 3
                or not all(isinstance(v, int | float) and not isinstance(v, bool) for v in mic)
            ):
                raise fail(f"each 'geometry' entry must be [x, y, z] in metres, got {mic!r}")
        if raw_channels is not None and len(geo) != len(raw_channels):
            raise fail(
                f"'geometry' has {len(geo)} positions but 'raw_channels' keeps "
                f"{len(raw_channels)} channels"
            )
        geometry = tuple(tuple(float(v) for v in mic) for mic in geo)

    return DeviceProfile(
        name=name,
        device=device,
        hostapi=hostapi,
        sample_rate=sample_rate,
        raw_channels=raw_channels,
        geometry=geometry,
    )


@dataclass(frozen=True)
class ApiConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    cors_origins: tuple[str, ...] = ("http://localhost:5173",)
    ws_heartbeat_s: float = 2.0
    ws_client_queue: int = 64


# The sections below have no defaults in code: every value comes from default.yaml.


@dataclass(frozen=True)
class CaptureConfig:
    stall_timeout_s: float
    queue_chunks: int
    startup_silence_max_s: float


@dataclass(frozen=True)
class EngineConfig:
    chunk_ms: float
    rtf_warning: float
    rtf_window_s: float
    status_interval_s: float

    def chunk_frames(self, sample_rate: int) -> int:
        return max(1, round(self.chunk_ms * sample_rate / 1000))


@dataclass(frozen=True)
class OutputConfig:
    device: str | int | None
    prebuffer_chunks: int
    max_queue_chunks: int
    limiter_ceiling_dbfs: float
    limiter_release_ms: float
    default_gain_db: float
    drain_after_s: float = 1.0  # queue above prebuffer + 1 for this long -> drop the oldest
    drain_fade_ms: float = 5.0  # crossfade that hides the jump


@dataclass(frozen=True)
class AnalysisConfig:
    levels_hz: float
    spectrogram_hz: float
    spectrogram_fft: int
    spectrogram_bands: int
    spectrogram_fmin_hz: float
    spectrogram_db_floor: float
    waveform_hz: float
    waveform_points: int


SPLITS: tuple[str, ...] = ("train", "val", "test")
DATASETS: tuple[str, ...] = ("esc50", "fsd50k", "custom", "librispeech")


@dataclass(frozen=True)
class MixtureConfig:
    duration_s: float
    counts: dict[str, int]
    seed: dict[str, int]
    n_foreground: tuple[int, int]
    n_background: tuple[int, int]
    target_size: tuple[int, int]
    empty_target_prob: float
    event_min_s: float
    fg_gain_db: tuple[float, float]
    bg_gain_db: tuple[float, float]
    target_to_rest_db: tuple[float, float]
    peak_limit: float
    reference_rms_dbfs: float
    # step 07b. The defaults reproduce the recipes generated before it (the frozen val / test
    # files); training draws its mixtures with the ``dynamic`` overrides.
    partial_target_prob: float = 0.0  # the target sounds only before or after ``switch_s``
    partial_switch: tuple[float, float] = (0.3, 0.7)  # where, as a share of the duration
    hard_negative_prob: float = 0.0  # absent target: put one of its confusable classes in
    hard_negative_pairs: dict[str, tuple[str, ...]] = field(default_factory=dict)
    source_mix: dict[str, dict[str, float]] = field(default_factory=dict)
    # ^ class -> {dataset: share of that class's clips drawn from that dataset}
    lowpass_prob: float = 0.0  # per foreground event: a random low-pass (augmentation)
    lowpass_hz: tuple[float, float] = (6000.0, 16000.0)


@dataclass(frozen=True)
class DataConfig:
    """``configs/data/default.yaml``: datasets, roles, activity, splits, mixtures."""

    enabled: dict[str, bool]
    min_fg_clips_warning: int
    activity_hop_s: float
    activity_threshold_db: float
    esc50_folds: dict[str, tuple[int, ...]]
    custom_val_fraction: float
    custom_test_fraction: float
    custom_seed: int
    fsd50k_min_val_clips_per_class: int
    mixtures: MixtureConfig
    librispeech_subsets: dict[str, str] = field(default_factory=dict)  # split -> subset folder
    librispeech_class: str = ""  # the class every LibriSpeech utterance is foreground for
    dynamic: MixtureConfig | None = None  # ``mixtures`` + ``mixtures.dynamic`` (training)


@dataclass(frozen=True)
class LiveDetectionConfig:
    hop_s: float
    ema_alpha: float
    hysteresis: float
    device: str  # auto | cuda | cpu
    cpu_threads: int  # PyTorch threads of the inference thread when it runs on the CPU


@dataclass(frozen=True)
class TrainingConfig:
    name: str
    epochs: int
    batch_size: int
    num_workers: int
    loader_timeout_s: float
    lr_backbone: float
    lr_head: float
    weight_decay: float
    warmup_epochs: float
    pos_weight: bool
    amp: bool
    seed: int
    smoke_epochs: int
    smoke_train_recipes: int
    smoke_val_recipes: int


@dataclass(frozen=True)
class AugmentConfig:
    gain_db: float
    noise_snr_db: tuple[float, float] | None
    freqm: int
    timem: int
    mel_range: bool


@dataclass(frozen=True)
class DetectionConfig:
    """``configs/detection/default.yaml`` (step 05)."""

    backbone: str
    model_sample_rate: int
    pretrained_file: str
    pretrained_url: str
    pretrained_sha256: str
    audioset_labels: Path  # absolute
    head_init: str  # audioset | random
    window_s: float
    input_context_s: float
    min_active_s: float
    activity_rate_hz: float
    checkpoint: Path | None  # absolute (resolved against the project root)
    live: LiveDetectionConfig
    offline_max_seconds: float
    training: TrainingConfig
    augment: AugmentConfig
    eval_windows_per_recipe: int
    eval_seed: int


BACKBONES = ("efficientat_mn10_as", "stub")
EXTRACTION_METHODS = ("hybrid", "mask_baseline", "neural_tse", "passthrough")


@dataclass(frozen=True)
class MaskBaselineConfig:
    """``configs/extraction/mask_baseline.yaml``: the detector-gated Wiener filter (step 06)."""

    n_fft: int
    hop: int
    open_ms: float
    hold_ms: float
    close_ms: float
    noise_alpha: float
    noise_guard_s: float
    noise_max_age_s: float
    minstat_window_s: float
    minstat_smoothing: float
    minstat_bias: float
    dd_alpha: float
    gain_smoothing: float


@dataclass(frozen=True)
class ExtractionEvalConfig:
    n: int
    seed: int
    sample_rate: int
    attenuation_db: float
    examples: int
    example_ids: tuple[str, ...]  # recipes that are always among the listening examples
    live_delay_s: float
    reverb_seed: int  # rooms of the reverberant copy of the test subset
    # step 07b
    compare_checkpoint: Path | None = None  # an earlier model, shown as its own row
    high_attenuation_db: float = 40.0  # the neural method is also reported at this setting
    bench_seed: int = 0
    bench_absent: int = 0  # recipes of the target-absent benchmark
    bench_pair: tuple[str, str] | None = None  # (class A, class B) of the pair benchmark
    bench_pair_n: int = 0  # recipes per scenario of the pair benchmark
    bench_pair_min_clip_s: float = 0.0  # class B clips at least this long (sustained)
    bench_librispeech: int = 0  # recipes with LibriSpeech test speech as the target
    bench_examples: int = 0  # listening examples per benchmark


@dataclass(frozen=True)
class WaveformerConfig:
    """Architecture of the Waveformer network (``model`` in ``neural_tse.yaml``)."""

    sample_rate: int
    L: int
    enc_dim: int
    num_enc_layers: int
    dec_dim: int
    num_dec_layers: int
    dec_buf_len: int
    dec_chunk_size: int
    out_buf_len: int
    use_pos_enc: bool

    def net_kwargs(self, label_len: int) -> dict[str, Any]:
        """Keyword arguments of ``extraction.models.waveformer.Net``."""
        return {"label_len": label_len, "L": self.L, "enc_dim": self.enc_dim,
                "num_enc_layers": self.num_enc_layers, "dec_dim": self.dec_dim,
                "num_dec_layers": self.num_dec_layers, "dec_buf_len": self.dec_buf_len,
                "dec_chunk_size": self.dec_chunk_size, "out_buf_len": self.out_buf_len,
                "use_pos_enc": self.use_pos_enc}


@dataclass(frozen=True)
class NeuralLiveConfig:
    device: str  # auto | cuda | cpu
    cpu_threads: int
    block_chunks: int
    on_target_change: str  # keep | reset
    ramp_ms: float
    compile: bool = True  # trace the streaming step with TorchScript (same output, faster)
    detector_on_cpu: bool = True  # live: do not share the graphics card with the detector


@dataclass(frozen=True)
class NeuralGateConfig:
    """The output-driven soft gate of the neural extractor (step 07b)."""

    enabled: bool = True
    attack_ms: float = 5.0
    release_ms: float = 150.0
    open_db: float = -12.0  # estimate vs input level above this: fully open
    close_db: float = -22.0  # below this: ``depth_db`` of extra attenuation
    depth_db: float = 25.0
    hold_ms: float = 200.0


@dataclass(frozen=True)
class HybridGateConfig:
    """The hybrid extractor's gate (step 07c): the detector closes it, the output reopens it."""

    enabled: bool = True
    close_prob_factor: float = 0.5  # "confidently absent": probability < factor x threshold
    close_hold_ms: float = 400.0  # ... for this long, and no output rise for this long
    close_ms: float = 150.0  # closing ramp
    open_ms: float = 5.0  # opening ramp
    reopen_ratio_db: float = -10.0  # estimate vs input above this: open at once
    attack_ms: float = 5.0  # level envelopes
    release_ms: float = 50.0
    min_level_dbfs: float = -60.0  # input quieter than this: no "rise" (nothing to judge)
    block_ms: float = 1.0  # decision grid
    warmup_ms: float = 2000.0  # after the speech enhancer starts, its output cannot reopen
    start_closed: bool = True  # a stream that starts with a selection (and a detector): closed


@dataclass(frozen=True)
class SpeechGateConfig:
    """The speech route's gate (step 07d): driven by DeepFilterNet3's own output."""

    enabled: bool = True
    close_db: float = -15.0  # r = L(output) - L(input) below this for the hold: close
    open_db: float = -12.0  # r above this: open (hysteresis: open_db > close_db)
    close_hold_ms: float = 150.0  # the hold when the detector is unsure (or absent)
    close_hold_confident_ms: float = 50.0  # the hold when the detector is sure nothing is there
    detector_override_max_ms: float = 1000.0  # the hold while the detector says "on"
    close_ms: float = 50.0  # closing ramp
    open_ms: float = 5.0  # opening ramp
    ratio_attack_ms: float = 5.0  # level envelopes of output and input
    ratio_release_ms: float = 50.0
    close_prob_factor: float = 0.5  # "sure": probability < factor x threshold
    min_level_dbfs: float = -60.0  # input quieter than this cannot open the gate
    block_ms: float = 1.0  # decision grid
    warmup_ms: float = 0.0  # after the speech route starts, r is not used for this long (closed)
    start_closed: bool = True  # a stream (or the route) starts closed


@dataclass(frozen=True)
class HybridSpeechConfig:
    """The speech route: DeepFilterNet3 (``speech`` in ``hybrid.yaml``)."""

    labels: tuple[str, ...]  # a selection of exactly these classes goes to this route
    route: str  # enhancer | cascade
    model_file: str
    model_url: str
    model_sha256: str
    atten_lim_db: float | None
    post_filter: bool
    device: str  # cpu | cuda | auto
    backend: str  # onnx | torch
    cpu_threads: int
    input_gain_db: float = 0.0  # the network sees the input this much louder (output scaled back)
    model_dir: Path | None = None  # None: <model_cache>/deepfilternet/<model_file stem> (tests
    #                                point it elsewhere)


@dataclass(frozen=True)
class HybridConfig:
    """``configs/extraction/hybrid.yaml`` (step 07c; ``speech_gate`` step 07d)."""

    speech: HybridSpeechConfig
    gate: HybridGateConfig
    ramp_ms: float
    performance_cores: bool
    tuning: dict[str, tuple[float, ...]]  # gate parameter -> values tried on val
    tuning_max_cost_db: float  # strongest suppression whose SI-SDRi cost stays within this
    speech_gate: SpeechGateConfig = SpeechGateConfig()
    speech_tuning: dict[str, tuple[float, ...]] = dataclasses.field(default_factory=dict)
    speech_tuning_max_cut: float = 0.02  # speech-cut rate allowed on the tuning session
    speech_tuning_max_cost_db: float = 0.3  # speech-over-music SI-SDRi below the 07c gate


def speech_model_dir(config: AppConfig) -> Path:
    """Where the speech enhancer's unpacked model lives."""
    speech = config.extraction.hybrid.speech
    if speech.model_dir is not None:
        return Path(speech.model_dir)
    return Path(config.paths["model_cache"]) / "deepfilternet" / Path(speech.model_file).stem


@dataclass(frozen=True)
class ReverbConfig:
    prob: float
    rt60_s: tuple[float, float]
    room_m: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    wall_margin_m: float
    max_order: int


@dataclass(frozen=True)
class ExtractionCacheConfig:
    shard_size: int
    seed: int
    min_free_gb: float
    reverb: ReverbConfig
    clip_cache_datasets: tuple[str, ...] = ()  # datasets converted once to the model rate
    clip_cache_splits: tuple[str, ...] = ("train",)
    rir_bank_rooms: int = 0  # precomputed rooms for dynamic mixing (0 = no bank)
    rir_bank_positions: int = 6
    rir_trim_db: float = -60.0


@dataclass(frozen=True)
class ExtractionLossConfig:
    snr_weight: float
    si_snr_weight: float
    leak_weight: float = 0.0  # frame-wise leakage term (target-silent frames)
    leak_frame_ms: float = 50.0
    leak_floor_db: float = -50.0  # no reward for suppressing a frame below this
    leak_silence_dbfs: float = -70.0  # a frame whose target is below this is "silent"


@dataclass(frozen=True)
class ExtractionTrainingConfig:
    name: str
    epochs: int
    train_recipes: int
    epoch_recipes: int  # crops per epoch (<= train_recipes: epochs rotate through the pool)
    train_crop_s: float
    crop_target_prob: float
    batch_size: int
    max_batch_size: int
    effective_batch_size: int
    num_workers: int
    loader_timeout_s: float
    lr_backbone: float
    lr_label: float
    weight_decay: float
    warmup_epochs: float
    grad_clip: float
    amp: bool
    seed: int
    val_examples: int
    patience: int
    time_budget_h: float
    loss: ExtractionLossConfig
    gain_db: float
    mic_noise_db: tuple[float, float] | None
    smoke_epochs: int
    smoke_train_recipes: int
    smoke_val_examples: int
    # step 07b
    dynamic: bool = False  # mixtures made on the fly instead of the cached training set
    init_checkpoint: Path | None = None  # start from this checkpoint (None: published weights)
    val_empty_examples: int = 0  # extra target-absent validation examples
    require_leak_not_worse: bool = False  # "best" must suppress at least as well as the start


@dataclass(frozen=True)
class NeuralTSEConfig:
    """``configs/extraction/neural_tse.yaml``: the Waveformer extractor (step 07)."""

    model: WaveformerConfig
    pretrained_file: str
    pretrained_url: str
    pretrained_sha256: str
    label_init: dict[str, tuple[str, ...]]  # our class -> pretrained classes ((), = mean of all)
    checkpoint: Path | None  # absolute (resolved against the project root)
    live: NeuralLiveConfig
    gate: NeuralGateConfig
    data: ExtractionCacheConfig
    training: ExtractionTrainingConfig


@dataclass(frozen=True)
class ExtractionConfig:
    """``configs/extraction/default.yaml`` + the per-method files."""

    method: str  # neural_tse | mask_baseline | passthrough (what a new session uses)
    default_attenuation_db: float  # live / UI / offline default; the evaluation has its own
    offline_max_seconds: float
    offline_chunk_frames: int
    evaluation: ExtractionEvalConfig
    mask_baseline: MaskBaselineConfig
    neural_tse: NeuralTSEConfig
    hybrid: HybridConfig


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    configs_dir: Path
    paths: dict[str, Path]
    device_profile: DeviceProfile
    labels: LabelSet
    capture: CaptureConfig
    engine: EngineConfig
    output: OutputConfig
    analysis: AnalysisConfig
    api: ApiConfig
    database_url: str
    data: DataConfig
    detection: DetectionConfig | None = None
    extraction: ExtractionConfig | None = None


class _Section:
    """Reads one mapping from a config file with type and range checks (clear ConfigError).

    ``name`` may be dotted for nested sections (``"mixtures.counts"``); ``source`` names the file.
    """

    def __init__(
        self,
        data: dict[str, Any],
        name: str,
        keys: tuple[str, ...],
        source: str = "default.yaml",
    ) -> None:
        section = data.get(name.rsplit(".", 1)[-1])
        if not isinstance(section, dict):
            raise ConfigError(f"{source}: missing section '{name}'")
        missing = [k for k in keys if k not in section]
        unknown = sorted(set(section) - set(keys))
        if missing:
            raise ConfigError(f"{source}: '{name}' is missing {missing}")
        if unknown:
            raise ConfigError(f"{source}: '{name}' has unknown key(s) {unknown}")
        self.data, self.name, self.source = section, name, source

    def sub(self, key: str, keys: tuple[str, ...]) -> _Section:
        return _Section(self.data, f"{self.name}.{key}", keys, self.source)

    def _fail(self, key: str, rule: str) -> ConfigError:
        value = self.data[key]
        path = f"{self.name}.{key}".lstrip(".")  # top-level sections have an empty name
        return ConfigError(f"{self.source}: '{path}' must be {rule}, got {value!r}")

    def boolean(self, key: str) -> bool:
        value = self.data[key]
        if not isinstance(value, bool):
            raise self._fail(key, "true or false")
        return value

    def pair(self, key: str, *, integer: bool = False, ge: float | None = None,
             le: float | None = None) -> tuple[Any, Any]:
        """A ``[low, high]`` range with low <= high."""
        value = self.data[key]
        kind = "integers" if integer else "numbers"
        ok = (
            isinstance(value, list)
            and len(value) == 2
            and all(
                not isinstance(v, bool) and isinstance(v, int if integer else int | float)
                for v in value
            )
        )
        if not ok or value[0] > value[1]:
            raise self._fail(key, f"[low, high] with two {kind}, low <= high")
        if (ge is not None and value[0] < ge) or (le is not None and value[1] > le):
            raise self._fail(key, f"within [{ge}, {le}]")
        cast = int if integer else float
        return cast(value[0]), cast(value[1])

    def int_list(self, key: str, *, ge: int, le: int) -> tuple[int, ...]:
        value = self.data[key]
        if (
            not isinstance(value, list)
            or not value
            or not all(isinstance(v, int) and not isinstance(v, bool) and ge <= v <= le
                       for v in value)
        ):
            raise self._fail(key, f"a non-empty list of integers in [{ge}, {le}]")
        return tuple(value)

    def number(self, key: str, *, gt: float | None = None, ge: float | None = None,
               le: float | None = None) -> float:
        value = self.data[key]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise self._fail(key, "a number")
        if gt is not None and not value > gt:
            raise self._fail(key, f"> {gt}")
        if ge is not None and not value >= ge:
            raise self._fail(key, f">= {ge}")
        if le is not None and not value <= le:
            raise self._fail(key, f"<= {le}")
        return float(value)

    def integer(self, key: str, *, ge: int) -> int:
        value = self.data[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < ge:
            raise self._fail(key, f"an integer >= {ge}")
        return int(value)


def _capture(data: dict[str, Any]) -> CaptureConfig:
    s = _Section(data, "capture", ("stall_timeout_s", "queue_chunks", "startup_silence_max_s"))
    return CaptureConfig(
        stall_timeout_s=s.number("stall_timeout_s", gt=0),
        queue_chunks=s.integer("queue_chunks", ge=1),
        startup_silence_max_s=s.number("startup_silence_max_s", ge=0),
    )


def _engine(data: dict[str, Any]) -> EngineConfig:
    s = _Section(data, "engine", ("chunk_ms", "rtf_warning", "rtf_window_s", "status_interval_s"))
    return EngineConfig(
        chunk_ms=s.number("chunk_ms", gt=0, le=1000),
        rtf_warning=s.number("rtf_warning", gt=0),
        rtf_window_s=s.number("rtf_window_s", gt=0),
        status_interval_s=s.number("status_interval_s", gt=0),
    )


def _output(data: dict[str, Any]) -> OutputConfig:
    keys = ("device", "prebuffer_chunks", "max_queue_chunks", "limiter_ceiling_dbfs",
            "limiter_release_ms", "default_gain_db", "drain_after_s", "drain_fade_ms")
    s = _Section(data, "output", keys)
    device = s.data["device"]
    if isinstance(device, bool) or not (device is None or isinstance(device, int | str)):
        raise s._fail("device", "null, a device index or a name substring")
    prebuffer = s.integer("prebuffer_chunks", ge=0)
    max_queue = s.integer("max_queue_chunks", ge=1)
    if max_queue < prebuffer:
        raise ConfigError("default.yaml: 'output.max_queue_chunks' must be >= prebuffer_chunks")
    return OutputConfig(
        device=device,
        prebuffer_chunks=prebuffer,
        max_queue_chunks=max_queue,
        limiter_ceiling_dbfs=s.number("limiter_ceiling_dbfs", le=0),
        limiter_release_ms=s.number("limiter_release_ms", gt=0),
        default_gain_db=s.number("default_gain_db", ge=-40, le=12),
        drain_after_s=s.number("drain_after_s", gt=0, le=60),
        drain_fade_ms=s.number("drain_fade_ms", ge=0, le=20),
    )


def _analysis(data: dict[str, Any]) -> AnalysisConfig:
    keys = ("levels_hz", "spectrogram_hz", "spectrogram_fft", "spectrogram_bands",
            "spectrogram_fmin_hz", "spectrogram_db_floor", "waveform_hz", "waveform_points")
    s = _Section(data, "analysis", keys)
    fft = s.integer("spectrogram_fft", ge=64)
    if fft & (fft - 1):
        raise s._fail("spectrogram_fft", "a power of two")
    return AnalysisConfig(
        levels_hz=s.number("levels_hz", gt=0, le=200),
        spectrogram_hz=s.number("spectrogram_hz", gt=0, le=200),
        spectrogram_fft=fft,
        spectrogram_bands=s.integer("spectrogram_bands", ge=8),
        spectrogram_fmin_hz=s.number("spectrogram_fmin_hz", ge=0),
        spectrogram_db_floor=s.number("spectrogram_db_floor", le=-10),
        waveform_hz=s.number("waveform_hz", gt=0, le=200),
        waveform_points=s.integer("waveform_points", ge=2),
    )


def load_data_config(path: Path) -> DataConfig:
    """Load and validate ``configs/data/default.yaml``."""
    source = f"data/{Path(path).name}"
    root = {"root": load_yaml(path)}
    top = _Section(root, "root", ("datasets", "roles", "activity", "splits", "mixtures"), source)
    top.name = ""  # error paths read "datasets.esc50.enabled", not "root.datasets..."

    def section(key: str, keys: tuple[str, ...]) -> _Section:
        return _Section(top.data, key, keys, source)

    datasets = section("datasets", DATASETS)
    enabled = {ds: datasets.sub(ds, ("enabled", "class") if ds == "librispeech"
                                else ("enabled",)).boolean("enabled") for ds in DATASETS}
    libri_class = _text(datasets.sub("librispeech", ("enabled", "class")), "class")
    roles = section("roles", ("min_fg_clips_warning",))
    activity = section("activity", ("hop_s", "threshold_db_below_peak"))
    splits = section("splits", ("esc50", "fsd50k", "custom", "librispeech"))
    libri = splits.sub("librispeech", SPLITS)
    subsets = {split: _text(libri, split) for split in SPLITS}
    if len(set(subsets.values())) != len(SPLITS):
        raise ConfigError(f"{source}: 'splits.librispeech' uses a subset for more than one split")
    esc = splits.sub("esc50", SPLITS)
    folds = {split: esc.int_list(split, ge=1, le=5) for split in SPLITS}
    all_folds = [f for fs in folds.values() for f in fs]
    if len(all_folds) != len(set(all_folds)):
        raise ConfigError(f"{source}: 'splits.esc50' uses a fold in more than one split")
    custom = splits.sub("custom", ("val_fraction", "test_fraction", "seed"))
    val_fraction = custom.number("val_fraction", ge=0, le=0.9)
    test_fraction = custom.number("test_fraction", ge=0, le=0.9)
    if val_fraction + test_fraction >= 1:
        raise ConfigError(f"{source}: 'splits.custom' val + test fractions must be < 1")

    mix_keys = ("duration_s", "counts", "seed", "n_foreground", "n_background", "target_size",
                "empty_target_prob", "event_min_s", "fg_gain_db", "bg_gain_db",
                "target_to_rest_db", "peak_limit", "reference_rms_dbfs", "partial_target_prob",
                "partial_switch", "hard_negative_prob", "hard_negative_pairs", "source_mix",
                "lowpass_prob", "lowpass_hz", "dynamic")
    mix = section("mixtures", mix_keys)
    counts = mix.sub("counts", SPLITS)
    seeds = mix.sub("seed", SPLITS)
    readers = _mixture_readers(source)
    values = {key: read(mix) for key, read in readers.items()}
    mixtures = MixtureConfig(
        counts={s: counts.integer(s, ge=0) for s in SPLITS},
        seed={s: seeds.integer(s, ge=0) for s in SPLITS},
        **values,
    )
    if mixtures.event_min_s > mixtures.duration_s:
        raise ConfigError(f"{source}: 'mixtures.event_min_s' must be <= duration_s")
    overrides = mix.data["dynamic"]
    if not isinstance(overrides, dict) or set(overrides) - set(readers):
        raise ConfigError(f"{source}: 'mixtures.dynamic' may only override {sorted(readers)}")
    dyn = _Section({"dynamic": overrides}, "mixtures.dynamic", tuple(overrides), source)
    dynamic = dataclasses.replace(mixtures, **{key: readers[key](dyn) for key in overrides})
    if dynamic.event_min_s > dynamic.duration_s:
        raise ConfigError(f"{source}: 'mixtures.dynamic.event_min_s' must be <= duration_s")
    return DataConfig(
        enabled=enabled,
        min_fg_clips_warning=roles.integer("min_fg_clips_warning", ge=0),
        activity_hop_s=activity.number("hop_s", gt=0, le=1),
        activity_threshold_db=activity.number("threshold_db_below_peak", gt=0, le=120),
        esc50_folds=folds,
        custom_val_fraction=val_fraction,
        custom_test_fraction=test_fraction,
        custom_seed=custom.integer("seed", ge=0),
        fsd50k_min_val_clips_per_class=splits.sub(
            "fsd50k", ("min_val_clips_per_class",)).integer("min_val_clips_per_class", ge=0),
        mixtures=mixtures,
        librispeech_subsets=subsets,
        librispeech_class=libri_class,
        dynamic=dynamic,
    )


def _mixture_readers(source: str) -> dict[str, Any]:
    """``key -> reader(section)`` for every mixture setting that ``mixtures.dynamic`` may also
    set (everything but ``counts`` and ``seed``)."""

    def pairs(sec: _Section) -> dict[str, tuple[str, ...]]:
        raw = sec.data["hard_negative_pairs"]
        ok = isinstance(raw, dict) and all(
            isinstance(k, str) and isinstance(v, list) and v
            and all(isinstance(x, str) and x != k for x in v) for k, v in raw.items())
        if not ok:
            raise sec._fail("hard_negative_pairs",
                            "a mapping class -> non-empty list of other classes")
        return {k: tuple(v) for k, v in raw.items()}

    def sources(sec: _Section) -> dict[str, dict[str, float]]:
        raw = sec.data["source_mix"]
        ok = isinstance(raw, dict) and all(
            isinstance(k, str) and isinstance(v, dict) and all(
                d in DATASETS and isinstance(p, int | float) and not isinstance(p, bool)
                and 0 <= p <= 1 for d, p in v.items()) for k, v in raw.items())
        if not ok:
            raise sec._fail("source_mix", f"a mapping class -> {{dataset: share in [0, 1]}} "
                                          f"with datasets from {list(DATASETS)}")
        return {k: {d: float(p) for d, p in v.items()} for k, v in raw.items()}

    return {
        "duration_s": lambda s: s.number("duration_s", gt=0, le=120),
        "n_foreground": lambda s: s.pair("n_foreground", integer=True, ge=1),
        "n_background": lambda s: s.pair("n_background", integer=True, ge=0),
        "target_size": lambda s: s.pair("target_size", integer=True, ge=1),
        "empty_target_prob": lambda s: s.number("empty_target_prob", ge=0, le=1),
        "event_min_s": lambda s: s.number("event_min_s", gt=0),
        "fg_gain_db": lambda s: s.pair("fg_gain_db", ge=-60, le=20),
        "bg_gain_db": lambda s: s.pair("bg_gain_db", ge=-60, le=20),
        "target_to_rest_db": lambda s: s.pair("target_to_rest_db", ge=-60, le=60),
        "peak_limit": lambda s: s.number("peak_limit", gt=0, le=1),
        "reference_rms_dbfs": lambda s: s.number("reference_rms_dbfs", ge=-80, le=0),
        "partial_target_prob": lambda s: s.number("partial_target_prob", ge=0, le=1),
        "partial_switch": lambda s: s.pair("partial_switch", ge=0.05, le=0.95),
        "hard_negative_prob": lambda s: s.number("hard_negative_prob", ge=0, le=1),
        "hard_negative_pairs": pairs,
        "source_mix": sources,
        "lowpass_prob": lambda s: s.number("lowpass_prob", ge=0, le=1),
        "lowpass_hz": lambda s: s.pair("lowpass_hz", ge=500, le=24000),
    }


def load_detection_config(path: Path, project_root: Path) -> DetectionConfig:
    """Load and validate ``configs/detection/default.yaml``."""
    path = Path(path)
    source = f"detection/{path.name}"
    top = _Section({"root": load_yaml(path)}, "root", (
        "backbone", "model_sample_rate", "pretrained", "audioset_labels", "head_init",
        "window_s", "input_context_s", "min_active_s", "activity_rate_hz", "checkpoint", "live",
        "offline",
        "training", "augment", "evaluation"), source)
    top.name = ""
    d = top.data

    def sub(key: str, keys: tuple[str, ...]) -> _Section:
        return _Section(d, key, keys, source)

    def choice(sec: _Section, key: str, options: tuple[str, ...]) -> str:
        if sec.data[key] not in options:
            raise sec._fail(key, f"one of {list(options)}")
        return str(sec.data[key])

    def text(sec: _Section, key: str) -> str:
        if not isinstance(sec.data[key], str) or not sec.data[key]:
            raise sec._fail(key, "a non-empty string")
        return str(sec.data[key])

    pre = sub("pretrained", ("file", "url", "sha256"))
    sha = text(pre, "sha256").lower()
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise pre._fail("sha256", "a 64-character hex SHA-256")
    live = sub("live", ("hop_s", "ema_alpha", "hysteresis", "device", "cpu_threads"))
    off = sub("offline", ("max_seconds",))
    tr = sub("training", ("name", "epochs", "batch_size", "num_workers", "loader_timeout_s",
                          "lr_backbone",
                          "lr_head", "weight_decay", "warmup_epochs", "pos_weight", "amp",
                          "seed", "smoke_epochs", "smoke_train_recipes", "smoke_val_recipes"))
    aug = sub("augment", ("gain_db", "noise_snr_db", "freqm", "timem", "mel_range"))
    ev = sub("evaluation", ("windows_per_recipe", "seed"))
    window = top.number("window_s", gt=0, le=10)
    checkpoint = d["checkpoint"]
    if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint):
        raise top._fail("checkpoint", "a path or null")
    return DetectionConfig(
        backbone=choice(top, "backbone", BACKBONES),
        model_sample_rate=top.integer("model_sample_rate", ge=8000),
        pretrained_file=text(pre, "file"),
        pretrained_url=text(pre, "url"),
        pretrained_sha256=sha,
        audioset_labels=path.parent / text(top, "audioset_labels"),
        head_init=choice(top, "head_init", ("audioset", "random")),
        window_s=window,
        input_context_s=top.number("input_context_s", ge=window, le=60),
        min_active_s=top.number("min_active_s", ge=0, le=window),
        activity_rate_hz=top.number("activity_rate_hz", gt=0, le=1000),
        checkpoint=resolve_path(checkpoint, project_root) if checkpoint else None,
        live=LiveDetectionConfig(
            hop_s=live.number("hop_s", gt=0, le=window),
            ema_alpha=live.number("ema_alpha", gt=0, le=1),
            hysteresis=live.number("hysteresis", ge=0, le=1),
            device=choice(live, "device", ("auto", "cuda", "cpu")),
            cpu_threads=live.integer("cpu_threads", ge=1),
        ),
        offline_max_seconds=off.number("max_seconds", gt=0),
        training=TrainingConfig(
            name=text(tr, "name"),
            epochs=tr.integer("epochs", ge=1),
            batch_size=tr.integer("batch_size", ge=1),
            num_workers=tr.integer("num_workers", ge=0),
            loader_timeout_s=tr.number("loader_timeout_s", gt=0),
            lr_backbone=tr.number("lr_backbone", gt=0),
            lr_head=tr.number("lr_head", gt=0),
            weight_decay=tr.number("weight_decay", ge=0),
            warmup_epochs=tr.number("warmup_epochs", ge=0),
            pos_weight=tr.boolean("pos_weight"),
            amp=tr.boolean("amp"),
            seed=tr.integer("seed", ge=0),
            smoke_epochs=tr.integer("smoke_epochs", ge=1),
            smoke_train_recipes=tr.integer("smoke_train_recipes", ge=1),
            smoke_val_recipes=tr.integer("smoke_val_recipes", ge=1),
        ),
        augment=AugmentConfig(
            gain_db=aug.number("gain_db", ge=0, le=40),
            noise_snr_db=(None if aug.data["noise_snr_db"] is None
                          else aug.pair("noise_snr_db", ge=-20, le=120)),
            freqm=aug.integer("freqm", ge=0),
            timem=aug.integer("timem", ge=0),
            mel_range=aug.boolean("mel_range"),
        ),
        eval_windows_per_recipe=ev.integer("windows_per_recipe", ge=1),
        eval_seed=ev.integer("seed", ge=0),
    )


def load_mask_baseline_config(path: Path) -> MaskBaselineConfig:
    """Load and validate ``configs/extraction/mask_baseline.yaml``."""
    source = f"extraction/{Path(path).name}"
    keys = ("n_fft", "hop", "open_ms", "hold_ms", "close_ms", "noise_alpha", "noise_guard_s",
            "noise_max_age_s", "minstat_window_s", "minstat_smoothing", "minstat_bias",
            "dd_alpha", "gain_smoothing")
    s = _Section({"root": load_yaml(path)}, "root", keys, source)
    s.name = ""
    n_fft, hop = s.integer("n_fft", ge=64), s.integer("hop", ge=1)
    if n_fft % 2:
        raise s._fail("n_fft", "an even number")
    if n_fft % hop or n_fft // hop < 2:
        raise s._fail("hop", f"a divisor of n_fft ({n_fft}) with n_fft / hop >= 2")
    return MaskBaselineConfig(
        n_fft=n_fft,
        hop=hop,
        open_ms=s.number("open_ms", ge=0, le=1000),
        hold_ms=s.number("hold_ms", ge=0, le=5000),
        close_ms=s.number("close_ms", ge=0, le=5000),
        noise_alpha=s.number("noise_alpha", ge=0, le=0.9999),
        noise_guard_s=s.number("noise_guard_s", ge=0, le=5),
        noise_max_age_s=s.number("noise_max_age_s", gt=0),
        minstat_window_s=s.number("minstat_window_s", gt=0, le=30),
        minstat_smoothing=s.number("minstat_smoothing", ge=0, le=0.999),
        minstat_bias=s.number("minstat_bias", ge=1, le=10),
        dd_alpha=s.number("dd_alpha", ge=0, le=0.9999),
        gain_smoothing=s.number("gain_smoothing", ge=0, le=0.999),
    )


def _text(sec: _Section, key: str) -> str:
    if not isinstance(sec.data[key], str) or not sec.data[key]:
        raise sec._fail(key, "a non-empty string")
    return str(sec.data[key])


def _choice(sec: _Section, key: str, options: tuple[str, ...]) -> str:
    if sec.data[key] not in options:
        raise sec._fail(key, f"one of {list(options)}")
    return str(sec.data[key])


def load_neural_tse_config(path: Path, project_root: Path, labels: LabelSet) -> NeuralTSEConfig:
    """Load and validate ``configs/extraction/neural_tse.yaml``."""
    path = Path(path)
    source = f"extraction/{path.name}"
    top = _Section({"root": load_yaml(path)}, "root",
                   ("model", "pretrained", "label_init", "checkpoint", "live", "gate", "data",
                    "training"), source)
    top.name = ""
    d = top.data

    def sub(key: str, keys: tuple[str, ...]) -> _Section:
        return _Section(d, key, keys, source)

    m = sub("model", ("sample_rate", "L", "enc_dim", "num_enc_layers", "dec_dim",
                      "num_dec_layers", "dec_buf_len", "dec_chunk_size", "out_buf_len",
                      "use_pos_enc"))
    model = WaveformerConfig(
        sample_rate=m.integer("sample_rate", ge=8000),
        L=m.integer("L", ge=1),
        enc_dim=m.integer("enc_dim", ge=8),
        num_enc_layers=m.integer("num_enc_layers", ge=1),
        dec_dim=m.integer("dec_dim", ge=8),
        num_dec_layers=m.integer("num_dec_layers", ge=1),
        dec_buf_len=m.integer("dec_buf_len", ge=1),
        dec_chunk_size=m.integer("dec_chunk_size", ge=1),
        out_buf_len=m.integer("out_buf_len", ge=1),
        use_pos_enc=m.boolean("use_pos_enc"),
    )
    if model.enc_dim % model.dec_dim or model.dec_dim % 8:
        raise m._fail("dec_dim", "a multiple of 8 that divides enc_dim")

    pre = sub("pretrained", ("file", "url", "sha256"))
    sha = _text(pre, "sha256").lower()
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise pre._fail("sha256", "a 64-character hex SHA-256")

    init = d["label_init"]
    if not isinstance(init, dict):
        raise ConfigError(f"{source}: 'label_init' must map our classes to pretrained classes")
    unknown = sorted(set(init) - set(labels))
    if unknown:
        raise ConfigError(f"{source}: 'label_init' names unknown class(es) {unknown}; "
                          f"classes: {list(labels)}")
    label_init: dict[str, tuple[str, ...]] = {}
    for name, sources in init.items():
        if not isinstance(sources, list) or not all(isinstance(x, str) for x in sources):
            raise ConfigError(f"{source}: 'label_init.{name}' must be a list of class names")
        label_init[name] = tuple(sources)

    checkpoint = d["checkpoint"]
    if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint):
        raise top._fail("checkpoint", "a path or null")

    live = sub("live", ("device", "cpu_threads", "block_chunks", "on_target_change", "ramp_ms",
                        "compile", "detector_on_cpu"))
    gate = sub("gate", ("enabled", "attack_ms", "release_ms", "open_db", "close_db",
                        "depth_db", "hold_ms"))
    gate_cfg = NeuralGateConfig(
        enabled=gate.boolean("enabled"),
        attack_ms=gate.number("attack_ms", gt=0, le=1000),
        release_ms=gate.number("release_ms", gt=0, le=5000),
        open_db=gate.number("open_db", ge=-80, le=0),
        close_db=gate.number("close_db", ge=-120, le=0),
        depth_db=gate.number("depth_db", ge=0, le=80),
        hold_ms=gate.number("hold_ms", ge=0, le=5000),
    )
    if gate_cfg.close_db >= gate_cfg.open_db:
        raise gate._fail("close_db", f"below open_db ({gate_cfg.open_db})")
    data = sub("data", ("shard_size", "seed", "min_free_gb", "reverb", "clip_cache",
                        "rir_bank"))
    clips = data.sub("clip_cache", ("datasets", "splits"))
    for key, allowed in (("datasets", DATASETS), ("splits", SPLITS)):
        value = clips.data[key]
        if not isinstance(value, list) or any(v not in allowed for v in value):
            raise clips._fail(key, f"a list from {list(allowed)}")
    bank = data.sub("rir_bank", ("rooms", "positions", "trim_db"))
    rv = data.sub("reverb", ("prob", "rt60_s", "room_m", "wall_margin_m", "max_order"))
    rooms = rv.data["room_m"]
    ok = (isinstance(rooms, list) and len(rooms) == 3 and all(
        isinstance(r, list) and len(r) == 2
        and all(isinstance(v, int | float) and not isinstance(v, bool) for v in r)
        and 0 < r[0] <= r[1] for r in rooms))
    if not ok:
        raise rv._fail("room_m", "three [low, high] ranges in metres (length, width, height)")
    margin = rv.number("wall_margin_m", gt=0, le=2)
    if any(r[0] <= 2 * margin for r in rooms):
        raise rv._fail("room_m", f"larger than twice wall_margin_m ({margin} m) in every axis")

    tr = sub("training", (
        "name", "epochs", "train_recipes", "epoch_recipes", "train_crop_s", "crop_target_prob",
        "batch_size",
        "max_batch_size", "effective_batch_size", "num_workers", "loader_timeout_s",
        "lr_backbone", "lr_label", "weight_decay", "warmup_epochs", "grad_clip", "amp", "seed",
        "val_examples", "patience", "time_budget_h", "loss", "augment", "smoke", "dynamic",
        "init_checkpoint", "val_empty_examples", "require_leak_not_worse"))
    loss = tr.sub("loss", ("snr_weight", "si_snr_weight", "leak_weight", "leak_frame_ms",
                           "leak_floor_db", "leak_silence_dbfs"))
    init = tr.data["init_checkpoint"]
    if init is not None and (not isinstance(init, str) or not init):
        raise tr._fail("init_checkpoint", "a path or null")
    aug = tr.sub("augment", ("gain_db", "mic_noise_db"))
    smoke = tr.sub("smoke", ("epochs", "train_recipes", "val_examples"))
    batch = tr.integer("batch_size", ge=1)
    max_batch = tr.integer("max_batch_size", ge=1)
    if max_batch < batch:
        raise tr._fail("max_batch_size", f">= batch_size ({batch})")
    loss_cfg = ExtractionLossConfig(
        snr_weight=loss.number("snr_weight", ge=0),
        si_snr_weight=loss.number("si_snr_weight", ge=0),
        leak_weight=loss.number("leak_weight", ge=0),
        leak_frame_ms=loss.number("leak_frame_ms", gt=0, le=1000),
        leak_floor_db=loss.number("leak_floor_db", le=0),
        leak_silence_dbfs=loss.number("leak_silence_dbfs", le=0),
    )
    if loss_cfg.snr_weight + loss_cfg.si_snr_weight <= 0:
        raise loss._fail("si_snr_weight", "> 0 together with snr_weight")

    return NeuralTSEConfig(
        model=model,
        pretrained_file=_text(pre, "file"),
        pretrained_url=_text(pre, "url"),
        pretrained_sha256=sha,
        label_init=label_init,
        checkpoint=resolve_path(checkpoint, project_root) if checkpoint else None,
        live=NeuralLiveConfig(
            device=_choice(live, "device", ("auto", "cuda", "cpu")),
            cpu_threads=live.integer("cpu_threads", ge=1),
            block_chunks=live.integer("block_chunks", ge=1),
            on_target_change=_choice(live, "on_target_change", ("keep", "reset")),
            ramp_ms=live.number("ramp_ms", ge=0, le=1000),
            compile=live.boolean("compile"),
            detector_on_cpu=live.boolean("detector_on_cpu"),
        ),
        gate=gate_cfg,
        data=ExtractionCacheConfig(
            shard_size=data.integer("shard_size", ge=1),
            seed=data.integer("seed", ge=0),
            min_free_gb=data.number("min_free_gb", ge=0),
            reverb=ReverbConfig(
                prob=rv.number("prob", ge=0, le=1),
                rt60_s=rv.pair("rt60_s", ge=0.05, le=3),
                room_m=tuple((float(r[0]), float(r[1])) for r in rooms),
                wall_margin_m=margin,
                max_order=rv.integer("max_order", ge=0),
            ),
            clip_cache_datasets=tuple(clips.data["datasets"]),
            clip_cache_splits=tuple(clips.data["splits"]),
            rir_bank_rooms=bank.integer("rooms", ge=0),
            rir_bank_positions=bank.integer("positions", ge=1),
            rir_trim_db=bank.number("trim_db", le=-20),
        ),
        training=ExtractionTrainingConfig(
            name=_text(tr, "name"),
            epochs=tr.integer("epochs", ge=1),
            train_recipes=tr.integer("train_recipes", ge=1),
            epoch_recipes=tr.integer("epoch_recipes", ge=1),
            train_crop_s=tr.number("train_crop_s", gt=0),
            crop_target_prob=tr.number("crop_target_prob", ge=0, le=1),
            batch_size=batch,
            max_batch_size=max_batch,
            effective_batch_size=tr.integer("effective_batch_size", ge=1),
            num_workers=tr.integer("num_workers", ge=0),
            loader_timeout_s=tr.number("loader_timeout_s", gt=0),
            lr_backbone=tr.number("lr_backbone", gt=0),
            lr_label=tr.number("lr_label", gt=0),
            weight_decay=tr.number("weight_decay", ge=0),
            warmup_epochs=tr.number("warmup_epochs", ge=0),
            grad_clip=tr.number("grad_clip", gt=0),
            amp=tr.boolean("amp"),
            seed=tr.integer("seed", ge=0),
            val_examples=tr.integer("val_examples", ge=1),
            patience=tr.integer("patience", ge=1),
            time_budget_h=tr.number("time_budget_h", gt=0),
            loss=loss_cfg,
            gain_db=aug.number("gain_db", ge=0, le=40),
            mic_noise_db=(None if aug.data["mic_noise_db"] is None
                          else aug.pair("mic_noise_db", ge=-140, le=0)),
            smoke_epochs=smoke.integer("epochs", ge=1),
            smoke_train_recipes=smoke.integer("train_recipes", ge=1),
            smoke_val_examples=smoke.integer("val_examples", ge=1),
            dynamic=tr.boolean("dynamic"),
            init_checkpoint=resolve_path(init, project_root) if init else None,
            val_empty_examples=tr.integer("val_empty_examples", ge=0),
            require_leak_not_worse=tr.boolean("require_leak_not_worse"),
        ),
    )


HYBRID_TUNABLE = ("close_prob_factor", "close_hold_ms", "reopen_ratio_db")
SPEECH_GATE_TUNABLE = ("close_db", "open_db", "close_hold_ms")
SPEECH_GATE_KEYS = ("enabled", "close_db", "open_db", "close_hold_ms", "close_hold_confident_ms",
                    "detector_override_max_ms", "close_ms", "open_ms", "ratio_attack_ms",
                    "ratio_release_ms", "close_prob_factor", "min_level_dbfs", "block_ms",
                    "warmup_ms", "start_closed")


def _grid(section: _Section, keys: tuple[str, ...]) -> dict[str, tuple[float, ...]]:
    grid: dict[str, tuple[float, ...]] = {}
    for key in keys:
        values = section.data[key]
        if (not isinstance(values, list) or not values
                or not all(isinstance(v, int | float) and not isinstance(v, bool) for v in values)):
            raise section._fail(key, "a non-empty list of numbers")
        grid[key] = tuple(float(v) for v in values)
    return grid


def _speech_gate(top: _Section, source: str) -> SpeechGateConfig:
    s = _Section(top.data, "speech_gate", SPEECH_GATE_KEYS, source)
    gate = SpeechGateConfig(
        enabled=s.boolean("enabled"),
        close_db=s.number("close_db", ge=-80, le=0),
        open_db=s.number("open_db", ge=-80, le=0),
        close_hold_ms=s.number("close_hold_ms", ge=0, le=10000),
        close_hold_confident_ms=s.number("close_hold_confident_ms", ge=0, le=10000),
        detector_override_max_ms=s.number("detector_override_max_ms", ge=0, le=10000),
        close_ms=s.number("close_ms", ge=0, le=5000),
        open_ms=s.number("open_ms", ge=0, le=1000),
        ratio_attack_ms=s.number("ratio_attack_ms", gt=0, le=1000),
        ratio_release_ms=s.number("ratio_release_ms", gt=0, le=5000),
        close_prob_factor=s.number("close_prob_factor", gt=0, le=1),
        min_level_dbfs=s.number("min_level_dbfs", ge=-140, le=0),
        block_ms=s.number("block_ms", gt=0, le=50),
        warmup_ms=s.number("warmup_ms", ge=0, le=10000),
        start_closed=s.boolean("start_closed"),
    )
    if gate.open_db <= gate.close_db:
        raise s._fail("open_db", f"above close_db ({gate.close_db:g}): the hysteresis")
    return gate


def load_hybrid_config(path: Path, labels: LabelSet) -> HybridConfig:
    """Load and validate ``configs/extraction/hybrid.yaml``."""
    path = Path(path)
    source = f"extraction/{path.name}"
    top = _Section({"root": load_yaml(path)}, "root", ("speech", "gate", "speech_gate", "live",
                                                       "tuning", "speech_tuning"), source)
    top.name = ""
    sp = _Section(top.data, "speech", ("labels", "route", "model", "atten_lim_db",
                                       "post_filter", "device", "backend", "cpu_threads",
                                       "input_gain_db"), source)
    chosen = sp.data["labels"]
    if (not isinstance(chosen, list) or not chosen
            or not all(isinstance(c, str) and c in labels for c in chosen)):
        raise sp._fail("labels", f"a non-empty list of classes from {list(labels)}")
    model = sp.sub("model", ("file", "url", "sha256"))
    sha = _text(model, "sha256").lower()
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise model._fail("sha256", "a 64-character hex SHA-256")
    lim = sp.data["atten_lim_db"]
    if lim is not None and (isinstance(lim, bool) or not isinstance(lim, int | float)
                            or not 0 < lim <= 100):
        raise sp._fail("atten_lim_db", "null or a number in (0, 100]")
    g = _Section(top.data, "gate", ("enabled", "close_prob_factor", "close_hold_ms", "close_ms",
                                    "open_ms", "reopen_ratio_db", "attack_ms", "release_ms",
                                    "min_level_dbfs", "block_ms", "warmup_ms", "start_closed"),
                 source)
    gate = HybridGateConfig(
        enabled=g.boolean("enabled"),
        close_prob_factor=g.number("close_prob_factor", gt=0, le=1),
        close_hold_ms=g.number("close_hold_ms", ge=0, le=10000),
        close_ms=g.number("close_ms", ge=0, le=5000),
        open_ms=g.number("open_ms", ge=0, le=1000),
        reopen_ratio_db=g.number("reopen_ratio_db", ge=-80, le=20),
        attack_ms=g.number("attack_ms", gt=0, le=1000),
        release_ms=g.number("release_ms", gt=0, le=5000),
        min_level_dbfs=g.number("min_level_dbfs", ge=-140, le=0),
        block_ms=g.number("block_ms", gt=0, le=50),
        warmup_ms=g.number("warmup_ms", ge=0, le=10000),
        start_closed=g.boolean("start_closed"),
    )
    live = _Section(top.data, "live", ("ramp_ms", "performance_cores"), source)
    tu = _Section(top.data, "tuning", (*HYBRID_TUNABLE, "max_cost_db"), source)
    grid = _grid(tu, HYBRID_TUNABLE)
    st = _Section(top.data, "speech_tuning", (*SPEECH_GATE_TUNABLE, "max_speech_cut",
                                              "max_cost_db"), source)
    return HybridConfig(
        speech=HybridSpeechConfig(
            labels=tuple(chosen),
            route=_choice(sp, "route", ("enhancer", "cascade")),
            model_file=_text(model, "file"),
            model_url=_text(model, "url"),
            model_sha256=sha,
            atten_lim_db=None if lim is None else float(lim),
            post_filter=sp.boolean("post_filter"),
            device=_choice(sp, "device", ("cpu", "cuda", "auto")),
            backend=_choice(sp, "backend", ("onnx", "torch")),
            cpu_threads=sp.integer("cpu_threads", ge=1),
            input_gain_db=sp.number("input_gain_db", ge=-20, le=40),
        ),
        gate=gate,
        ramp_ms=live.number("ramp_ms", ge=0, le=1000),
        performance_cores=live.boolean("performance_cores"),
        tuning=grid,
        tuning_max_cost_db=tu.number("max_cost_db", ge=0, le=10),
        speech_gate=_speech_gate(top, source),
        speech_tuning=_grid(st, SPEECH_GATE_TUNABLE),
        speech_tuning_max_cut=st.number("max_speech_cut", ge=0, le=1),
        speech_tuning_max_cost_db=st.number("max_cost_db", ge=0, le=10),
    )


def load_extraction_config(directory: Path, project_root: Path, labels: LabelSet
                           ) -> ExtractionConfig:
    """Load ``configs/extraction/``: ``default.yaml`` and the per-method files."""
    directory = Path(directory)
    source = "extraction/default.yaml"
    top = _Section({"root": load_yaml(directory / "default.yaml")}, "root",
                   ("method", "default_attenuation_db", "offline", "evaluation"), source)
    top.name = ""
    off = _Section(top.data, "offline", ("max_seconds", "chunk_frames"), source)
    ev = _Section(top.data, "evaluation", ("n", "seed", "sample_rate", "attenuation_db",
                                           "examples", "example_ids", "live_delay_s",
                                           "reverb_seed", "compare_checkpoint",
                                           "high_attenuation_db", "benchmarks"), source)
    ids = ev.data["example_ids"]
    if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
        raise ev._fail("example_ids", "a list of recipe ids")
    compare = ev.data["compare_checkpoint"]
    if compare is not None and (not isinstance(compare, str) or not compare):
        raise ev._fail("compare_checkpoint", "a path or null")
    bench = ev.sub("benchmarks", ("seed", "absent", "pair", "pair_n", "pair_min_clip_s",
                                  "librispeech", "examples"))
    pair = bench.data["pair"]
    if pair is not None and not (isinstance(pair, list) and len(pair) == 2
                                 and all(p in labels for p in pair) and pair[0] != pair[1]):
        raise bench._fail("pair", f"null or two different classes from {list(labels)}")
    return ExtractionConfig(
        method=_choice(top, "method", EXTRACTION_METHODS),
        default_attenuation_db=top.number("default_attenuation_db", ge=0, le=60),
        offline_max_seconds=off.number("max_seconds", gt=0),
        offline_chunk_frames=off.integer("chunk_frames", ge=1),
        evaluation=ExtractionEvalConfig(
            n=ev.integer("n", ge=1),
            seed=ev.integer("seed", ge=0),
            sample_rate=ev.integer("sample_rate", ge=8000),
            attenuation_db=ev.number("attenuation_db", ge=0, le=120),
            examples=ev.integer("examples", ge=0),
            example_ids=tuple(ids),
            live_delay_s=ev.number("live_delay_s", ge=0, le=5),
            reverb_seed=ev.integer("reverb_seed", ge=0),
            compare_checkpoint=resolve_path(compare, project_root) if compare else None,
            high_attenuation_db=ev.number("high_attenuation_db", ge=0, le=120),
            bench_seed=bench.integer("seed", ge=0),
            bench_absent=bench.integer("absent", ge=0),
            bench_pair=tuple(pair) if pair else None,
            bench_pair_n=bench.integer("pair_n", ge=0),
            bench_pair_min_clip_s=bench.number("pair_min_clip_s", ge=0),
            bench_librispeech=bench.integer("librispeech", ge=0),
            bench_examples=bench.integer("examples", ge=0),
        ),
        mask_baseline=load_mask_baseline_config(directory / "mask_baseline.yaml"),
        neural_tse=load_neural_tse_config(directory / "neural_tse.yaml", project_root, labels),
        hybrid=load_hybrid_config(directory / "hybrid.yaml", labels),
    )


CACHE_ENV_VARS: dict[str, str] = {
    "TORCH_HOME": "torch",
    "HF_HOME": "huggingface",
    "XDG_CACHE_HOME": "cache",
}


def configure_cache_env(config: AppConfig) -> dict[str, str]:
    """Point library caches (PyTorch, Hugging Face, generic XDG) at ``paths.model_cache``.

    Only variables the user has not set already are changed; returns the ones that were set.
    Large pretrained weights then land on D: instead of in the user profile on C:.
    """
    cache = config.paths.get("model_cache")
    if cache is None:
        return {}
    applied = {}
    for var, sub in CACHE_ENV_VARS.items():
        if not os.environ.get(var):
            os.environ[var] = str(cache / sub)
            applied[var] = os.environ[var]
    return applied


def load_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return data


def resolve_path(path: str | Path, project_root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (project_root / p).resolve()


def load_device_profile(name: str, configs_dir: Path = DEFAULT_CONFIGS_DIR) -> DeviceProfile:
    path = Path(configs_dir) / "devices" / f"{name}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in (Path(configs_dir) / "devices").glob("*.yaml"))
        raise ConfigError(f"device profile {name!r} not found at {path}; available: {available}")
    return parse_device_profile(name, load_yaml(path), source=str(path))


def load_label_set(path: Path) -> LabelSet:
    """Load ``classes.yaml``: ``{name: {display_name, description, audioset}}``."""
    try:
        return LabelSet.from_mapping(load_yaml(path))
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def load_config(
    configs_dir: Path = DEFAULT_CONFIGS_DIR,
    project_root: Path = DEFAULT_PROJECT_ROOT,
) -> AppConfig:
    configs_dir = Path(configs_dir)
    project_root = Path(project_root).resolve()
    default = load_yaml(configs_dir / "default.yaml")

    paths = {key: resolve_path(value, project_root) for key, value in default["paths"].items()}
    api = default.get("api", {})
    labels = load_label_set(configs_dir / "classes.yaml")

    return AppConfig(
        project_root=project_root,
        configs_dir=configs_dir,
        paths=paths,
        device_profile=load_device_profile(default["active_device_profile"], configs_dir),
        labels=labels,
        capture=_capture(default),
        engine=_engine(default),
        output=_output(default),
        analysis=_analysis(default),
        api=ApiConfig(
            host=str(api.get("host", "127.0.0.1")),
            port=int(api.get("port", 8765)),
            cors_origins=tuple(api.get("cors_origins", ["http://localhost:5173"])),
            ws_heartbeat_s=float(api.get("ws_heartbeat_s", 2.0)),
            ws_client_queue=int(api.get("ws_client_queue", 64)),
        ),
        database_url=str(default["database"]["url"]),
        data=load_data_config(configs_dir / "data" / "default.yaml"),
        detection=load_detection_config(configs_dir / "detection" / "default.yaml",
                                        project_root),
        extraction=load_extraction_config(configs_dir / "extraction", project_root, labels),
    )
