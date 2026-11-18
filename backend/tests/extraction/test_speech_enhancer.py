"""The speech route (step 07c): DeepFilterNet3's DSP against libDF, the streaming wrapper
(chunking, state, delay, resting), streaming == whole signal, loading, the ONNX backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from soundspy.extraction.models.deepfilternet import (
    Analysis,
    DfParams,
    Features,
    Synthesis,
    erb_widths,
    norm_alpha,
    vorbis_window,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "deepfilternet_dsp_reference.npz"
P = DfParams()


# --- DSP: the NumPy version against libDF's own output (stored once from deepfilterlib 0.5.6)


def test_dsp_matches_libdf_reference() -> None:
    ref = np.load(FIXTURE)
    assert np.array_equal(erb_widths(P), ref["erb_widths"])
    assert np.abs(vorbis_window(P.fft_size) - ref["window"]).max() < 1e-7
    x = ref["x"]
    analysis = Analysis(P)  # in uneven pieces: the analysis memory carries over
    spec = np.concatenate([analysis.frames(x[i * 480:(i + 3) * 480].reshape(-1, 480))
                           for i in range(0, 16, 3)])
    assert np.abs(spec - ref["spec"]).max() < 1e-6
    erb, low = Features(P).frames(spec)
    assert np.abs(erb - ref["erb_feat"]).max() < 1e-5
    assert np.abs(low - ref["spec_feat"]).max() < 1e-5
    synthesis = Synthesis(P)
    y = np.concatenate([synthesis.frames(ref["spec"][i:i + 5]) for i in range(0, 16, 5)])
    assert np.abs(y - ref["synthesis"]).max() < 1e-6


def test_stft_reconstructs_the_input_one_window_late() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(20 * 480).astype(np.float32)
    y = Synthesis(P).frames(Analysis(P).frames(x.reshape(-1, 480)))
    assert np.abs(y[480:] - x[:-480]).max() < 1e-5


def test_norm_alpha_rounds_like_deepfilternet() -> None:
    assert norm_alpha(P) == 0.99  # exp(-10 ms / 1 s) = 0.99005 -> 0.99


# --- the streaming wrapper with a stub network: the delay is exact


class IdentityNet:
    """Stands in for ``DfNet``: returns the spectrum of the output frame unchanged."""

    def __init__(self, p: DfParams) -> None:
        import torch

        from soundspy.extraction.models.deepfilternet.model import DfNet

        self.p = p
        self.torch = torch
        self.df_lookahead = p.df_lookahead
        self.conv_lookahead = p.conv_lookahead
        self.df_order = p.df_order
        self.spec_context = DfNet.spec_context.fget(self)
        self.feat_context = p.conv_kernel_inp[0] - 1
        self.calls = 0

    def to(self, device):
        return self

    def eval(self):
        return self

    def modules(self):
        return []

    def init_state(self, device=None):
        t = self.torch
        return (t.zeros(1, 1, self.feat_context, self.p.nb_erb),
                t.zeros(1, 2, self.feat_context, self.p.nb_df),
                t.zeros(1, 1, self.spec_context, self.p.freq_bins, 2), t.zeros(1), t.zeros(1),
                t.zeros(1), t.zeros(1, 1, 1))

    def stream(self, spec, feat_erb, feat_spec, state):
        self.calls += 1
        k = spec.shape[2]
        window = self.torch.cat((state[2], spec), dim=2)
        back = self.df_order - 1 - self.df_lookahead
        out = window[:, :, back:back + k]
        counter = state[6] + k  # some recurrent state that must be carried
        new = (*state[:2], window[:, :, window.shape[2] - self.spec_context:], *state[3:6],
               counter)
        return out, new

    def push_spectra(self, spec, state):
        window = self.torch.cat((state[2], spec), dim=2)
        return (*state[:2], window[:, :, window.shape[2] - self.spec_context:], *state[3:])

    def push_features(self, spec, feat_erb, feat_spec, state):
        return self.push_spectra(spec, state)

    def __call__(self, spec, feat_erb, feat_spec):
        return spec


def _stub_core():
    from soundspy.extraction.speech_enhancer import SpeechCore

    return SpeechCore(IdentityNet(P), P, "cpu", name="stub")


def _stream(enhancer, x: np.ndarray, sizes) -> np.ndarray:
    parts, i, k = [], 0, 0
    while i < x.size:
        n = int(sizes[k % len(sizes)])
        parts.append(enhancer.process(x[i:i + n]))
        i += n
        k += 1
    return np.concatenate(parts)


def test_delay_is_reported_exactly_for_any_chunking() -> None:
    from soundspy.extraction.speech_enhancer import SpeechEnhancer

    enhancer = SpeechEnhancer(_stub_core())
    assert enhancer.latency == 479 + 2 * 480 + 480 == 1919  # 40.0 ms at 48 kHz
    rng = np.random.default_rng(3)
    x = rng.standard_normal(30 * 480).astype(np.float32)
    for sizes in ((960,), (1, 7, 480, 1000, 3), (4096,)):
        enhancer.reset()
        y = _stream(enhancer, np.concatenate((x, np.zeros(enhancer.latency, np.float32))), sizes)
        assert y.size == x.size + enhancer.latency  # as many samples out as in
        assert np.abs(y[enhancer.latency:] - x).max() < 1e-5


def test_resting_outputs_zeros_and_starts_again() -> None:
    from soundspy.extraction.speech_enhancer import SpeechEnhancer

    core = _stub_core()
    enhancer = SpeechEnhancer(core)
    x = np.ones(960 * 10, dtype=np.float32) * 0.1
    rest = enhancer.process(x[:4800], active=False)
    assert not rest.any() and core.net.calls == 0  # the network does not run
    enhancer.process(x[4800:], active=True)
    assert core.net.calls > 0


def test_reset_gives_a_fresh_stream() -> None:
    from soundspy.extraction.speech_enhancer import SpeechEnhancer

    enhancer = SpeechEnhancer(_stub_core())
    rng = np.random.default_rng(4)
    x = rng.standard_normal(9600).astype(np.float32)
    first = _stream(enhancer, x, (960,))
    enhancer.reset()
    assert np.array_equal(_stream(enhancer, x, (960,)), first)


# --- the tiny real network: streaming == whole signal; loading; ONNX


@pytest.fixture
def tiny_dir(speech_kit, tmp_path) -> Path:
    return speech_kit.write(tmp_path)


def test_model_folder_loads_strictly(tiny_dir, speech_kit) -> None:
    from soundspy.extraction.speech_model import load_model, read_params

    p = read_params(tiny_dir / "config.ini")
    assert p.conv_ch == speech_kit.params["conv_ch"] and p.conv_kernel == (1, 3)
    net, params, path = load_model(tiny_dir)
    assert params == p and path.name == "model_1.ckpt.best"


def test_tiny_network_streams_like_the_whole_signal(tiny_dir) -> None:
    from soundspy.extraction.speech_enhancer import (
        SpeechEnhancer,
        enhance_whole,
        load_speech_core,
    )

    core = load_speech_core(tiny_dir, "cpu", backend="torch")
    rng = np.random.default_rng(5)
    x = (0.1 * rng.standard_normal(48000)).astype(np.float32)
    whole = enhance_whole(core, x)
    assert whole.shape == x.shape and np.abs(whole).max() > 0
    enhancer = SpeechEnhancer(core)
    y = _stream(enhancer, np.concatenate((x, np.zeros(enhancer.latency, np.float32))),
                (960, 333, 1777))
    assert np.abs(y[enhancer.latency:] - whole).max() < 1e-5


def test_attenuation_limit_mixes_the_input_back(tiny_dir) -> None:
    from soundspy.extraction.speech_enhancer import enhance_whole, load_speech_core

    core = load_speech_core(tiny_dir, "cpu", backend="torch")
    x = (0.1 * np.random.default_rng(6).standard_normal(9600)).astype(np.float32)
    plain, limited = enhance_whole(core, x), enhance_whole(core, x, atten_lim_db=6)
    lim = 10 ** (-6 / 20)
    assert np.abs(limited - (lim * x + (1 - lim) * plain)).max() < 1e-6


def test_onnx_backend_matches_torch(tiny_dir) -> None:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from soundspy.extraction.speech_enhancer import SpeechEnhancer, load_speech_core

    torch_core = load_speech_core(tiny_dir, "cpu", backend="torch")
    onnx_core = load_speech_core(tiny_dir, "cpu", backend="onnx")
    assert onnx_core.backend == "onnx" and onnx_core.onnx_file.is_file()
    assert onnx_core.device_name == "cpu (onnx)"
    x = (0.1 * np.random.default_rng(7).standard_normal(24000)).astype(np.float32)
    a = _stream(SpeechEnhancer(torch_core), x, (960,))
    b = _stream(SpeechEnhancer(onnx_core), x, (960, 4096))
    assert np.abs(a - b).max() < 1e-4


def test_missing_model_is_a_user_message(tmp_path) -> None:
    from soundspy.extraction.speech_enhancer import SpeechUnavailable, load_speech_core

    with pytest.raises(SpeechUnavailable, match="soundspy data download deepfilternet"):
        load_speech_core(tmp_path / "nothing", "cpu")
