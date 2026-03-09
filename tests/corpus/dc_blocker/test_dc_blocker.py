"""
Test: dc_blocker
Verifies a stateful first-order DC blocking filter:
  y[n] = x[n] - x[n-1] + R * y[n-1],  R = 0.995

Key checks:
  - Pure DC input is blocked (output decays to near zero)
  - AC signal passes with only minor attenuation
  - State persists correctly across samples (matches reference implementation)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from eel2py import eel2_to_python
from eel2py.runtime import make_state

EEL2_SRC = open(os.path.join(os.path.dirname(__file__), "input.eel2")).read()
TOLERANCE = 1e-6
R = 0.995


def reference(samples: np.ndarray) -> np.ndarray:
    """Hand-written DC blocker: y[n] = x[n] - x[n-1] + R * y[n-1]."""
    out = np.zeros_like(samples)
    x_prev = np.zeros(2)
    y_prev = np.zeros(2)
    for i in range(len(samples)):
        out[i] = samples[i] - x_prev + R * y_prev
        x_prev = samples[i].copy()
        y_prev = out[i].copy()
    return out


def _run(signal: np.ndarray) -> np.ndarray:
    py_src = eel2_to_python(EEL2_SRC)
    ns: dict = {}
    exec(py_src, ns)
    state = make_state()
    ns["init"](state)
    return ns["process_block"](signal, state)


def test_dc_blocker_transpiles() -> None:
    py_src = eel2_to_python(EEL2_SRC)
    assert "def process_sample" in py_src
    assert "def init" in py_src


def test_dc_is_blocked() -> None:
    # 1 second of pure DC at 44100 Hz
    dc = np.ones((44100, 2)) * 0.5
    out = _run(dc)
    # After settling, the DC component should be near zero
    assert abs(out[-1, 0]) < 0.01, f"DC not blocked: final output = {out[-1, 0]:.4f}"
    assert abs(out[-1, 1]) < 0.01, f"DC not blocked: final output = {out[-1, 1]:.4f}"


def test_ac_passes_with_low_attenuation() -> None:
    srate = 44100
    t = np.linspace(0, 1.0, srate, endpoint=False)
    freq = 1000.0  # 1 kHz — well above DC blocker cutoff
    ac = np.column_stack([np.sin(2 * np.pi * freq * t)] * 2)
    out = _run(ac)
    # Ignore first 100 samples for filter to settle
    rms_in = np.sqrt(np.mean(ac[100:] ** 2))
    rms_out = np.sqrt(np.mean(out[100:] ** 2))
    ratio = rms_out / rms_in
    assert ratio > 0.99, f"AC signal attenuated too much: ratio={ratio:.4f}"


def test_matches_reference() -> None:
    rng = np.random.default_rng(42)
    signal = rng.uniform(-0.5, 0.5, (512, 2))
    out = _run(signal)
    np.testing.assert_allclose(out, reference(signal), atol=TOLERANCE)
