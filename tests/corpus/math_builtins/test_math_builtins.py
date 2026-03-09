"""
Test: math_builtins
Verifies sin, cos, abs, floor, sign built-in functions.
spl0 = sin(spl0) + cos(spl0)
spl1 = floor(abs(spl1) * 4) / 4 * sign(spl1)
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from eel2py import eel2_to_python
from eel2py.runtime import make_state, _eel2_sign

EEL2_SRC = open(os.path.join(os.path.dirname(__file__), "input.eel2")).read()
TOLERANCE = 1e-6


def reference(samples: np.ndarray) -> np.ndarray:
    out = np.empty_like(samples)
    for i in range(len(samples)):
        x, y = samples[i, 0], samples[i, 1]
        out[i, 0] = math.sin(x) + math.cos(x)
        out[i, 1] = math.floor(abs(y) * 4) / 4 * _eel2_sign(y)
    return out


def _run(signal: np.ndarray) -> np.ndarray:
    py_src = eel2_to_python(EEL2_SRC)
    ns: dict = {}
    exec(py_src, ns)
    state = make_state()
    return ns["process_block"](signal, state)


def test_math_builtins_transpiles() -> None:
    py_src = eel2_to_python(EEL2_SRC)
    assert "math.sin" in py_src
    assert "math.cos" in py_src
    assert "math.floor" in py_src
    assert "_eel2_sign" in py_src


def test_sin_cos_combination() -> None:
    signal = np.zeros((1, 2))
    out = _run(signal)
    # sin(0) + cos(0) = 0 + 1 = 1
    np.testing.assert_allclose(out[0, 0], 1.0, atol=TOLERANCE)


def test_floor_quantisation() -> None:
    signal = np.array([[0.0, 0.37]])
    out = _run(signal)
    # floor(0.37 * 4) / 4 * sign(0.37) = floor(1.48)/4 * 1 = 1/4 = 0.25
    np.testing.assert_allclose(out[0, 1], 0.25, atol=TOLERANCE)


def test_sign_preserves_polarity() -> None:
    pos = np.array([[0.0, 0.6]])
    neg = np.array([[0.0, -0.6]])
    out_pos = _run(pos)
    out_neg = _run(neg)
    assert out_pos[0, 1] > 0, "Positive input should give positive output"
    assert out_neg[0, 1] < 0, "Negative input should give negative output"


def test_matches_reference() -> None:
    rng = np.random.default_rng(0)
    signal = rng.uniform(-1.0, 1.0, (256, 2))
    out = _run(signal)
    np.testing.assert_allclose(out, reference(signal), atol=TOLERANCE)
