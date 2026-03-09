"""
Test: constants
Verifies $pi and $e predefined constants.
spl0 = spl0 * $pi
spl1 = spl1 * $e
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from eel2py import eel2_to_python
from eel2py.runtime import make_state

EEL2_SRC = open(os.path.join(os.path.dirname(__file__), "input.eel2")).read()
TOLERANCE = 1e-6


def _run(signal: np.ndarray) -> np.ndarray:
    py_src = eel2_to_python(EEL2_SRC)
    ns: dict = {}
    exec(py_src, ns)
    state = make_state()
    return ns["process_block"](signal, state)


def test_constants_transpiles() -> None:
    py_src = eel2_to_python(EEL2_SRC)
    assert "math.pi" in py_src
    assert "math.e" in py_src


def test_pi_scales_left_channel() -> None:
    signal = np.ones((1, 2))
    out = _run(signal)
    np.testing.assert_allclose(out[0, 0], math.pi, atol=TOLERANCE)


def test_e_scales_right_channel() -> None:
    signal = np.ones((1, 2))
    out = _run(signal)
    np.testing.assert_allclose(out[0, 1], math.e, atol=TOLERANCE)


def test_matches_reference() -> None:
    rng = np.random.default_rng(0)
    signal = rng.uniform(-1.0, 1.0, (256, 2))
    out = _run(signal)
    np.testing.assert_allclose(out[:, 0], signal[:, 0] * math.pi, atol=TOLERANCE)
    np.testing.assert_allclose(out[:, 1], signal[:, 1] * math.e, atol=TOLERANCE)
