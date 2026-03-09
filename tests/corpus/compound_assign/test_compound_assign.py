"""
Test: compound_assign
Verifies +=, -=, *=, /= operators.
spl0 = (spl0 + 0.1) * 2.0
spl1 = (spl1 - 0.1) / 2.0
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from eel2py import eel2_to_python
from eel2py.runtime import make_state

EEL2_SRC = open(os.path.join(os.path.dirname(__file__), "input.eel2")).read()
TOLERANCE = 1e-6


def reference(samples: np.ndarray) -> np.ndarray:
    out = np.empty_like(samples)
    out[:, 0] = (samples[:, 0] + 0.1) * 2.0
    out[:, 1] = (samples[:, 1] - 0.1) / 2.0
    return out


def _run(signal: np.ndarray) -> np.ndarray:
    py_src = eel2_to_python(EEL2_SRC)
    ns: dict = {}
    exec(py_src, ns)
    state = make_state()
    ns["init"](state)
    return ns["process_block"](signal, state)


def test_compound_assign_transpiles() -> None:
    py_src = eel2_to_python(EEL2_SRC)
    assert "+=" in py_src
    assert "-=" in py_src
    assert "*=" in py_src
    assert "/=" in py_src


def test_plus_equals() -> None:
    signal = np.zeros((64, 2))
    out = _run(signal)
    # spl0: (0 + 0.1) * 2 = 0.2
    np.testing.assert_allclose(out[:, 0], 0.2, atol=TOLERANCE)


def test_minus_equals() -> None:
    signal = np.zeros((64, 2))
    out = _run(signal)
    # spl1: (0 - 0.1) / 2 = -0.05
    np.testing.assert_allclose(out[:, 1], -0.05, atol=TOLERANCE)


def test_matches_reference() -> None:
    rng = np.random.default_rng(0)
    signal = rng.uniform(-0.5, 0.5, (512, 2))
    out = _run(signal)
    np.testing.assert_allclose(out, reference(signal), atol=TOLERANCE)
