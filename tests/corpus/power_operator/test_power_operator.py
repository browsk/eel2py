"""
Test: power_operator
Verifies that EEL2 ^ maps to Python ** (not XOR).
spl0 = spl0 ^ 3  (cube)
spl1 = spl1 ^ 2  (square)
"""

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


def test_power_operator_transpiles() -> None:
    py_src = eel2_to_python(EEL2_SRC)
    assert "**" in py_src
    assert "^" not in py_src.split('"""')[2]  # not in code after docstring


def test_cube() -> None:
    signal = np.full((32, 2), 2.0)
    out = _run(signal)
    np.testing.assert_allclose(out[:, 0], 8.0, atol=TOLERANCE)


def test_square() -> None:
    signal = np.full((32, 2), 3.0)
    out = _run(signal)
    np.testing.assert_allclose(out[:, 1], 9.0, atol=TOLERANCE)


def test_matches_reference() -> None:
    rng = np.random.default_rng(0)
    signal = rng.uniform(-1.0, 1.0, (256, 2))
    out = _run(signal)
    np.testing.assert_allclose(out[:, 0], signal[:, 0] ** 3, atol=TOLERANCE)
    np.testing.assert_allclose(out[:, 1], signal[:, 1] ** 2, atol=TOLERANCE)
