"""
Test: loop_count
Verifies loop(n, expr) by accumulating input n=4 times — output should be input * 4.
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
    ns["init"](state)
    return ns["process_block"](signal, state)


def test_loop_count_transpiles() -> None:
    py_src = eel2_to_python(EEL2_SRC)
    assert "for _loop_i in range" in py_src


def test_loop_multiplies_input_by_n() -> None:
    signal = np.full((64, 2), 0.25)
    out = _run(signal)
    np.testing.assert_allclose(out, 1.0, atol=TOLERANCE)


def test_loop_matches_multiply_by_4() -> None:
    rng = np.random.default_rng(0)
    signal = rng.uniform(-0.2, 0.2, (256, 2))
    out = _run(signal)
    # loop accumulates spl0 (left channel) n=4 times; both output channels
    # are set to acc, so both equal signal[:, 0] * 4
    np.testing.assert_allclose(out[:, 0], signal[:, 0] * 4, atol=TOLERANCE)
    np.testing.assert_allclose(out[:, 1], signal[:, 0] * 4, atol=TOLERANCE)


def test_silence_stays_silent() -> None:
    silence = np.zeros((256, 2))
    out = _run(silence)
    np.testing.assert_allclose(out, 0.0, atol=TOLERANCE)
