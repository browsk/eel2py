"""
Test: unary_ops
Verifies unary negation (-) and logical not (!).
With mute=0: spl0 = !0 ? -spl0 : 0  =>  spl0 = -spl0
             spl1 = -spl1
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


def test_unary_ops_transpiles() -> None:
    py_src = eel2_to_python(EEL2_SRC)
    assert "not " in py_src
    assert "(-state" in py_src  # unary negation: (-state.get(...))


def test_negation_inverts_signal() -> None:
    signal = np.full((64, 2), 0.5)
    out = _run(signal)
    np.testing.assert_allclose(out, -0.5, atol=TOLERANCE)


def test_negation_of_negative_is_positive() -> None:
    signal = np.full((64, 2), -0.3)
    out = _run(signal)
    np.testing.assert_allclose(out, 0.3, atol=TOLERANCE)


def test_matches_reference() -> None:
    rng = np.random.default_rng(0)
    signal = rng.uniform(-1.0, 1.0, (512, 2))
    out = _run(signal)
    np.testing.assert_allclose(out, -signal, atol=TOLERANCE)
