"""
Test: logical_ops
Verifies && and || operators via a stereo noise gate.
Gate opens only when BOTH channels exceed threshold=0.3 in magnitude.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from eel2py import eel2_to_python
from eel2py.runtime import make_state

EEL2_SRC = open(os.path.join(os.path.dirname(__file__), "input.eel2")).read()
TOLERANCE = 1e-6


def reference(samples: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    out = samples.copy()
    above_l = (samples[:, 0] > threshold) | (samples[:, 0] < -threshold)
    above_r = (samples[:, 1] > threshold) | (samples[:, 1] < -threshold)
    gate_open = above_l & above_r
    out[~gate_open, 0] = 0.0
    out[~gate_open, 1] = 0.0
    return out


def _run(signal: np.ndarray) -> np.ndarray:
    py_src = eel2_to_python(EEL2_SRC)
    ns: dict = {}
    exec(py_src, ns)
    state = make_state()
    ns["init"](state)
    return ns["process_block"](signal, state)


def test_logical_ops_transpiles() -> None:
    py_src = eel2_to_python(EEL2_SRC)
    assert " and " in py_src
    assert " or " in py_src


def test_gate_closed_when_both_below_threshold() -> None:
    signal = np.full((64, 2), 0.1)
    out = _run(signal)
    np.testing.assert_allclose(out, 0.0, atol=TOLERANCE)


def test_gate_closed_when_one_channel_below_threshold() -> None:
    signal = np.column_stack([np.full(64, 0.8), np.full(64, 0.1)])
    out = _run(signal)
    np.testing.assert_allclose(out, 0.0, atol=TOLERANCE)


def test_gate_open_when_both_above_threshold() -> None:
    signal = np.full((64, 2), 0.8)
    out = _run(signal)
    np.testing.assert_allclose(out, 0.8, atol=TOLERANCE)


def test_matches_reference() -> None:
    rng = np.random.default_rng(0)
    signal = rng.uniform(-1.0, 1.0, (512, 2))
    out = _run(signal)
    np.testing.assert_allclose(out, reference(signal), atol=TOLERANCE)
