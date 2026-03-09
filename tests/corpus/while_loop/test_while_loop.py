"""
Test: while_loop
Verifies while(cond)(body) by halving a signal until it falls below threshold=0.5.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from eel2py import eel2_to_python
from eel2py.runtime import make_state

EEL2_SRC = open(os.path.join(os.path.dirname(__file__), "input.eel2")).read()
TOLERANCE = 1e-6


def reference(samples: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Halve each sample while it exceeds threshold."""
    out = samples.copy()
    for i in range(len(out)):
        for ch in range(2):
            while out[i, ch] > threshold:
                out[i, ch] *= 0.5
    return out


def _run(signal: np.ndarray) -> np.ndarray:
    py_src = eel2_to_python(EEL2_SRC)
    ns: dict = {}
    exec(py_src, ns)
    state = make_state()
    ns["init"](state)
    return ns["process_block"](signal, state)


def test_while_loop_transpiles() -> None:
    py_src = eel2_to_python(EEL2_SRC)
    assert "while " in py_src


def test_output_never_exceeds_threshold() -> None:
    signal = np.full((64, 2), 4.0)
    out = _run(signal)
    assert np.all(out <= 0.5 + TOLERANCE), "Output should never exceed threshold"


def test_signal_below_threshold_unchanged() -> None:
    signal = np.full((64, 2), 0.3)
    out = _run(signal)
    np.testing.assert_allclose(out, 0.3, atol=TOLERANCE)


def test_matches_reference() -> None:
    rng = np.random.default_rng(0)
    signal = np.abs(rng.uniform(0.0, 8.0, (256, 2)))
    out = _run(signal)
    np.testing.assert_allclose(out, reference(signal), atol=TOLERANCE)
