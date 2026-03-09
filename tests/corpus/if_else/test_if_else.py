"""
Test: if_else
Verifies multi-statement if/else blocks via a full-wave rectifier with gain.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from eel2py import eel2_to_python
from eel2py.runtime import make_state

EEL2_SRC = open(os.path.join(os.path.dirname(__file__), "input.eel2")).read()
TOLERANCE = 1e-6


def reference(samples: np.ndarray, gain: float = 2.0) -> np.ndarray:
    """Hand-written reference: full-wave rectify then apply gain."""
    return np.abs(samples) * gain


def _run(signal: np.ndarray) -> np.ndarray:
    py_src = eel2_to_python(EEL2_SRC)
    ns: dict = {}
    exec(py_src, ns)
    state = make_state()
    ns["init"](state)
    return ns["process_block"](signal, state)


def test_if_else_transpiles() -> None:
    py_src = eel2_to_python(EEL2_SRC)
    assert "def process_sample" in py_src
    assert "if " in py_src
    assert "else:" in py_src


def test_positive_signal_passes_through_scaled() -> None:
    signal = np.full((64, 2), 0.3)
    out = _run(signal)
    np.testing.assert_allclose(out, 0.6, atol=TOLERANCE)


def test_negative_signal_is_inverted_and_scaled() -> None:
    signal = np.full((64, 2), -0.3)
    out = _run(signal)
    np.testing.assert_allclose(out, 0.6, atol=TOLERANCE)


def test_output_is_always_non_negative() -> None:
    rng = np.random.default_rng(0)
    signal = rng.uniform(-1.0, 1.0, (1024, 2))
    out = _run(signal)
    assert np.all(out >= -TOLERANCE), "Output should always be non-negative"


def test_matches_reference() -> None:
    rng = np.random.default_rng(1)
    signal = rng.uniform(-1.0, 1.0, (512, 2))
    out = _run(signal)
    np.testing.assert_allclose(out, reference(signal), atol=TOLERANCE)
