"""
Test: threshold
Verifies hard clipping at a threshold level.
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


def reference_clip(samples: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    out = samples.copy()
    for ch in range(2):
        level = np.abs(out[:, ch])
        mask = level > threshold
        out[mask, ch] *= threshold / level[mask]
    return out


def test_threshold_signal_below_passes_unchanged():
    py_src = eel2_to_python(EEL2_SRC)
    ns = {}
    exec(py_src, ns)

    # Signal well below threshold
    signal = np.full((100, 2), 0.1)
    state = make_state()
    ns["init"](state)
    out = ns["process_block"](signal, state)

    np.testing.assert_allclose(
        out,
        signal,
        atol=TOLERANCE,
        err_msg="Signal below threshold should pass unchanged",
    )


def test_threshold_signal_above_is_clipped():
    py_src = eel2_to_python(EEL2_SRC)
    ns = {}
    exec(py_src, ns)

    # Signal at 2x threshold
    signal = np.full((100, 2), 1.0)
    state = make_state()
    ns["init"](state)
    out = ns["process_block"](signal, state)

    assert np.all(
        np.abs(out) <= 0.5 + TOLERANCE
    ), "Signal above threshold should be clipped to threshold"


def test_threshold_matches_reference():
    py_src = eel2_to_python(EEL2_SRC)
    ns = {}
    exec(py_src, ns)

    t = np.linspace(0, 1, 44100)
    signal = np.column_stack(
        [np.sin(2 * math.pi * 440 * t) * 1.5, np.sin(2 * math.pi * 220 * t) * 0.8]
    )

    state = make_state()
    ns["init"](state)
    transpiled = ns["process_block"](signal, state)
    reference = reference_clip(signal)

    np.testing.assert_allclose(transpiled, reference, atol=TOLERANCE)
