"""
Test: basic_math
Verifies that a simple gain stage transpiles and runs correctly.
Reference: multiply both channels by 0.5
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


def reference_process(samples: np.ndarray) -> np.ndarray:
    """Hand-written Python equivalent of input.eel2"""
    out = samples.copy()
    out[:, 0] *= 0.5
    out[:, 1] *= 0.5
    return out


def test_basic_math_transpiles():
    py_src = eel2_to_python(EEL2_SRC)
    assert "def process_sample" in py_src
    assert "def init" in py_src


def test_basic_math_output_matches_reference():
    py_src = eel2_to_python(EEL2_SRC)
    ns = {}
    exec(py_src, ns)

    # Build a test signal: 1 second of 440Hz sine, stereo
    srate = 44100
    t = np.linspace(0, 1, srate)
    signal = np.column_stack(
        [
            np.sin(2 * math.pi * 440 * t),
            np.sin(2 * math.pi * 440 * t) * 0.8,
        ]
    )

    state = make_state(srate=srate)
    ns["init"](state)

    transpiled = ns["process_block"](signal, state)
    reference = reference_process(signal)

    np.testing.assert_allclose(
        transpiled,
        reference,
        atol=TOLERANCE,
        err_msg="Transpiled output does not match reference",
    )


def test_silence_passes_through_as_silence():
    py_src = eel2_to_python(EEL2_SRC)
    ns = {}
    exec(py_src, ns)

    silence = np.zeros((1024, 2))
    state = make_state()
    ns["init"](state)
    out = ns["process_block"](silence, state)

    assert np.allclose(out, 0.0), "Silence should remain silence"
