"""
Test: memory_buffer
Verifies buf[] indexing and memset via a 1-sample delay.
First output sample is (0, 0); thereafter each output is the previous input.
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
    state["buf"] = [0.0, 0.0]  # pre-allocate buffer (EEL2 memory is pre-allocated)
    ns["init"](state)
    return ns["process_block"](signal, state)


def test_memory_buffer_transpiles() -> None:
    py_src = eel2_to_python(EEL2_SRC)
    assert "_eel2_memset" in py_src
    assert 'state.get("buf"' in py_src or "buf" in py_src


def test_first_output_is_zero() -> None:
    signal = np.ones((4, 2))
    out = _run(signal)
    np.testing.assert_allclose(out[0], [0.0, 0.0], atol=TOLERANCE)


def test_output_is_previous_input() -> None:
    signal = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    out = _run(signal)
    # out[0] = [0, 0] (initial buffer)
    # out[1] = [1, 2] (previous sample)
    # out[2] = [3, 4]
    np.testing.assert_allclose(out[0], [0.0, 0.0], atol=TOLERANCE)
    np.testing.assert_allclose(out[1], [1.0, 2.0], atol=TOLERANCE)
    np.testing.assert_allclose(out[2], [3.0, 4.0], atol=TOLERANCE)


def test_delay_matches_numpy_roll() -> None:
    rng = np.random.default_rng(0)
    signal = rng.uniform(-1.0, 1.0, (64, 2))
    out = _run(signal)
    expected = np.roll(signal, 1, axis=0)
    expected[0] = 0.0
    np.testing.assert_allclose(out, expected, atol=TOLERANCE)
