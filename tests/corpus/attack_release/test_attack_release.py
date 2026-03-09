"""
Test: attack_release
Verifies that a single-pole envelope follower behaves correctly.
Key checks:
  - Envelope reaches ~63% of target within one time constant
  - Envelope never exceeds input peak
  - Silence → envelope decays to zero
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from eel2py import eel2_to_python
from eel2py.runtime import make_state

EEL2_SRC = open(os.path.join(os.path.dirname(__file__), "input.eel2")).read()


def _run(signal_mono: np.ndarray, srate: float = 44100.0):
    py_src = eel2_to_python(EEL2_SRC)
    ns = {}
    exec(py_src, ns)

    state = make_state(srate=srate)
    ns["init"](state)
    ns["on_slider"](state)  # compute coefficients

    samples = np.column_stack([signal_mono, signal_mono])
    out = ns["process_block"](samples, state)
    return out[:, 0]  # mono envelope


def test_envelope_reaches_63_percent_within_attack_time():
    srate = 44100.0
    attack_ms = 10.0
    attack_samples = int(srate * attack_ms * 0.001)

    # Step function: silence then full amplitude
    signal = np.zeros(attack_samples * 3)
    signal[attack_samples:] = 1.0

    env = _run(signal, srate)

    # After one attack time constant, envelope should be ~63% of 1.0
    idx = attack_samples + attack_samples  # one TC after step
    assert (
        0.55 <= env[idx] <= 0.75
    ), f"Envelope at one attack TC should be ~63%, got {env[idx]:.3f}"


def test_envelope_never_exceeds_input():
    srate = 44100.0
    t = np.linspace(0, 0.5, int(srate * 0.5))
    signal = np.sin(2 * math.pi * 100 * t)
    env = _run(signal, srate)

    assert np.all(
        env <= np.abs(signal).max() + 1e-6
    ), "Envelope should never exceed input peak"


def test_envelope_decays_on_silence():
    srate = 44100.0
    # 100ms of signal then 500ms of silence
    sig_len = int(srate * 0.1)
    sil_len = int(srate * 0.5)
    signal = np.concatenate([np.ones(sig_len), np.zeros(sil_len)])

    env = _run(signal, srate)

    # Envelope at the end should be close to zero
    assert (
        env[-1] < 0.01
    ), f"Envelope should decay near zero after silence, got {env[-1]:.4f}"
