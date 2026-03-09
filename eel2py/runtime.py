"""EEL2 Runtime.

Stubs for EEL2 built-in functions and globals that have no direct Python
equivalent. Import these into transpiled output or use make_state() to
create the initial state dict.
"""


def _eel2_sign(x: float) -> float:
    """Return the sign of x as -1.0, 0.0, or 1.0.

    Args:
        x: Input value.

    Returns:
        1.0 if x > 0, -1.0 if x < 0, 0.0 if x == 0.
    """
    if x > 0:
        return 1.0
    if x < 0:
        return -1.0
    return 0.0


def _eel2_memset(buf: list, value: float, length: int) -> None:
    """Fill the first length elements of buf with value.

    Args:
        buf: Target list (modified in place).
        value: Fill value.
        length: Number of elements to set.
    """
    for i in range(int(length)):
        buf[i] = float(value)


def _eel2_memcpy(dst: list, src: list, length: int) -> None:
    """Copy length elements from src into dst.

    Args:
        dst: Destination list (modified in place).
        src: Source list.
        length: Number of elements to copy.
    """
    for i in range(int(length)):
        dst[i] = src[i]


def _eel2_assign(state: dict, name: str, value: float) -> float:
    """Store value in state[name] and return it.

    Used when an assignment expression appears in a value context (EEL2
    allows assignment to return the assigned value).

    Args:
        state: The mutable state dictionary.
        name: Variable name to assign.
        value: Value to store.

    Returns:
        The stored value as a float.
    """
    state[name] = float(value)
    return float(value)


def make_state(
    srate: float = 44100.0,
    num_ch: int = 2,
    samplesblock: int = 512,
) -> dict:
    """Create a fresh state dict with EEL2 globals pre-populated.

    Pass the returned dict into init(), on_slider(), and process_sample().

    Args:
        srate: Sample rate in Hz. Defaults to 44100.0.
        num_ch: Number of audio channels. Defaults to 2.
        samplesblock: Block size in samples. Defaults to 512.

    Returns:
        A dict containing srate, num_ch, samplesblock, spl0, and spl1.
    """
    return {
        "srate": srate,
        "num_ch": float(num_ch),
        "samplesblock": float(samplesblock),
        "spl0": 0.0,
        "spl1": 0.0,
    }
