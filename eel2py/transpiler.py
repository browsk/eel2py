"""Top-level transpiler API.

Provides a single entry point for converting EEL2 source code to a Python
source string.
"""

from .codegen import CodeGen
from .parser import parse


def eel2_to_python(source: str) -> str:
    """Transpile an EEL2 source string to a Python source string.

    The returned string defines init(), on_slider(), process_sample(), and
    process_block() functions backed by a mutable state dict.

    Example::

        python_src = eel2_to_python(eel2_code)
        ns = {}
        exec(python_src, ns)
        from eel2py.runtime import make_state
        state = make_state()
        ns['init'](state)
        l, r = ns['process_sample'](0.5, 0.5, state)

    Args:
        source: Raw EEL2 source code to transpile.

    Returns:
        A Python source string ready to be exec()'d or written to a .py file.
    """
    ast = parse(source)
    return CodeGen().generate(ast)
