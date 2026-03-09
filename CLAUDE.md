# eel2py — EEL2 to Python Transpiler

## What this project does
Transpiles EEL2 (Reaper JSFX scripting language) into runnable Python/numpy code.
Primary use case: offline testing of JSFX DSP logic without running Reaper.

## Scope

### In scope (v1)
- `@init`, `@slider`, `@sample` sections
- Math expressions: `+`, `-`, `*`, `/`, `%`, `^` (power)
- Comparisons: `==`, `!=`, `<`, `>`, `<=`, `>=`
- Logical: `&&`, `||`, `!`
- Assignment: `=`, `+=`, `-=`, `*=`, `/=`
- Conditionals: `if/else`, ternary `(cond ? a : b)`
- Loops: `while(cond)`, `loop(count, body)`
- Memory buffers: `buf[index]`, `memset`, `memcpy`
- Built-in math: `sin`, `cos`, `tan`, `sqrt`, `abs`, `log`, `exp`, `floor`, `ceil`, `min`, `max`, `sign`
- Predefined constants: `$pi`, `$e`, `$phi`
- EEL2 globals: `srate`, `num_ch`, `samplesblock`

### Out of scope (v1)
- `@gfx` (graphics/UI)
- `@midi` (MIDI processing)
- `@serialize` (state saving)
- File I/O (`fopen`, `fread`, etc.)

## Architecture
```
eel2py/
  tokenizer.py   — text → token stream
  parser.py      — token stream → AST (dataclasses)
  codegen.py     — AST → Python source string
  runtime.py     — EEL2 built-ins and globals stub
  transpiler.py  — top-level API: eel2_to_python(source: str) -> str
```

## Output contract
Every transpiled `@sample` section becomes:

```python
def process_sample(left: float, right: float, state: dict) -> tuple[float, float]:
    # transpiled logic
    return left, right
```

And a block processor wrapper:

```python
def process_block(samples: np.ndarray, state: dict) -> np.ndarray:
    out = np.zeros_like(samples)
    for i in range(len(samples)):
        l, r = process_sample(samples[i, 0], samples[i, 1], state)
        out[i] = [l, r]
    return out
```

## Test approach
Each test in `tests/corpus/` has:
- `input.eel2` — the EEL2 source
- `expected.py` — hand-written reference Python
- `test.py` — runs both against the same signal and compares numerically (tolerance: 1e-6)

Run all tests: `pytest tests/`

## Key EEL2 gotchas to handle
- `=` is assignment; `==` is comparison (unlike C)
- `loop(count, expr)` runs expr count times — map to a Python for loop
- Variables are global by default — use state dict to track them
- `^` is power, not XOR
- Uninitialized variables default to 0.0
- `$pi`, `$e`, `$phi` are constants (not variables) — they map to `math.pi`, `math.e`, and the golden ratio
