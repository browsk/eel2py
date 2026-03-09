"""EEL2 Tokenizer.

Converts raw EEL2 source text into a flat list of tokens.
"""

import re
from dataclasses import dataclass
from enum import Enum, auto


class TT(Enum):
    """Token type enumeration for all EEL2 lexical elements."""

    NUMBER = auto()
    IDENT = auto()
    STRING = auto()
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    CARET = auto()
    EQ = auto()
    NEQ = auto()
    LT = auto()
    GT = auto()
    LTE = auto()
    GTE = auto()
    ASSIGN = auto()
    PLUS_EQ = auto()
    MINUS_EQ = auto()
    STAR_EQ = auto()
    SLASH_EQ = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    QUESTION = auto()
    COLON = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    SEMICOLON = auto()
    COMMA = auto()
    SECTION = auto()
    EOF = auto()


@dataclass
class Token:
    """A single lexical token with its type, raw value, and source line number.

    Attributes:
        type: The token type.
        value: The raw string matched from source.
        line: The 1-based line number where this token appears.
    """

    type: TT
    value: str
    line: int


_PATTERNS: list[tuple[TT, str]] = [
    (TT.SECTION, r"@[a-z_]+"),
    (TT.NUMBER, r"0x[0-9a-fA-F]+|\d+\.?\d*(?:[eE][+-]?\d+)?"),
    (TT.STRING, r'"[^"]*"'),
    (TT.PLUS_EQ, r"\+="),
    (TT.MINUS_EQ, r"-="),
    (TT.STAR_EQ, r"\*="),
    (TT.SLASH_EQ, r"/="),
    (TT.EQ, r"=="),
    (TT.NEQ, r"!="),
    (TT.LTE, r"<="),
    (TT.GTE, r">="),
    (TT.AND, r"&&"),
    (TT.OR, r"\|\|"),
    (TT.LT, r"<"),
    (TT.GT, r">"),
    (TT.ASSIGN, r"="),
    (TT.PLUS, r"\+"),
    (TT.MINUS, r"-"),
    (TT.STAR, r"\*"),
    (TT.SLASH, r"/"),
    (TT.PERCENT, r"%"),
    (TT.CARET, r"\^"),
    (TT.NOT, r"!"),
    (TT.QUESTION, r"\?"),
    (TT.COLON, r":"),
    (TT.LPAREN, r"\("),
    (TT.RPAREN, r"\)"),
    (TT.LBRACKET, r"\["),
    (TT.RBRACKET, r"\]"),
    (TT.SEMICOLON, r";"),
    (TT.COMMA, r","),
    (TT.IDENT, r"[a-zA-Z_][a-zA-Z0-9_.]*"),
]

_MASTER = re.compile(
    r"//[^\n]*|/\*.*?\*/|[ \t\r\n]+|"
    + "|".join(f"(?P<T{i}>{pat})" for i, (_, pat) in enumerate(_PATTERNS)),
    re.DOTALL,
)


def tokenize(source: str) -> list[Token]:
    """Lex an EEL2 source string into a flat token list.

    Comments and whitespace are discarded. A sentinel EOF token is appended.

    Args:
        source: Raw EEL2 source code.

    Returns:
        List of Token objects ending with a TT.EOF sentinel.
    """
    tokens: list[Token] = []
    line = 1
    for m in _MASTER.finditer(source):
        text = m.group()
        line += text.count("\n")
        if m.lastgroup is None:
            continue
        idx = int(m.lastgroup[1:])
        tt, _ = _PATTERNS[idx]
        tokens.append(Token(tt, text, line))
    tokens.append(Token(TT.EOF, "", line))
    return tokens
