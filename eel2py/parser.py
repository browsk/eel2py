"""EEL2 Parser.

Recursive-descent parser that converts a token stream into an AST
composed of dataclass nodes.
"""

from __future__ import annotations

from dataclasses import dataclass

from .tokenizer import TT, Token, tokenize


@dataclass
class Program:
    """Root AST node representing a complete EEL2 source file.

    Attributes:
        sections: Ordered list of Section nodes (@init, @slider, @sample, etc.).
    """

    sections: list[Section]


@dataclass
class Section:
    """An EEL2 section block (e.g. @init, @slider, @sample).

    Attributes:
        name: Section name without the leading '@'.
        body: List of statement nodes in the section body.
    """

    name: str
    body: list


@dataclass
class Assign:
    """Assignment expression: target op= value.

    Attributes:
        target: The left-hand side (Ident or Index node).
        op: Operator string: '=', '+=', '-=', '*=', or '/='.
        value: The right-hand side expression node.
    """

    target: object
    op: str
    value: object


@dataclass
class If:
    """Conditional statement: if (cond) then [else else_].

    Attributes:
        cond: Condition expression node.
        then: List of statement nodes for the true branch.
        else_: List of statement nodes for the false branch (may be empty).
    """

    cond: object
    then: list
    else_: list


@dataclass
class While:
    """While loop: while (cond) body.

    Attributes:
        cond: Loop condition expression node.
        body: List of statement nodes executed while cond is truthy.
    """

    cond: object
    body: list


@dataclass
class Loop:
    """Fixed-count loop: loop(count, expr).

    Attributes:
        count: Expression evaluating to the iteration count.
        body: Single expression node executed count times.
    """

    count: object
    body: object


@dataclass
class BinOp:
    """Binary operation expression.

    Attributes:
        op: Operator string (e.g. '+', '==', '&&').
        left: Left operand node.
        right: Right operand node.
    """

    op: str
    left: object
    right: object


@dataclass
class UnaryOp:
    """Unary operation expression.

    Attributes:
        op: Operator string: '-' or '!'.
        operand: Operand expression node.
    """

    op: str
    operand: object


@dataclass
class Ternary:
    """Ternary expression: cond ? then : else_.

    Attributes:
        cond: Condition expression node.
        then: Value when condition is truthy.
        else_: Value when condition is falsy.
    """

    cond: object
    then: object
    else_: object


@dataclass
class Call:
    """Function call expression.

    Attributes:
        name: Function name as a string.
        args: List of argument expression nodes.
    """

    name: str
    args: list


@dataclass
class Index:
    """Array index expression: array[idx].

    Attributes:
        array: The array expression node.
        idx: The index expression node.
    """

    array: object
    idx: object


@dataclass
class Ident:
    """Identifier (variable reference).

    Attributes:
        name: Variable name string.
    """

    name: str


@dataclass
class Number:
    """Numeric literal.

    Attributes:
        value: Parsed float value.
    """

    value: float


@dataclass
class StringLit:
    """String literal (without surrounding quotes).

    Attributes:
        value: String content.
    """

    value: str


class ParseError(Exception):
    """Raised when the parser encounters unexpected syntax."""


class Parser:
    """Recursive-descent parser for EEL2 source code.

    Args:
        tokens: Flat token list produced by the tokenizer.
    """

    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token:
        """Return the current token without consuming it."""
        return self.tokens[self.pos]

    def advance(self) -> Token:
        """Consume and return the current token."""
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def expect(self, tt: TT) -> Token:
        """Consume the current token, raising ParseError if it isn't tt.

        Args:
            tt: Expected token type.

        Returns:
            The consumed token.

        Raises:
            ParseError: If the current token type doesn't match tt.
        """
        t = self.advance()
        if t.type != tt:
            raise ParseError(
                f"Line {t.line}: expected {tt}, got {t.type} ({t.value!r})"
            )
        return t

    def match(self, *types: TT) -> bool:
        """Return True if the current token type is in types.

        Args:
            *types: One or more TT values to test against.

        Returns:
            True if current token matches any of the given types.
        """
        return self.peek().type in types

    def parse(self) -> Program:
        """Parse the full token stream into a Program AST node.

        Returns:
            The root Program node.
        """
        sections = []
        while not self.match(TT.EOF):
            if self.match(TT.SECTION):
                sections.append(self.parse_section())
            else:
                stmts = self.parse_block_until(TT.SECTION, TT.EOF)
                if stmts:
                    sections.insert(0, Section("init", stmts))
        return Program(sections)

    def parse_section(self) -> Section:
        """Parse a single @section block.

        Returns:
            A Section node with the section name and its body statements.
        """
        tok = self.advance()
        name = tok.value[1:]
        body = self.parse_block_until(TT.SECTION, TT.EOF)
        return Section(name, body)

    def parse_block_until(self, *stop_types: TT) -> list:
        """Parse statements until a token of one of stop_types is seen.

        Args:
            *stop_types: Token types that terminate the block.

        Returns:
            List of parsed statement nodes (None entries dropped).
        """
        stmts = []
        while not self.match(*stop_types):
            stmt = self.parse_statement()
            if stmt is not None:
                stmts.append(stmt)
        return stmts

    def parse_statement(self) -> object:
        """Parse a single statement.

        Returns:
            An AST node, or None if the statement was an empty semicolon.
        """
        while self.match(TT.SEMICOLON):
            self.advance()
            return None

        t = self.peek()
        if t.type == TT.IDENT and t.value == "if":
            return self.parse_if()
        if t.type == TT.IDENT and t.value == "while":
            return self.parse_while()
        if t.type == TT.IDENT and t.value == "loop":
            return self.parse_loop()

        expr = self.parse_expr()
        if self.match(TT.SEMICOLON):
            self.advance()
        return expr

    def parse_if(self) -> If:
        """Parse an if/else statement.

        Returns:
            An If node.
        """
        self.advance()
        self.expect(TT.LPAREN)
        cond = self.parse_expr()
        self.expect(TT.RPAREN)
        then = self.parse_block_body()
        else_: list = []
        if self.match(TT.IDENT) and self.peek().value == "else":
            self.advance()
            else_ = self.parse_block_body()
        return If(cond, then, else_)

    def parse_while(self) -> While:
        """Parse a while loop.

        Returns:
            A While node.
        """
        self.advance()
        self.expect(TT.LPAREN)
        cond = self.parse_expr()
        self.expect(TT.RPAREN)
        body = self.parse_block_body()
        return While(cond, body)

    def parse_loop(self) -> Loop:
        """Parse a loop(count, expr) construct.

        Returns:
            A Loop node.
        """
        self.advance()
        self.expect(TT.LPAREN)
        count = self.parse_expr()
        self.expect(TT.COMMA)
        body = self.parse_expr()
        self.expect(TT.RPAREN)
        return Loop(count, body)

    def parse_block_body(self) -> list:
        """Parse a block body: either a parenthesised sequence or a single statement.

        Returns:
            List of statement nodes.
        """
        if self.match(TT.LPAREN):
            self.advance()
            stmts = []
            while not self.match(TT.RPAREN):
                s = self.parse_statement()
                if s is not None:
                    stmts.append(s)
            self.expect(TT.RPAREN)
            return stmts
        else:
            s = self.parse_statement()
            return [s] if s is not None else []

    def parse_expr(self) -> object:
        """Parse an expression (entry point for expression parsing).

        Returns:
            An expression AST node.
        """
        return self.parse_assign()

    def parse_assign(self) -> object:
        """Parse an assignment expression.

        Returns:
            An Assign node or a lower-precedence expression node.
        """
        left = self.parse_ternary()
        ops = {
            TT.ASSIGN: "=",
            TT.PLUS_EQ: "+=",
            TT.MINUS_EQ: "-=",
            TT.STAR_EQ: "*=",
            TT.SLASH_EQ: "/=",
        }
        if self.peek().type in ops:
            op = ops[self.advance().type]
            right = self.parse_assign()
            return Assign(left, op, right)
        return left

    def parse_ternary(self) -> object:
        """Parse a ternary expression: cond ? then : else_.

        Returns:
            A Ternary node or a lower-precedence expression node.
        """
        cond = self.parse_or()
        if self.match(TT.QUESTION):
            self.advance()
            then = self.parse_expr()
            self.expect(TT.COLON)
            else_ = self.parse_expr()
            return Ternary(cond, then, else_)
        return cond

    def parse_or(self) -> object:
        """Parse a logical OR expression.

        Returns:
            A BinOp('||') node or a lower-precedence expression node.
        """
        left = self.parse_and()
        while self.match(TT.OR):
            self.advance()
            left = BinOp("||", left, self.parse_and())
        return left

    def parse_and(self) -> object:
        """Parse a logical AND expression.

        Returns:
            A BinOp('&&') node or a lower-precedence expression node.
        """
        left = self.parse_compare()
        while self.match(TT.AND):
            self.advance()
            left = BinOp("&&", left, self.parse_compare())
        return left

    def parse_compare(self) -> object:
        """Parse a comparison expression.

        Returns:
            A BinOp node for a comparison operator, or a lower-precedence node.
        """
        left = self.parse_add()
        ops = {
            TT.EQ: "==",
            TT.NEQ: "!=",
            TT.LT: "<",
            TT.GT: ">",
            TT.LTE: "<=",
            TT.GTE: ">=",
        }
        while self.peek().type in ops:
            op = ops[self.advance().type]
            left = BinOp(op, left, self.parse_add())
        return left

    def parse_add(self) -> object:
        """Parse an additive expression.

        Returns:
            A BinOp('+'/'-') node or a lower-precedence expression node.
        """
        left = self.parse_mul()
        while self.match(TT.PLUS, TT.MINUS):
            op = self.advance().value
            left = BinOp(op, left, self.parse_mul())
        return left

    def parse_mul(self) -> object:
        """Parse a multiplicative expression.

        Returns:
            A BinOp('*'/'/'/'%') node or a lower-precedence expression node.
        """
        left = self.parse_power()
        while self.match(TT.STAR, TT.SLASH, TT.PERCENT):
            op = self.advance().value
            left = BinOp(op, left, self.parse_power())
        return left

    def parse_power(self) -> object:
        """Parse a power expression (right-associative).

        Returns:
            A BinOp('^') node or a lower-precedence expression node.
        """
        left = self.parse_unary()
        if self.match(TT.CARET):
            self.advance()
            right = self.parse_power()
            return BinOp("^", left, right)
        return left

    def parse_unary(self) -> object:
        """Parse a unary expression.

        Returns:
            A UnaryOp('-'/'!') node or a postfix expression node.
        """
        if self.match(TT.MINUS):
            self.advance()
            return UnaryOp("-", self.parse_unary())
        if self.match(TT.NOT):
            self.advance()
            return UnaryOp("!", self.parse_unary())
        return self.parse_postfix()

    def parse_postfix(self) -> object:
        """Parse a postfix expression (array indexing).

        Returns:
            An Index node or a primary expression node.
        """
        node = self.parse_primary()
        while self.match(TT.LBRACKET):
            self.advance()
            idx = self.parse_expr()
            self.expect(TT.RBRACKET)
            node = Index(node, idx)
        return node

    def parse_primary(self) -> object:
        """Parse a primary expression: literal, identifier, call, or parenthesised expr.

        Returns:
            A Number, StringLit, Ident, Call, or parenthesised expression node.

        Raises:
            ParseError: If no valid primary expression is found.
        """
        t = self.peek()

        if t.type == TT.NUMBER:
            self.advance()
            v = t.value
            return Number(float(int(v, 16)) if v.startswith("0x") else float(v))

        if t.type == TT.STRING:
            self.advance()
            return StringLit(t.value[1:-1])

        if t.type == TT.LPAREN:
            self.advance()
            expr = self.parse_expr()
            self.expect(TT.RPAREN)
            return expr

        if t.type == TT.IDENT:
            self.advance()
            if self.match(TT.LPAREN):
                self.advance()
                args = []
                while not self.match(TT.RPAREN):
                    args.append(self.parse_expr())
                    if self.match(TT.COMMA):
                        self.advance()
                self.expect(TT.RPAREN)
                return Call(t.value, args)
            return Ident(t.value)

        raise ParseError(f"Line {t.line}: unexpected token {t.type} ({t.value!r})")


def parse(source: str) -> Program:
    """Tokenize and parse an EEL2 source string into a Program AST.

    Args:
        source: Raw EEL2 source code.

    Returns:
        The root Program AST node.
    """
    tokens = tokenize(source)
    return Parser(tokens).parse()
