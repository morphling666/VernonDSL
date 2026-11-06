from __future__ import annotations

import ast
import itertools
from typing import Protocol

from .lowering_types import DslType, ModuleContext, Value, element_type
from .model import TypedExpression


class NumericEmitter(Protocol):
    context: ModuleContext

    def _typed_expression(self, node: ast.expr) -> TypedExpression: ...

    def _expression(self, node: ast.expr, expected: DslType | None = None) -> Value: ...

    def _coerce_numeric(self, node: ast.AST, value: Value, target: DslType) -> Value: ...

    def _intrinsic(self, node: ast.AST, name: str, arguments: list[Value], result_type: DslType) -> Value: ...

    def _fresh(self) -> str: ...

    def _line(self, text: str) -> None: ...

    def _require_same_type(self, node: ast.AST, expected: DslType, actual: DslType) -> None: ...

    def _default_value(self, node: ast.AST, value_type: DslType) -> Value: ...


def lower_constant(emitter: NumericEmitter, node: ast.Constant, expected: DslType | None) -> Value:
    if isinstance(node.value, bool):
        value_type = DslType("scalar", "bool")
        literal = "1" if node.value else "0"
    elif isinstance(node.value, int):
        value_type = (
            expected
            if expected and expected.kind == "scalar" and (expected.is_float or expected.name in {"i32", "u32"})
            else DslType("scalar", "i32")
        )
        literal = f"{node.value}.0" if value_type.is_float else str(node.value)
    elif isinstance(node.value, float):
        value_type = (
            expected if expected and expected.kind == "scalar" and expected.is_float else DslType("scalar", "f32")
        )
        literal = f"{node.value:.17g}"
        if "." not in literal:
            # MLIR float literals need a fraction, also before an exponent: 1e-10 must print as 1.0e-10.
            mantissa, exponent, power = literal.partition("e")
            literal = f"{mantissa}.0{exponent}{power}"
    else:
        raise emitter.context.error(node, "only bool, integer, and float constants are supported")
    result = emitter._fresh()
    emitter._line(f"{result} = arith.constant {literal} : {value_type.mlir}")
    return Value(result, value_type)


def lower_binary(emitter: NumericEmitter, node: ast.BinOp) -> Value:
    result_type = emitter._typed_expression(node).type
    left = emitter._coerce_numeric(node.left, emitter._expression(node.left), result_type)
    right = emitter._coerce_numeric(node.right, emitter._expression(node.right), result_type)
    floating = left.type.is_float
    unsigned = element_type(left.type).name == "u32"
    if isinstance(node.op, ast.Pow):
        if not floating:
            raise emitter.context.error(node, "power requires floating-point operands")
        return emitter._intrinsic(node, "pow", [left, right], result_type)
    if not (floating or left.type.is_integer):
        raise emitter.context.error(node, f"unsupported binary operation for {left.type.name}")
    if type(node.op) in _BITWISE:
        if floating:
            raise emitter.context.error(node, f"{_BITWISE[type(node.op)]} requires integer operands")
        return Value(_bitwise(emitter, node, left, right).name, result_type)
    if isinstance(node.op, ast.FloorDiv) and floating:
        return Value(_floored_float_quotient(emitter, node, left, right).name, result_type)
    if isinstance(node.op, ast.Mod) and not unsigned:
        if floating:
            return Value(_floored_float_remainder(emitter, node, left, right).name, result_type)
        quotient = _emit(emitter, "arith.floordivsi", (left, right), left.type)
        product = _emit(emitter, "arith.muli", (quotient, right), left.type)
        return Value(_emit(emitter, "arith.subi", (left, product), left.type).name, result_type)
    operations = {
        ast.Add: "arith.addf" if floating else "arith.addi",
        ast.Sub: "arith.subf" if floating else "arith.subi",
        ast.Mult: "arith.mulf" if floating else "arith.muli",
        ast.Div: "arith.divf" if floating else ("arith.divui" if unsigned else "arith.divsi"),
        ast.FloorDiv: "arith.divui" if unsigned else "arith.floordivsi",
        ast.Mod: "arith.remui",
    }
    operation = operations.get(type(node.op))
    if operation is None:
        raise emitter.context.error(node, f"unsupported binary operation for {left.type.name}")
    return Value(_emit(emitter, operation, (left, right), left.type).name, result_type)


_BITWISE = {ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^", ast.LShift: "<<", ast.RShift: ">>"}


def _bitwise(emitter: NumericEmitter, node: ast.BinOp, left: Value, right: Value) -> Value:
    value_type = left.type
    if isinstance(node.op, (ast.BitAnd, ast.BitOr, ast.BitXor)):
        operation = {ast.BitAnd: "arith.andi", ast.BitOr: "arith.ori", ast.BitXor: "arith.xori"}[type(node.op)]
        return _emit(emitter, operation, (left, right), value_type)
    # arith shifts are poison from the bit width on. Host code and numpy define them, so device code does too: a
    # left or unsigned right shift by the width or more, or by a negative amount, is 0, and a signed right shift
    # fills with the sign. The unsigned compare also catches negative amounts.
    scalar = element_type(value_type)
    width = int(scalar.mlir.removeprefix("i"))
    limit = _constant(emitter, width, value_type)
    oversized = _emit(emitter, "arith.cmpi uge,", (right, limit), value_type, _mask_type(value_type))
    if isinstance(node.op, ast.RShift) and not scalar.name.startswith("u"):
        amount = _select(emitter, oversized, _constant(emitter, width - 1, value_type), right)
        return _emit(emitter, "arith.shrsi", (left, amount), value_type)
    operation = "arith.shli" if isinstance(node.op, ast.LShift) else "arith.shrui"
    shifted = _emit(emitter, operation, (left, right), value_type)
    return _select(emitter, oversized, _constant(emitter, 0, value_type), shifted)


def _constant(emitter: NumericEmitter, value: int, value_type: DslType) -> Value:
    literal = f"dense<{value}>" if value_type.kind == "tensor" else str(value)
    result = emitter._fresh()
    emitter._line(f"{result} = arith.constant {literal} : {value_type.mlir}")
    return Value(result, value_type)


def _mask_type(value_type: DslType) -> DslType:
    boolean = DslType("scalar", "bool")
    if value_type.kind == "tensor":
        return DslType("tensor", value_type.name, (boolean, *value_type.arguments[1:]))
    return boolean


def _emit(
    emitter: NumericEmitter,
    operation: str,
    operands: tuple[Value, ...],
    operand_type: DslType,
    result_type: DslType | None = None,
) -> Value:
    result = emitter._fresh()
    emitter._line(f"{result} = {operation} {', '.join(value.name for value in operands)} : {operand_type.mlir}")
    return Value(result, result_type or operand_type)


def _select(emitter: NumericEmitter, condition: Value, true_value: Value, false_value: Value) -> Value:
    types = true_value.type.mlir
    if true_value.type.kind == "tensor":
        types = f"{condition.type.mlir}, {types}"
    result = emitter._fresh()
    emitter._line(f"{result} = arith.select {condition.name}, {true_value.name}, {false_value.name} : {types}")
    return Value(result, true_value.type)


def _floored_float_remainder(emitter: NumericEmitter, node: ast.AST, left: Value, right: Value) -> Value:
    # CPython float_rem: fmod, then shift by the divisor when the signs differ; an exact zero takes the divisor's
    # sign. `une` keeps NaN remainders on the shifting path so they propagate instead of collapsing to zero. The
    # signed zero is built from the divisor's sign bit because Vulkan and D3D12 compilers may treat -0.0 and +0.0
    # constants as interchangeable.
    value_type = left.type
    mask = _mask_type(value_type)
    zero = emitter._default_value(node, value_type)
    remainder = _emit(emitter, "arith.remf", (left, right), value_type)
    nonzero = _emit(emitter, "arith.cmpf une,", (remainder, zero), value_type, mask)
    remainder_negative = _emit(emitter, "arith.cmpf olt,", (remainder, zero), value_type, mask)
    divisor_negative = _emit(emitter, "arith.cmpf olt,", (right, zero), value_type, mask)
    signs_differ = _emit(emitter, "arith.xori", (remainder_negative, divisor_negative), mask)
    shift = _emit(emitter, "arith.andi", (nonzero, signs_differ), mask)
    shifted = _emit(emitter, "arith.addf", (remainder, right), value_type)
    adjusted = _select(emitter, shift, shifted, remainder)
    return _select(emitter, nonzero, adjusted, _zero_with_sign_of(emitter, right))


def _floored_float_quotient(emitter: NumericEmitter, node: ast.AST, left: Value, right: Value) -> Value:
    # numpy npy_divmod, which CPython float_floor_div matches except that a zero divisor gives left / right instead
    # of raising: (left - fmod(left, right)) / right, less one when the remainder's sign differs from the divisor's,
    # then floored, rounding up when the inexact division left it more than half below the next integer. A zero
    # quotient takes the sign of left / right, built from its sign bit as in _floored_float_remainder.
    value_type = left.type
    mask = _mask_type(value_type)
    zero = emitter._default_value(node, value_type)
    one = _constant(emitter, 1.0, value_type)
    quotient = _emit(emitter, "arith.divf", (left, right), value_type)
    remainder = _emit(emitter, "arith.remf", (left, right), value_type)
    exact = _emit(
        emitter, "arith.divf", (_emit(emitter, "arith.subf", (left, remainder), value_type), right), value_type
    )
    nonzero = _emit(emitter, "arith.cmpf une,", (remainder, zero), value_type, mask)
    remainder_negative = _emit(emitter, "arith.cmpf olt,", (remainder, zero), value_type, mask)
    divisor_negative = _emit(emitter, "arith.cmpf olt,", (right, zero), value_type, mask)
    signs_differ = _emit(emitter, "arith.xori", (remainder_negative, divisor_negative), mask)
    shift = _emit(emitter, "arith.andi", (nonzero, signs_differ), mask)
    exact = _select(emitter, shift, _emit(emitter, "arith.subf", (exact, one), value_type), exact)
    floored = _emit(emitter, "math.floor", (exact,), value_type)
    fraction = _emit(emitter, "arith.subf", (exact, floored), value_type)
    rounds_up = _emit(emitter, "arith.cmpf ogt,", (fraction, _constant(emitter, 0.5, value_type)), value_type, mask)
    floored = _select(emitter, rounds_up, _emit(emitter, "arith.addf", (floored, one), value_type), floored)
    nonzero_quotient = _emit(emitter, "arith.cmpf une,", (exact, zero), value_type, mask)
    result = _select(emitter, nonzero_quotient, floored, _zero_with_sign_of(emitter, quotient))
    divisor_zero = _emit(emitter, "arith.cmpf oeq,", (right, zero), value_type, mask)
    return _select(emitter, divisor_zero, quotient, result)


def integer_extremum(emitter: NumericEmitter, minimum: bool, left: Value, right: Value) -> Value:
    """Elementwise min or max of integers; a tie picks `right`, as host code does. MLIR integers are signless, so the
    signedness comes from the DSL type."""

    unsigned = element_type(left.type).name == "u32"
    predicate = ("ult" if unsigned else "slt") if minimum else ("ugt" if unsigned else "sgt")
    picked = _emit(emitter, f"arith.cmpi {predicate},", (left, right), left.type, _mask_type(left.type))
    return _select(emitter, picked, left, right)


def reduce_bool(emitter: NumericEmitter, any_true: bool, value: Value, constant) -> Value:
    """any() or all() of a bool or a bool Tensor."""

    boolean = DslType("scalar", "bool")
    if value.type.kind != "tensor":
        return Value(value.name, boolean)
    extents = [int(extent) for extent in value.type.arguments[1:]]  # type: ignore[arg-type]
    combined: Value | None = None
    for coordinate in itertools.product(*(range(extent) for extent in extents)):
        element = emitter._fresh()
        indices = ", ".join(constant(index) for index in coordinate)
        emitter._line(f"{element} = tensor.extract {value.name}[{indices}] : {value.type.mlir}")
        item = Value(element, boolean)
        combined = (
            item
            if combined is None
            else _emit(emitter, "arith.ori" if any_true else "arith.andi", (combined, item), boolean)
        )
    assert combined is not None
    return combined


def _zero_with_sign_of(emitter: NumericEmitter, value: Value) -> Value:
    float_mlir = element_type(value.type).mlir
    bits = {"f16": 16, "f32": 32, "f64": 64}[float_mlir]
    integer_mlir = f"i{bits}"
    sign_literal = str(-(1 << (bits - 1)))
    if value.type.kind == "tensor":
        integer_mlir = value.type.mlir.removesuffix(f"{float_mlir}>") + f"{integer_mlir}>"
        sign_literal = f"dense<{sign_literal}>"
    sign_mask, value_bits, sign_bits, result = (emitter._fresh() for _ in range(4))
    emitter._line(f"{sign_mask} = arith.constant {sign_literal} : {integer_mlir}")
    emitter._line(f"{value_bits} = arith.bitcast {value.name} : {value.type.mlir} to {integer_mlir}")
    emitter._line(f"{sign_bits} = arith.andi {value_bits}, {sign_mask} : {integer_mlir}")
    emitter._line(f"{result} = arith.bitcast {sign_bits} : {integer_mlir} to {value.type.mlir}")
    return Value(result, value.type)


def lower_unary(emitter: NumericEmitter, node: ast.UnaryOp) -> Value:
    operand = emitter._expression(node.operand)
    result_type = emitter._typed_expression(node).type
    result = emitter._fresh()
    if isinstance(node.op, ast.Not):
        emitter._require_same_type(node, DslType("scalar", "bool"), operand.type)
        truth = emitter._fresh()
        emitter._line(f"{truth} = arith.constant true")
        emitter._line(f"{result} = arith.xori {operand.name}, {truth} : i1")
        return Value(result, result_type)
    if isinstance(node.op, ast.USub) and (operand.type.is_float or operand.type.is_integer):
        zero = emitter._default_value(node, operand.type)
        operation = "arith.subf" if operand.type.is_float else "arith.subi"
        emitter._line(f"{result} = {operation} {zero.name}, {operand.name} : {operand.type.mlir}")
        return Value(result, result_type)
    if isinstance(node.op, ast.UAdd):
        return Value(operand.name, result_type, operand.fields, operand.access)
    if isinstance(node.op, ast.Invert):
        if not operand.type.is_integer:
            raise emitter.context.error(node, "~ requires integer operands")
        ones = _constant(emitter, -1, operand.type)
        return Value(_emit(emitter, "arith.xori", (operand, ones), operand.type).name, result_type)
    raise emitter.context.error(node, "unsupported unary operation")


def lower_compare(emitter: NumericEmitter, node: ast.Compare) -> Value:
    if len(node.ops) != 1:
        raise emitter.context.error(node, "chained comparisons are not supported")
    typed = emitter._typed_expression(node)
    if len(typed.operand_types) != 2:
        raise emitter.context.error(node, "internal error: comparison has no typed operands")
    common = typed.operand_types[0]
    left = emitter._coerce_numeric(node.left, emitter._expression(node.left), common)
    right = emitter._coerce_numeric(node.comparators[0], emitter._expression(node.comparators[0]), common)
    predicates = {
        ast.Eq: ("oeq", "eq", "eq"),
        ast.NotEq: ("one", "ne", "ne"),
        ast.Lt: ("olt", "slt", "ult"),
        ast.LtE: ("ole", "sle", "ule"),
        ast.Gt: ("ogt", "sgt", "ugt"),
        ast.GtE: ("oge", "sge", "uge"),
    }
    selected = predicates.get(type(node.ops[0]))
    if selected is None:
        raise emitter.context.error(node, f"unsupported comparison: {type(node.ops[0]).__name__}")
    floating_predicate, signed_predicate, unsigned_predicate = selected
    integer_predicate = unsigned_predicate if element_type(left.type).name == "u32" else signed_predicate
    if left.type.is_float:
        operation, predicate = "arith.cmpf", floating_predicate
    elif left.type.is_integer or left.type.name == "bool":
        operation, predicate = "arith.cmpi", integer_predicate
    else:
        raise emitter.context.error(node, "comparison requires scalar or tensor numeric operands")
    result = emitter._fresh()
    emitter._line(f"{result} = {operation} {predicate}, {left.name}, {right.name} : {left.type.mlir}")
    return Value(result, typed.type)
