from __future__ import annotations

import ast
import copy
import hashlib
from collections.abc import Callable
from dataclasses import replace

from ..diagnostics import copy_location, fix_missing_locations
from ..language.ast_utils import (
    assigned_names,
    decorator_name,
    dotted_name,
    rectangular_literal,
    tensor_update_parts,
)
from ..language.slicing import (
    AxisIndex,
    AxisRange,
    AxisSelection,
    AxisWindow,
    SelectionError,
    constant_integer,
    is_slicing,
    plan_selection,
    selection_shape,
)
from ..language.stage_registry import ENTRY_DECORATOR_STAGES, ENTRY_DECORATORS, GRAPHICS_STAGES
from ..shader_contracts import (
    ATOMIC_OPERATION_NAMES,
    FRAGMENT_DISCARD_OPERATION,
    GENERATED_INTERFACE_CONTRACTS,
    TypeContract,
    resource_stage_error,
    texture_bias_sampling_contract,
    texture_sampling_contract,
)
from ..struct_methods import method_tables
from .abi import workgroup_physical_bytes
from .activity import analyze_storage_activity
from .model import (
    HOST_STORAGE_ELEMENTS,
    HOST_STORAGE_MAX_RANK,
    AccessMode,
    AtomicEffect,
    BarrierEffect,
    BranchMerge,
    ConcreteType,
    EffectScope,
    FragmentDiscardEffect,
    InterfaceMetadata,
    LiteralType,
    LValue,
    MemoryOrdering,
    ResourceEffect,
    StorageEffect,
    StorageEffectKind,
    StorageOwner,
    StorageOwnerKind,
    StorageRegion,
    StorageRegionKind,
    Termination,
    TypedEffect,
    TypedExpression,
    TypedFunctionInstance,
    TypedParameter,
    TypedStatement,
    is_abi_stable_value,
    resource_sampler_modes,
)
from .tensor_shapes import matmul_shape
from .texture_dimensions import SAMPLE_COORDINATE_RANK, SIZE_RANK, TEXEL_COORDINATE_RANK
from .type_solver import (
    InferenceType,
    can_convert,
    common_type,
    contextualize,
    default_type,
    describe,
    element_type,
    literal,
    scalar,
)

ParseType = Callable[[ast.AST], ConcreteType]
ParseInterface = Callable[[ast.AST], tuple[InterfaceMetadata, ...]]
Error = Callable[[ast.AST, str], Exception]


def _scalar(name: str) -> ConcreteType:
    return scalar(name)


def _is_fragment_discard(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and (dotted_name(statement.value.func) or "").split(".")[-1] == FRAGMENT_DISCARD_OPERATION
    )


def _first_statement_with_effect(statements: tuple[TypedStatement, ...], effect: TypedEffect) -> ast.stmt | None:
    for statement in statements:
        if effect in statement.effects:
            return statement.source
        nested = _first_statement_with_effect(statement.children, effect)
        if nested is not None:
            return nested
    return None


def _contract_type(contract: TypeContract) -> ConcreteType:
    if contract.kind == "tensor":
        return ConcreteType("tensor", "Tensor", (_scalar(contract.name), *contract.shape))
    return ConcreteType(contract.kind, contract.name)


def _element(value_type: InferenceType) -> InferenceType:
    return element_type(value_type)


def _shape_rank(value_type: InferenceType) -> int | None:
    if not isinstance(value_type, ConcreteType):
        return None
    if value_type.kind == "tensor":
        return len(value_type.arguments) - 1
    if value_type.kind == "tensor_view":
        shape = value_type.arguments[1]
        return len(shape) if isinstance(shape, tuple) else None
    return None


def _common(left: InferenceType, right: InferenceType, *, division: bool = False) -> InferenceType | None:
    return common_type(left, right, division=division)


def _annotation(value_type: ConcreteType) -> ast.expr:
    if value_type.kind == "scalar":
        return ast.Name(id=value_type.name, ctx=ast.Load())
    if value_type.kind == "struct":
        return ast.Name(id=value_type.name, ctx=ast.Load())
    if value_type.kind == "tensor":
        element = value_type.arguments[0]
        assert isinstance(element, ConcreteType)
        shape = ast.Tuple(
            elts=[ast.Constant(value=value) for value in value_type.arguments[1:]],
            ctx=ast.Load(),
        )
        return ast.Subscript(
            value=ast.Name(id="Tensor", ctx=ast.Load()),
            slice=ast.Tuple(elts=[_annotation(element), shape], ctx=ast.Load()),
            ctx=ast.Load(),
        )
    if value_type.kind == "tuple":
        elements = [element for element in value_type.arguments if isinstance(element, ConcreteType)]
        return ast.Subscript(
            value=ast.Name(id="Tuple", ctx=ast.Load()),
            slice=ast.Tuple(elts=[_annotation(element) for element in elements], ctx=ast.Load()),
            ctx=ast.Load(),
        )
    if value_type.kind == "tensor_view":
        element, shape, access, _ = value_type.arguments
        assert isinstance(element, ConcreteType)
        assert isinstance(shape, tuple)
        return ast.Subscript(
            value=ast.Name(id="TensorView", ctx=ast.Load()),
            slice=ast.Tuple(
                elts=[
                    _annotation(element),
                    ast.Tuple(
                        elts=[
                            ast.Name(id="dyn", ctx=ast.Load()) if extent == "?" else ast.Constant(value=extent)
                            for extent in shape
                        ],
                        ctx=ast.Load(),
                    ),
                    ast.Name(id=str(access), ctx=ast.Load()),
                ],
                ctx=ast.Load(),
            ),
            ctx=ast.Load(),
        )
    if value_type.kind == "texture":
        dimension, element, format_name, access = value_type.arguments
        assert isinstance(element, ConcreteType)
        elements = [ast.Constant(value=dimension), _annotation(element)]
        if access != "sampled":
            elements = [
                ast.Constant(value=dimension),
                ast.Name(id=str(format_name), ctx=ast.Load()),
                ast.Name(id=str(access), ctx=ast.Load()),
            ]
        return ast.Subscript(
            value=ast.Name(id="Texture", ctx=ast.Load()),
            slice=ast.Tuple(
                elts=elements,
                ctx=ast.Load(),
            ),
            ctx=ast.Load(),
        )
    if value_type.kind == "sampler":
        return ast.Name(id="Sampler", ctx=ast.Load())
    raise ValueError(f"cannot render inferred type {value_type.kind}")


class _Inference:
    def __init__(
        self,
        module: ast.Module,
        parse_type: ParseType,
        parse_interface: ParseInterface,
        error: Error,
        enabled_features: tuple[str, ...],
        *,
        host: bool = False,
    ):
        self.host = host
        self.module = module
        self.parse_type = parse_type
        self.parse_interface = parse_interface
        self.error = error
        self.enabled_features = enabled_features
        self.functions = {node.name: node for node in module.body if isinstance(node, ast.FunctionDef)}
        # Extern declarations have only a signature: calls type-check against it, and their bodies are never typed.
        self.externs = {
            name for name, node in self.functions.items() if decorator_name(node.decorator_list[0]) == "extern"
        }
        # Value-bound generic externs, declared only as their per-Struct specializations `name__Struct`: a call
        # resolves to the specialization of its value's Struct type. Maps the name to the value's index.
        self.value_generics: dict[str, int] = getattr(module, "_vernon_value_generics", {})
        self.methods = method_tables(module)
        self.handles = {
            node.name
            for node in module.body
            if isinstance(node, ast.ClassDef) and any(decorator_name(item) == "handle" for item in node.decorator_list)
        }
        # Host enum name -> its member names; the Host frontend has checked the declarations.
        self.enums = {
            node.name: {
                target.id
                for statement in node.body
                if isinstance(statement, ast.Assign)
                for target in statement.targets
                if isinstance(target, ast.Name)
            }
            for node in module.body
            if host
            and isinstance(node, ast.ClassDef)
            and any(decorator_name(item) == "enum" for item in node.decorator_list)
        }
        self.structs = {
            node.name: tuple(
                (field.target.id, self.parse_type(field.annotation))
                for field in node.body
                if isinstance(field, ast.AnnAssign) and isinstance(field.target, ast.Name)
            )
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name not in self.handles and node.name not in self.enums
        }
        # The layouts of Host classes, whose `Name()` creates an instance from the field defaults.
        self.host_classes = {
            node.name
            for node in module.body
            if isinstance(node, ast.ClassDef) and getattr(node, "_vernon_host_class", False)
        }
        self.generic = {
            name
            for name, node in self.functions.items()
            if decorator_name(node.decorator_list[0]) == "func"
            and (any(argument.annotation is None for argument in node.args.args) or node.returns is None)
        }
        self.instances: dict[tuple[str, tuple[ConcreteType, ...], tuple[str, ...]], ast.FunctionDef] = {}
        self.instance_results: dict[tuple[str, tuple[ConcreteType, ...], tuple[str, ...]], ConcreteType | None] = {}
        self.specialized_signatures: dict[str, tuple[tuple[ConcreteType, ...], ConcreteType | None]] = {}
        self.active: list[tuple[str, tuple[ConcreteType, ...], tuple[str, ...]]] = []
        self.expression_records: dict[int, tuple[ast.expr, InferenceType, str | None]] = {}
        self.statement_merges: dict[int, tuple[BranchMerge, ...]] = {}
        self.workgroup_storage_bytes = 0

    def run(self) -> ast.Module:
        retained: list[ast.stmt] = []
        roots: list[ast.FunctionDef] = []
        for statement in self.module.body:
            if isinstance(statement, ast.FunctionDef) and statement.name in self.generic:
                continue
            retained.append(statement)
            if isinstance(statement, ast.FunctionDef) and statement.name not in self.externs:
                roots.append(statement)
        for function in roots:
            self.workgroup_storage_bytes = 0
            self._validate_direct_fragment_discard(function)
            environment = {
                argument.arg: self.parse_type(argument.annotation)
                for argument in function.args.args
                if argument.annotation is not None
            }
            returns = self._statements(function.body, environment)
            self._validate_declared_returns(function, returns)
            if (
                function.returns is not None
                and not (isinstance(function.returns, ast.Constant) and function.returns.value is None)
                and not self._block_always_returns(function.body)
            ):
                raise self.error(function, f"function '{function.name}' may exit without returning a value")
        reachable = self._reachable_instance_symbols(roots)
        specialization_keys = tuple(
            (key[0], tuple(value.mlir for value in key[1]), key[2])
            for key in sorted(
                self.instances,
                key=lambda value: (value[0], tuple(item.mlir for item in value[1]), value[2]),
            )
            if self.instances[key].name in reachable
        )
        self.module._vernon_helper_specializations = specialization_keys
        specializations = [
            self.instances[key]
            for key in sorted(
                self.instances,
                key=lambda value: (value[0], tuple(item.mlir for item in value[1]), value[2]),
            )
            if self.instances[key].name in reachable
        ]
        self.module.body = [
            *[statement for statement in retained if not isinstance(statement, ast.FunctionDef)],
            *specializations,
            *[statement for statement in retained if isinstance(statement, ast.FunctionDef)],
        ]
        self.module._vernon_typed_functions = self._build_typed_functions()
        return fix_missing_locations(self.module)

    def _validate_direct_fragment_discard(self, function: ast.FunctionDef) -> None:
        """Statements after a discard are not typed, so a non-fragment entry is rejected here before its return
        validation reports the skipped return instead."""
        if not function.decorator_list:
            return
        function_kind = decorator_name(function.decorator_list[0])
        if function_kind not in ENTRY_DECORATORS or ENTRY_DECORATOR_STAGES[function_kind] == "fragment":
            return
        discard = next(
            (
                statement
                for statement in ast.walk(function)
                if isinstance(statement, ast.stmt) and _is_fragment_discard(statement)
            ),
            None,
        )
        if discard is not None:
            raise self.error(discard, f"{FRAGMENT_DISCARD_OPERATION}() is not available in this shader stage")

    def _validate_declared_returns(
        self,
        function: ast.FunctionDef,
        returns: list[InferenceType | None],
    ) -> None:
        if function.returns is None:
            return
        expected = (
            None
            if isinstance(function.returns, ast.Constant) and function.returns.value is None
            else self.parse_type(function.returns)
        )
        if expected is None:
            if any(value is not None for value in returns):
                raise self.error(function, f"void function '{function.name}' returns a value")
            return
        if not returns:
            raise self.error(function, f"function '{function.name}' requires a return value")
        return_nodes = [node for node in ast.walk(function) if isinstance(node, ast.Return) and node.value is not None]
        for index, value in enumerate(returns):
            if value is None or not can_convert(value, expected):
                if value is None:
                    raise self.error(function, f"function '{function.name}' requires a return value")
                raise self.error(function, f"unsafe implicit conversion from {describe(value)} to {expected.spelling}")
            if index < len(return_nodes):
                assert return_nodes[index].value is not None
                self._constrain_literal(return_nodes[index].value, expected)

    @classmethod
    def _block_always_returns(cls, statements: list[ast.stmt]) -> bool:
        for statement in statements:
            if isinstance(statement, ast.Return) or _is_fragment_discard(statement):
                return True
            if isinstance(statement, ast.With) and cls._block_always_returns(statement.body):
                return True
            if (
                isinstance(statement, ast.If)
                and statement.orelse
                and cls._block_always_returns(statement.body)
                and cls._block_always_returns(statement.orelse)
            ):
                return True
        return False

    def _reachable_instance_symbols(self, roots: list[ast.FunctionDef]) -> set[str]:
        by_symbol = {instance.name: instance for instance in self.instances.values()}
        reachable: set[str] = set()
        pending = [
            (dotted_name(node.func) or "").split(".")[-1]
            for root in roots
            for node in ast.walk(root)
            if isinstance(node, ast.Call)
        ]
        while pending:
            symbol = pending.pop()
            if symbol in reachable or symbol not in by_symbol:
                continue
            reachable.add(symbol)
            pending.extend(
                (dotted_name(node.func) or "").split(".")[-1]
                for node in ast.walk(by_symbol[symbol])
                if isinstance(node, ast.Call)
            )
        return reachable

    def _build_typed_functions(self) -> tuple[TypedFunctionInstance, ...]:
        functions: list[TypedFunctionInstance] = []
        for function in self.module.body:
            if not isinstance(function, ast.FunctionDef) or function.name in self.externs:
                continue
            parameters: list[TypedParameter] = []
            for argument in function.args.args:
                argument_type = self.parse_type(argument.annotation)
                access = AccessMode.READ
                if argument_type.kind == "tensor_view":
                    access = AccessMode(str(argument_type.arguments[2]))
                elif argument_type.kind == "texture" and argument_type.arguments[3] != "sampled":
                    access = AccessMode(str(argument_type.arguments[3]))
                parameters.append(
                    TypedParameter(
                        argument.arg,
                        argument_type,
                        access,
                        self.parse_interface(argument.annotation),
                    )
                )
            result_type = (
                None
                if function.returns is None
                or (isinstance(function.returns, ast.Constant) and function.returns.value is None)
                else self.parse_type(function.returns)
            )
            parameter_map = {parameter.name: parameter for parameter in parameters}
            body = self._typed_block(function.body, parameter_map)
            functions.append(
                TypedFunctionInstance(
                    getattr(function, "_vernon_qualified_name", function.name),
                    function.name,
                    tuple(parameter.type for parameter in parameters),
                    result_type,
                    self.enabled_features,
                    function,
                    body,
                    tuple(parameters),
                )
            )
        by_symbol = {function.symbol: function for function in functions}
        resolved: dict[str, TypedFunctionInstance] = {}
        active: list[str] = []

        def resolve(symbol: str) -> TypedFunctionInstance:
            existing = resolved.get(symbol)
            if existing is not None:
                return existing
            function = by_symbol[symbol]
            if symbol in active:
                # Host functions may recurse. They take no device resources, so they carry no storage effects
                # for a caller to map, and the unresolved instance stands for the one being resolved.
                if self.host and decorator_name(function.source.decorator_list[0]) == "host":
                    return function
                raise self.error(function.source, f"recursive typed effect propagation for '{function.qualified_name}'")
            active.append(symbol)
            body = tuple(resolve_statement(statement, function) for statement in function.body)
            active.pop()
            effects = self._function_effects(body)
            result = replace(function, body=body, effects=effects)
            result = replace(result, storage_activity=analyze_storage_activity(result))
            resolved[symbol] = result
            return result

        def resolve_statement(statement: TypedStatement, caller: TypedFunctionInstance) -> TypedStatement:
            children = tuple(resolve_statement(child, caller) for child in statement.children)
            effects = list(statement.effects)
            caller_parameters = {parameter.name: parameter for parameter in caller.parameters}
            for expression in statement.expressions:
                call = expression.source
                if not isinstance(call, ast.Call):
                    continue
                callee = by_symbol.get(expression.operation or "")
                if callee is None:
                    continue
                callee = resolve(callee.symbol)
                formal_indices = {parameter.name: index for index, parameter in enumerate(callee.parameters)}
                mapped: list[tuple[str, StorageEffect]] = []
                for effect in callee.effects:
                    if isinstance(effect, FragmentDiscardEffect):
                        if effect not in effects:
                            effects.append(effect)
                        continue
                    if isinstance(effect, ResourceEffect):
                        formal_index = formal_indices.get(effect.owner)
                        if formal_index is None or formal_index >= len(call.args):
                            raise self.error(call, f"cannot bind resource effect owner '{effect.owner}'")
                        actual = call.args[formal_index]
                        if not isinstance(actual, ast.Name) or actual.id not in caller_parameters:
                            raise self.error(actual, "resource helper arguments must be Resource parameters")
                        parameter = caller_parameters[actual.id]
                        if parameter.type.kind != "texture":
                            raise self.error(actual, "resource helper owner must bind to a Texture parameter")
                        mapped_effect = replace(effect, owner=parameter.name)
                        if mapped_effect not in effects:
                            effects.append(mapped_effect)
                        continue
                    if not isinstance(effect, StorageEffect):
                        continue
                    if effect.owner.kind is not StorageOwnerKind.PARAMETER:
                        raise self.error(
                            call,
                            f"cannot propagate {effect.owner.kind.value} effect owner '{effect.owner.name}'",
                        )
                    formal_index = formal_indices.get(effect.owner.name)
                    if formal_index is None or formal_index >= len(call.args):
                        raise self.error(call, f"cannot bind effect owner '{effect.owner.name}'")
                    actual = call.args[formal_index]
                    if not isinstance(actual, ast.Name) or actual.id not in caller_parameters:
                        raise self.error(actual, "effectful helper arguments must be TensorView parameters")
                    parameter = caller_parameters[actual.id]
                    if parameter.type.kind != "tensor_view":
                        raise self.error(actual, "effectful helper owner must bind to a TensorView parameter")
                    self._validate_effect_access(effect, parameter, actual)
                    mapped_effect = replace(
                        effect,
                        owner=StorageOwner(StorageOwnerKind.PARAMETER, parameter.name),
                    )
                    mapped.append((effect.owner.name, mapped_effect))
                self._validate_call_aliases(call, mapped)
                for _, effect in mapped:
                    if effect not in effects:
                        effects.append(effect)
            return replace(statement, effects=tuple(effects), children=children)

        typed_functions = tuple(resolve(function.symbol) for function in functions)
        for function in typed_functions:
            for effect in function.effects:
                if not isinstance(effect, StorageEffect):
                    continue
                if effect.owner.kind is StorageOwnerKind.WORKGROUP_LOCAL:
                    if decorator_name(function.source.decorator_list[0]) != "kernel":
                        raise self.error(
                            function.source,
                            f"workgroup-local effect owner '{effect.owner.name}' is valid only in compute kernels",
                        )
                    continue
                parameter = next(
                    (parameter for parameter in function.parameters if parameter.name == effect.owner.name),
                    None,
                )
                if parameter is None:
                    raise self.error(
                        function.source,
                        f"parameter effect owner '{effect.owner.name}' does not name a function parameter",
                    )
                self._validate_effect_access(effect, parameter, function.source)
            function_kind = decorator_name(function.source.decorator_list[0])
            if function_kind != "kernel" and any(
                isinstance(effect, (AtomicEffect, BarrierEffect)) for effect in function.effects
            ):
                raise self.error(function.source, "workgroup synchronization is supported only in compute kernels")
            if (
                function_kind in ENTRY_DECORATORS
                and ENTRY_DECORATOR_STAGES[function_kind] != "fragment"
                and FragmentDiscardEffect() in function.effects
            ):
                raise self.error(
                    _first_statement_with_effect(function.body, FragmentDiscardEffect()) or function.source,
                    f"{FRAGMENT_DISCARD_OPERATION}() is not available in this shader stage",
                )
            invalid_resource = next(
                (
                    effect
                    for effect in function.effects
                    if isinstance(effect, ResourceEffect)
                    and effect.stages
                    and function_kind in ENTRY_DECORATORS
                    and ENTRY_DECORATOR_STAGES[function_kind] not in effect.stages
                ),
                None,
            )
            if invalid_resource is not None:
                raise self.error(
                    function.source,
                    resource_stage_error(
                        invalid_resource.operation,
                        invalid_resource.stages,
                        has_lod=invalid_resource.has_lod,
                    ),
                )
            _, mixed_sampler = resource_sampler_modes(function.effects)
            if mixed_sampler is not None:
                raise self.error(
                    _first_statement_with_effect(function.body, mixed_sampler) or function.source,
                    f"texture parameter '{mixed_sampler.owner}' cannot be sampled through both "
                    "implicit and explicit sampler forms in the same stage",
                )
            if function_kind not in ENTRY_DECORATORS | {"func"} and function.effects:
                raise self.error(
                    function.source,
                    f"function stage '{function_kind}' does not allow Storage effects",
                )
        return typed_functions

    @staticmethod
    def _function_effects(body: tuple[TypedStatement, ...]) -> tuple[TypedEffect, ...]:
        effects: list[TypedEffect] = []

        def collect(statement: TypedStatement) -> None:
            for effect in statement.effects:
                if effect not in effects:
                    effects.append(effect)
            for child in statement.children:
                collect(child)

        for statement in body:
            collect(statement)
        return tuple(effects)

    def _validate_effect_access(
        self,
        effect: StorageEffect,
        parameter: TypedParameter,
        source: ast.AST,
    ) -> None:
        if effect.kind is StorageEffectKind.READ and parameter.access is AccessMode.WRITE:
            raise self.error(source, f"effect reads write-only TensorView '{parameter.name}'")
        if effect.kind is StorageEffectKind.WRITE and parameter.access is AccessMode.READ:
            raise self.error(source, f"effect writes read-only TensorView '{parameter.name}'")

    def _validate_call_aliases(
        self,
        call: ast.Call,
        effects: list[tuple[str, StorageEffect]],
    ) -> None:
        for index, (left_formal, left) in enumerate(effects):
            for right_formal, right in effects[index + 1 :]:
                if left_formal == right_formal or left.owner != right.owner:
                    continue
                if (
                    left.kind is StorageEffectKind.WRITE or right.kind is StorageEffectKind.WRITE
                ) and left.region.overlaps(right.region):
                    name = (dotted_name(call.func) or "").split(".")[-1]
                    raise self.error(call, f"helper call '{name}' has incompatible aliased Storage effects")

    def _typed_statement(
        self,
        statement: ast.stmt,
        parameters: dict[str, TypedParameter],
        loop_depth: int = 0,
    ) -> TypedStatement:
        typed_expressions: list[TypedExpression] = []
        for expression in self._statement_expressions(statement):
            record = self.expression_records.get(id(expression))
            if record is None:
                continue
            _, inferred, operation = record
            value_type = default_type(inferred)
            access = AccessMode.READ
            if isinstance(expression, ast.Name) and expression.id in parameters:
                parameter = parameters[expression.id]
                access = parameter.access
            typed_expressions.append(
                TypedExpression(
                    expression,
                    value_type,
                    operation,
                    self._typed_operand_types(expression, value_type),
                    access,
                )
            )
        lvalues = self._statement_lvalues(statement, parameters)
        children: list[TypedStatement] = []
        if isinstance(statement, ast.If):
            children.extend(self._typed_block(statement.body, parameters, loop_depth))
            children.extend(self._typed_block(statement.orelse, parameters, loop_depth))
        elif isinstance(statement, (ast.For, ast.While)):
            children.extend(self._typed_block(statement.body, parameters, loop_depth + 1))
            children.extend(self._typed_block(statement.orelse, parameters, loop_depth))
        elif isinstance(statement, ast.With):
            children.extend(self._typed_block(statement.body, parameters, loop_depth))
        effects = (
            *self._storage_effects(statement, typed_expressions, parameters),
            *self._resource_effects(typed_expressions, parameters),
            *self._synchronization_effects(typed_expressions),
            *((FragmentDiscardEffect(),) if _is_fragment_discard(statement) else ()),
        )
        termination = (
            Termination.RETURN
            if isinstance(statement, ast.Return)
            else Termination.BREAK
            if isinstance(statement, ast.Break)
            else Termination.CONTINUE
            if isinstance(statement, ast.Continue)
            else Termination.DISCARD
            if _is_fragment_discard(statement)
            else Termination.FALLTHROUGH
        )
        return_type = None
        if isinstance(statement, ast.Return) and statement.value is not None:
            record = self.expression_records.get(id(statement.value))
            if record is not None:
                return_type = default_type(record[1])
        return TypedStatement(
            statement,
            termination,
            effects,
            tuple(typed_expressions),
            lvalues,
            self.statement_merges.get(id(statement), ()),
            tuple(children),
            loop_depth,
            return_type,
        )

    def _typed_block(
        self,
        statements: list[ast.stmt],
        parameters: dict[str, TypedParameter],
        loop_depth: int = 0,
    ) -> tuple[TypedStatement, ...]:
        typed: list[TypedStatement] = []
        for statement in statements:
            item = self._typed_statement(statement, parameters, loop_depth)
            typed.append(item)
            if item.termination is not Termination.FALLTHROUGH:
                break
        return tuple(typed)

    @staticmethod
    def _storage_region(node: ast.Subscript) -> StorageRegion:
        indices = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
        constants: list[int] = []
        for index in indices:
            if not isinstance(index, ast.Constant) or not isinstance(index.value, int) or isinstance(index.value, bool):
                return StorageRegion(StorageRegionKind.UNKNOWN)
            constants.append(index.value)
        return StorageRegion(StorageRegionKind.ELEMENT, tuple(constants))

    @classmethod
    def _storage_effects(
        cls,
        statement: ast.stmt,
        expressions: list[TypedExpression],
        parameters: dict[str, TypedParameter],
    ) -> tuple[StorageEffect, ...]:
        effects: list[StorageEffect] = []
        workgroup_subscripts = {
            id(expression.source)
            for expression in expressions
            if isinstance(expression.source, ast.Subscript)
            and expression.operand_types
            and expression.operand_types[0].kind == "tensor_view"
            and expression.operand_types[0].arguments[3] == "workgroup"
        }

        def effect_for(node: ast.Subscript, kind: StorageEffectKind) -> StorageEffect | None:
            if not isinstance(node.value, ast.Name):
                return None
            parameter = parameters.get(node.value.id)
            if parameter is None and id(node) in workgroup_subscripts:
                return StorageEffect(
                    kind,
                    StorageOwner(StorageOwnerKind.WORKGROUP_LOCAL, node.value.id),
                    cls._storage_region(node),
                )
            if parameter is None or parameter.type.kind != "tensor_view":
                return None
            return StorageEffect(
                kind,
                StorageOwner(StorageOwnerKind.PARAMETER, parameter.name),
                cls._storage_region(node),
            )

        augmented_target = statement.target if isinstance(statement, ast.AugAssign) else None
        for expression in expressions:
            node = expression.source
            if not isinstance(node, ast.Subscript):
                continue
            if not isinstance(node.ctx, ast.Load) and node is not augmented_target:
                continue
            effect = effect_for(node, StorageEffectKind.READ)
            if effect is not None and effect not in effects:
                effects.append(effect)

        target: ast.expr | None = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
        elif isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
            target = statement.target
        if isinstance(target, ast.Subscript):
            effect = effect_for(target, StorageEffectKind.WRITE)
            if effect is not None and effect not in effects:
                effects.append(effect)

        return tuple(effects)

    def _resource_effects(
        self,
        expressions: list[TypedExpression],
        parameters: dict[str, TypedParameter],
    ) -> tuple[ResourceEffect, ...]:
        effects: list[ResourceEffect] = []
        for expression in expressions:
            if expression.operation not in {"texture_sample", "texture_sample_bias", "texture_size", "texture_load"}:
                continue
            call = expression.source
            if not isinstance(call, ast.Call) or not call.args or not isinstance(call.args[0], ast.Name):
                continue
            texture_parameter = parameters.get(call.args[0].id)
            if texture_parameter is None or texture_parameter.type.kind != "texture":
                continue
            sampler_mode: str | None = None
            if expression.operation == "texture_load":
                if texture_parameter.type.arguments[3] != "sampled":
                    continue
                stages = GRAPHICS_STAGES
                has_lod = False
            elif expression.operation == "texture_size":
                stages = GRAPHICS_STAGES
                has_lod = len(call.args) == 2
            else:
                argument_kinds = ["texture"]
                for argument in call.args[1:]:
                    parameter = parameters.get(argument.id) if isinstance(argument, ast.Name) else None
                    argument_kinds.append(parameter.type.kind if parameter is not None else "value")
                sampling = (
                    texture_bias_sampling_contract(argument_kinds)
                    if expression.operation == "texture_sample_bias"
                    else texture_sampling_contract(argument_kinds)
                )
                if sampling is None:
                    continue
                stages = sampling.stages
                has_lod = sampling.has_lod
                sampler_mode = sampling.sampler_mode
            effect = ResourceEffect(expression.operation, texture_parameter.name, stages, has_lod, sampler_mode)
            if effect not in effects:
                effects.append(effect)
        return tuple(effects)

    def _synchronization_effects(
        self, expressions: list[TypedExpression]
    ) -> tuple[AtomicEffect | BarrierEffect | StorageEffect, ...]:
        effects: list[AtomicEffect | BarrierEffect | StorageEffect] = []
        for expression in expressions:
            call = expression.source
            if not isinstance(call, ast.Call):
                continue
            if expression.operation in {"workgroup_barrier", "storage_barrier"}:
                effect = BarrierEffect(
                    MemoryOrdering.ACQUIRE_RELEASE,
                    EffectScope.WORKGROUP if expression.operation == "workgroup_barrier" else EffectScope.DEVICE,
                )
            elif expression.operation in ATOMIC_OPERATION_NAMES:
                if len(call.args) != 3:
                    raise self.error(call, "internal compiler error: typed atomic call has invalid arity")
                if not isinstance(call.args[0], ast.Name):
                    raise self.error(call.args[0], "internal compiler error: typed atomic owner is not a name")
                index = call.args[1]
                index_nodes = list(index.elts) if isinstance(index, ast.Tuple) else [index]
                constant_indices = tuple(
                    item.value
                    for item in index_nodes
                    if isinstance(item, ast.Constant)
                    and isinstance(item.value, int)
                    and not isinstance(item.value, bool)
                )
                region = (
                    StorageRegion(StorageRegionKind.ELEMENT, constant_indices)
                    if len(constant_indices) == len(index_nodes)
                    else StorageRegion(StorageRegionKind.UNKNOWN)
                )
                owner_record = self.expression_records.get(id(call.args[0]))
                if owner_record is None or not isinstance(owner_record[1], ConcreteType):
                    raise self.error(call.args[0], "internal compiler error: atomic storage owner has no type")
                owner_type = owner_record[1]
                if owner_type.kind == "tensor_view" and owner_type.arguments[3] == "workgroup":
                    scope = EffectScope.WORKGROUP
                    owner_kind = StorageOwnerKind.WORKGROUP_LOCAL
                elif owner_type.kind == "tensor_view":
                    scope = EffectScope.DEVICE
                    owner_kind = StorageOwnerKind.PARAMETER
                else:
                    raise self.error(
                        call.args[0],
                        f"internal compiler error: atomic storage owner has unexpected type '{owner_type.kind}'",
                    )
                owner = StorageOwner(owner_kind, call.args[0].id)
                effect = AtomicEffect(
                    expression.operation.removeprefix("atomic_"),
                    owner,
                    region,
                    MemoryOrdering.RELAXED,
                    scope,
                )
                storage_effect = StorageEffect(StorageEffectKind.WRITE, owner, region, atomic=True)
                if scope is EffectScope.DEVICE and storage_effect not in effects:
                    effects.append(storage_effect)
            else:
                continue
            if effect not in effects:
                effects.append(effect)
        return tuple(effects)

    def _reject_bool_tensor(self, node: ast.expr, value_type: InferenceType) -> None:
        value_type = default_type(value_type)
        if (
            isinstance(value_type, ConcreteType)
            and value_type.kind == "tensor"
            and _element(value_type) == _scalar("bool")
        ):
            raise self.error(node, "a bool Tensor has no single truth value; reduce it with any() or all()")

    def _require_bool(self, node: ast.expr, value_type: InferenceType, message: str) -> None:
        self._reject_bool_tensor(node, value_type)
        if default_type(value_type) != _scalar("bool"):
            raise self.error(node, message)

    def _typed_operand_types(
        self,
        expression: ast.expr,
        result_type: ConcreteType,
    ) -> tuple[ConcreteType, ...]:
        if isinstance(expression, ast.BinOp):
            return (result_type, result_type)
        if isinstance(expression, ast.BoolOp):
            return tuple(result_type for _ in expression.values)
        if isinstance(expression, ast.UnaryOp) and isinstance(expression.op, ast.Not):
            return (_scalar("bool"),)
        if isinstance(expression, ast.IfExp):
            return (_scalar("bool"), result_type, result_type)
        if isinstance(expression, ast.Compare) and expression.comparators:
            left = self.expression_records.get(id(expression.left))
            right = self.expression_records.get(id(expression.comparators[0]))
            if left is not None and right is not None:
                common = common_type(left[1], right[1])
                if common is not None:
                    concrete = default_type(common)
                    return (concrete, concrete)
        if isinstance(expression, ast.Subscript):
            value = self.expression_records.get(id(expression.value))
            if value is not None:
                return (default_type(value[1]),)
        return ()

    @staticmethod
    def _statement_expressions(statement: ast.stmt) -> list[ast.expr]:
        result: list[ast.expr] = []

        def visit(node: ast.AST) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.stmt):
                    continue
                if isinstance(child, ast.expr):
                    result.append(child)
                visit(child)

        visit(statement)
        return result

    def _statement_lvalues(
        self,
        statement: ast.stmt,
        parameters: dict[str, TypedParameter],
    ) -> tuple[LValue, ...]:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(statement, ast.With):
            bound: list[LValue] = []
            for item in statement.items:
                record = self.expression_records.get(id(item.context_expr))
                if isinstance(item.optional_vars, ast.Name) and record is not None:
                    bound.append(LValue("local", item.optional_vars.id, default_type(record[1])))
            return tuple(bound)
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            target = statement.target
            value = statement.value
        elif isinstance(statement, ast.AugAssign):
            target = statement.target
            value = statement.value
        if target is None:
            return ()
        if isinstance(target, ast.Name):
            record = self.expression_records.get(id(value)) if value is not None else None
            value_type = default_type(record[1]) if record is not None else ConcreteType("void", "void")
            return (LValue("local", target.id, value_type),)
        if isinstance(target, (ast.Tuple, ast.List)):
            record = self.expression_records.get(id(value)) if value is not None else None
            value_type = default_type(record[1]) if record is not None else ConcreteType("void", "void")

            def destructured_lvalues(node: ast.expr, node_type: ConcreteType) -> tuple[LValue, ...]:
                if isinstance(node, ast.Name):
                    return (LValue("local", node.id, node_type),)
                if not isinstance(node, (ast.Tuple, ast.List)) or node_type.kind != "tuple":
                    return ()
                return tuple(
                    lvalue
                    for item, element in zip(node.elts, node_type.arguments, strict=True)
                    if isinstance(element, ConcreteType)
                    for lvalue in destructured_lvalues(item, element)
                )

            return destructured_lvalues(target, value_type)
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            parameter = parameters.get(target.value.id)
            if parameter is not None:
                element = parameter.type.arguments[0]
                assert isinstance(element, ConcreteType)
                return (
                    LValue(
                        "index",
                        target.value.id,
                        element,
                        parameter.access,
                    ),
                )
        return ()

    def _statements(
        self,
        statements: list[ast.stmt],
        environment: dict[str, InferenceType],
        loop_depth: int = 0,
    ) -> list[InferenceType | None]:
        returns: list[InferenceType | None] = []
        for statement in statements:
            if (
                self.host
                and isinstance(statement, ast.Assign)
                and isinstance(statement.value, (ast.List, ast.Dict))
                and not (statement.value.elts if isinstance(statement.value, ast.List) else statement.value.keys)
            ):
                # `[]` and `{}` have no type of their own: each is only the empty value of the field or local it
                # is assigned to.
                kind = "list" if isinstance(statement.value, ast.List) else "dict"
                spelling = "[]" if kind == "list" else "{}"
                target = statement.targets[0]
                if isinstance(target, ast.Name) and target.id not in environment:
                    raise self.error(
                        statement,
                        f"a {kind} local is declared with its type, as in {target.id}: "
                        f"{'list[T]' if kind == 'list' else 'dict[K, V]'} = {spelling}",
                    )
                target_type = self._expression(target, environment)
                if not isinstance(target_type, ConcreteType) or target_type.kind != kind:
                    raise self.error(statement.value, f"{spelling} can only be assigned to a {kind} field or local")
                self.expression_records[id(statement.value)] = (statement.value, target_type, kind)
            elif (
                self.host
                and isinstance(statement, ast.AnnAssign)
                and isinstance(statement.value, (ast.List, ast.Dict))
                and not (statement.value.elts if isinstance(statement.value, ast.List) else statement.value.keys)
            ):
                kind = "list" if isinstance(statement.value, ast.List) else "dict"
                expected = self.parse_type(statement.annotation)
                if expected.kind != kind or not isinstance(statement.target, ast.Name):
                    spelling = "[]" if kind == "list" else "{}"
                    raise self.error(statement.value, f"{spelling} declares a local annotated {kind}[...]")
                environment[statement.target.id] = expected
                self.expression_records[id(statement.value)] = (statement.value, expected, kind)
            elif self.host and isinstance(statement, ast.Delete):
                for target in statement.targets:
                    container = (
                        self._expression(target.value, environment) if isinstance(target, ast.Subscript) else None
                    )
                    if not isinstance(container, ConcreteType) or container.kind != "dict":
                        raise self.error(target, "del removes a dict entry, as in del self.counts[key]")
                    assert isinstance(target, ast.Subscript)
                    self._dict_key(target.slice, container, environment)
            elif isinstance(statement, ast.Assign):
                value_type = self._expression(statement.value, environment)
                if len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                    name = statement.targets[0].id
                    previous = environment.get(name)
                    environment[name] = _common(previous, value_type) if previous is not None else value_type
                    if environment[name] is None:
                        raise self.error(statement, f"local '{name}' has incompatible assignment types")
                    if previous is not None:
                        self._constrain_literal(statement.value, environment[name])
                elif len(statement.targets) == 1 and isinstance(statement.targets[0], (ast.Tuple, ast.List)):
                    self._bind_destructuring(statement.targets[0], value_type, environment)
                elif (
                    self.host
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Subscript)
                    and self._is_dict(statement.targets[0].value, environment)
                ):
                    target_type = self._expression(statement.targets[0].value, environment)
                    assert isinstance(target_type, ConcreteType)
                    self._dict_key(statement.targets[0].slice, target_type, environment)
                    expected = target_type.arguments[1]
                    assert isinstance(expected, ConcreteType)
                    if not can_convert(value_type, expected):
                        raise self.error(statement.value, f"cannot store {describe(value_type)} as {expected.spelling}")
                    self._constrain_literal(statement.value, expected)
                elif (
                    self.host
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Subscript)
                    and isinstance(storage := self._expression(statement.targets[0].value, environment), ConcreteType)
                    and storage.kind == "storage"
                ):
                    expected = self._storage_element(statement.targets[0], storage, environment)
                    if expected.kind == "tensor":
                        self._require_block(statement.value, value_type, expected, "store")
                    elif not can_convert(value_type, expected):
                        raise self.error(statement.value, f"cannot store {describe(value_type)} as {expected.spelling}")
                    else:
                        self._constrain_literal(statement.value, expected)
                elif len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Subscript):
                    target_type = self._expression(statement.targets[0].value, environment)
                    self._infer_indices(statement.targets[0].slice, environment)
                    self._reject_value_write(statement.targets[0], target_type)
                    if not isinstance(target_type, ConcreteType) or target_type.kind not in {
                        "tensor_view",
                        *({"list"} if self.host else ()),
                    }:
                        raise self.error(statement.targets[0], "indexed assignment requires writable Storage")
                    if target_type.kind == "list" and isinstance(statement.targets[0].slice, ast.Tuple):
                        raise self.error(statement.targets[0].slice, "a list takes one index")
                    if target_type.kind == "tensor_view" and target_type.arguments[2] == "read":
                        raise self.error(statement.targets[0], "cannot assign through a read-only TensorView")
                    expected = target_type.arguments[0]
                    assert isinstance(expected, ConcreteType)
                    if not can_convert(value_type, expected):
                        raise self.error(statement.value, f"cannot store {describe(value_type)} as {expected.spelling}")
                elif self.host and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Attribute):
                    # Which fields are writable places is a Host rule; the Host validator checks it.
                    expected = self._expression(statement.targets[0], environment)
                    if not can_convert(value_type, expected):
                        raise self.error(
                            statement.value, f"cannot store {describe(value_type)} as {describe(expected)}"
                        )
                    self._constrain_literal(statement.value, expected)
                else:
                    raise self.error(statement, "assignment target must be a local name or TensorView element")
            elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
                value_type = self._expression(statement.value, environment)
                expected = self.parse_type(statement.annotation)
                if not can_convert(value_type, expected):
                    raise self.error(statement.value, f"cannot infer assignment as {expected.spelling}")
                self._constrain_literal(statement.value, expected)
                if isinstance(statement.target, ast.Name):
                    environment[statement.target.id] = expected
            elif isinstance(statement, ast.AugAssign):
                if isinstance(statement.target, ast.Subscript):
                    self._reject_value_write(statement.target, self._expression(statement.target.value, environment))
                target_type = self._expression(statement.target, environment)
                value_type = self._expression(statement.value, environment)
                common = _common(target_type, value_type)
                if common is None:
                    raise self.error(statement, "augmented assignment has incompatible types")
                self._constrain_literal(statement.value, common)
                if isinstance(statement.target, ast.Name) and statement.target.id in environment:
                    environment[statement.target.id] = common
            elif isinstance(statement, ast.Expr):
                self._expression(statement.value, environment)
                if _is_fragment_discard(statement):
                    break
            elif isinstance(statement, ast.Break):
                if loop_depth == 0:
                    raise self.error(statement, "break is only valid inside a loop")
                break
            elif isinstance(statement, ast.Continue):
                if loop_depth == 0:
                    raise self.error(statement, "continue is only valid inside a loop")
                break
            elif isinstance(statement, ast.Return):
                returns.append(None if statement.value is None else self._expression(statement.value, environment))
                break
            elif self.host and isinstance(statement, ast.With):
                for item in statement.items:
                    value_type = self._expression(item.context_expr, environment)
                    if item.optional_vars is None:
                        continue
                    if not isinstance(item.optional_vars, ast.Name):
                        raise self.error(item.optional_vars, "'with ... as' binds one local name")
                    if value_type is None or (isinstance(value_type, ConcreteType) and value_type.kind == "void"):
                        raise self.error(item.context_expr, "this scope returns nothing to bind with 'as'")
                    environment[item.optional_vars.id] = value_type
                # The body runs exactly once, so its assignments stand after the block.
                returns.extend(self._statements(statement.body, environment, loop_depth))
            elif isinstance(statement, ast.If):
                self._reject_bool_tensor(statement.test, self._expression(statement.test, environment))
                before = environment.copy()
                then_environment = before.copy()
                else_environment = before.copy()
                returns.extend(self._statements(statement.body, then_environment, loop_depth))
                returns.extend(self._statements(statement.orelse, else_environment, loop_depth))
                branch_merges: list[BranchMerge] = []
                assigned = assigned_names((*statement.body, *statement.orelse))
                for name in sorted(then_environment.keys() & else_environment.keys() & assigned):
                    merged = _common(then_environment[name], else_environment[name])
                    if merged is None:
                        raise self.error(statement, f"branch local '{name}' has incompatible types")
                    environment[name] = merged
                    branch_merges.append(BranchMerge(name, default_type(merged)))
                self.statement_merges[id(statement)] = tuple(branch_merges)
            elif isinstance(statement, ast.While):
                self._reject_bool_tensor(statement.test, self._expression(statement.test, environment))
                before = environment.copy()
                loop_environment = before.copy()
                loop_returns: list[InferenceType | None] = []
                assigned = assigned_names(statement.body)
                for _ in range(8):
                    candidate = loop_environment.copy()
                    loop_returns = self._statements(statement.body, candidate, loop_depth + 1)
                    merged_environment = before.copy()
                    for name in sorted(before.keys() & candidate.keys() & assigned):
                        merged = _common(before[name], candidate[name])
                        if merged is None:
                            raise self.error(statement, f"loop local '{name}' has incompatible types")
                        merged_environment[name] = merged
                    if merged_environment == loop_environment:
                        break
                    loop_environment = merged_environment
                else:
                    raise self.error(statement, "loop-carried type inference did not converge")
                else_returns = self._merge_loop_else(statement, before, loop_environment, loop_depth)
                environment.update(loop_environment)
                self.statement_merges[id(statement)] = tuple(
                    BranchMerge(name, default_type(loop_environment[name]))
                    for name in sorted(before.keys() & loop_environment.keys() & assigned)
                )
                returns.extend((*loop_returns, *else_returns))
            elif isinstance(statement, ast.For):
                view = self._dict_view(statement.iter, environment) if self.host else None
                if view is not None:
                    returns.extend(self._dict_loop(statement, view, environment, loop_depth))
                    continue
                if not isinstance(statement.target, ast.Name):
                    raise self.error(statement.target, "for loop target must be a local name")
                is_range = (
                    isinstance(statement.iter, ast.Call)
                    and dotted_name(statement.iter.func) == "range"
                    and 1 <= len(statement.iter.args) <= 3
                    and not statement.iter.keywords
                )
                calls_range = isinstance(statement.iter, ast.Call) and dotted_name(statement.iter.func) == "range"
                iterated = None if calls_range or not self.host else self._expression(statement.iter, environment)
                if isinstance(iterated, ConcreteType) and iterated.kind == "list":
                    element = iterated.arguments[0]
                    assert isinstance(element, ConcreteType)
                    range_type = element
                elif not is_range:
                    raise self.error(
                        statement.iter,
                        "for loops require range(stop), range(start, stop), or range(start, stop, step)"
                        + (", or a list" if self.host else ""),
                    )
                else:
                    range_type = ConcreteType("scalar", "i32")
                for argument in statement.iter.args if is_range else ():  # type: ignore[union-attr]
                    inferred_argument = self._expression(argument, environment)
                    concrete_argument = default_type(inferred_argument)
                    if concrete_argument.kind != "scalar" or not concrete_argument.is_integer:
                        raise self.error(argument, "range arguments must be i32 Values or integer literals")
                    if concrete_argument.name == "u32":
                        raise self.error(argument, "range u32 arguments require an explicit i32 conversion")
                    self._constrain_literal(argument, range_type)
                if is_range and len(statement.iter.args) == 3 and self._is_literal_zero(statement.iter.args[2]):  # type: ignore[union-attr]
                    raise self.error(statement.iter.args[2], "range step must not be zero")
                before = environment.copy()
                loop_environment = {**before, statement.target.id: range_type}
                loop_returns: list[InferenceType | None] = []
                assigned = assigned_names(statement.body)
                for _ in range(8):
                    candidate = loop_environment.copy()
                    loop_returns = self._statements(statement.body, candidate, loop_depth + 1)
                    merged_environment = before.copy()
                    for name in sorted(before.keys() & candidate.keys() & assigned):
                        merged = _common(before[name], candidate[name])
                        if merged is None:
                            raise self.error(statement, f"loop local '{name}' has incompatible types")
                        merged_environment[name] = merged
                    next_environment = {**merged_environment, statement.target.id: range_type}
                    if next_environment == loop_environment:
                        break
                    loop_environment = next_environment
                else:
                    raise self.error(statement, "loop-carried type inference did not converge")
                else_returns = self._merge_loop_else(
                    statement, before, loop_environment, loop_depth, excluded_name=statement.target.id
                )
                environment.update(
                    {name: value for name, value in loop_environment.items() if name != statement.target.id}
                )
                self.statement_merges[id(statement)] = tuple(
                    BranchMerge(name, default_type(loop_environment[name]))
                    for name in sorted(before.keys() & loop_environment.keys() & assigned)
                )
                returns.extend((*loop_returns, *else_returns))
        return returns

    def _is_dict(self, node: ast.expr, environment: dict[str, InferenceType]) -> bool:
        value_type = self._expression(node, environment)
        return isinstance(value_type, ConcreteType) and value_type.kind == "dict"

    def _dict_key(self, node: ast.expr, dict_type: ConcreteType, environment: dict[str, InferenceType]) -> None:
        key = dict_type.arguments[0]
        assert isinstance(key, ConcreteType)
        key_type = self._expression(node, environment)
        if not can_convert(key_type, key):
            raise self.error(node, f"{dict_type.spelling} has {key.spelling} keys, not {describe(key_type)}")
        self._constrain_literal(node, key)

    def _dict_view(self, node: ast.expr, environment: dict[str, InferenceType]) -> tuple[str, ConcreteType] | None:
        """For `d`, `d.keys()`, `d.values()` or `d.items()` over a dict `d`: what the loop visits and the dict's
        type; None for other iterables."""

        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"keys", "values", "items"}
            and self._receiver(node.func.value, environment)
        ):
            receiver = self._expression(node.func.value, environment)
            if isinstance(receiver, ConcreteType) and receiver.kind == "dict":
                if node.args or node.keywords:
                    raise self.error(node, f"dict.{node.func.attr} takes no arguments")
                self.expression_records[id(node)] = (node, receiver, node.func.attr)
                return node.func.attr, receiver
            return None
        if isinstance(node, ast.Call):
            return None
        iterated = self._expression(node, environment)
        if isinstance(iterated, ConcreteType) and iterated.kind == "dict":
            return "keys", iterated
        return None

    def _dict_loop(
        self,
        statement: ast.For,
        view: tuple[str, ConcreteType],
        environment: dict[str, InferenceType],
        loop_depth: int,
    ) -> list[InferenceType | None]:
        kind, dict_type = view
        key, value = dict_type.arguments
        assert isinstance(key, ConcreteType) and isinstance(value, ConcreteType)
        if kind == "items":
            target = statement.target
            if not (
                isinstance(target, ast.Tuple)
                and len(target.elts) == 2
                and all(isinstance(item, ast.Name) for item in target.elts)
            ):
                raise self.error(target, "a loop over dict.items() binds two local names, as in for key, value in")
            bound = {target.elts[0].id: key, target.elts[1].id: value}  # type: ignore[attr-defined]
        else:
            if not isinstance(statement.target, ast.Name):
                raise self.error(statement.target, "for loop target must be a local name")
            bound = {statement.target.id: key if kind == "keys" else value}
        before = environment.copy()
        loop_environment = {**before, **bound}
        loop_returns: list[InferenceType | None] = []
        assigned = assigned_names(statement.body)
        for _ in range(8):
            candidate = loop_environment.copy()
            loop_returns = self._statements(statement.body, candidate, loop_depth + 1)
            merged_environment = before.copy()
            for name in sorted(before.keys() & candidate.keys() & assigned):
                merged = _common(before[name], candidate[name])
                if merged is None:
                    raise self.error(statement, f"loop local '{name}' has incompatible types")
                merged_environment[name] = merged
            next_environment = {**merged_environment, **bound}
            if next_environment == loop_environment:
                break
            loop_environment = next_environment
        else:
            raise self.error(statement, "loop-carried type inference did not converge")
        else_environment = {name: item for name, item in loop_environment.items() if name not in bound}
        else_returns = self._statements(statement.orelse, else_environment, loop_depth)
        for name in sorted(before.keys() & else_environment.keys() & assigned_names(statement.orelse)):
            merged = _common(loop_environment[name], else_environment[name])
            if merged is None:
                raise self.error(statement, f"loop-else local '{name}' has incompatible types")
            loop_environment[name] = merged
        environment.update({name: item for name, item in loop_environment.items() if name not in bound})
        self.statement_merges[id(statement)] = tuple(
            BranchMerge(name, default_type(loop_environment[name]))
            for name in sorted(before.keys() & loop_environment.keys() & assigned)
        )
        return [*loop_returns, *else_returns]

    def _merge_loop_else(
        self,
        statement: ast.While | ast.For,
        before: dict[str, InferenceType],
        loop_environment: dict[str, InferenceType],
        loop_depth: int,
        *,
        excluded_name: str | None = None,
    ) -> list[InferenceType | None]:
        else_environment = {name: value for name, value in loop_environment.items() if name != excluded_name}
        else_returns = self._statements(statement.orelse, else_environment, loop_depth)
        for name in sorted(before.keys() & else_environment.keys() & assigned_names(statement.orelse)):
            merged = _common(loop_environment[name], else_environment[name])
            if merged is None:
                raise self.error(statement, f"loop-else local '{name}' has incompatible types")
            loop_environment[name] = merged
        return else_returns

    @staticmethod
    def _is_literal_zero(node: ast.expr) -> bool:
        if isinstance(node, ast.Constant):
            return isinstance(node.value, int) and not isinstance(node.value, bool) and node.value == 0
        return (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, (ast.UAdd, ast.USub))
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, int)
            and not isinstance(node.operand.value, bool)
            and node.operand.value == 0
        )

    def _bind_destructuring(
        self,
        target: ast.Tuple | ast.List,
        value_type: InferenceType,
        environment: dict[str, InferenceType],
    ) -> None:
        if not isinstance(value_type, ConcreteType) or value_type.kind != "tuple":
            raise self.error(target, "assignment target destructuring requires a Tuple value")
        if len(target.elts) != len(value_type.arguments):
            raise self.error(target, "Tuple destructuring arity does not match the value")
        for item, element in zip(target.elts, value_type.arguments, strict=True):
            assert isinstance(element, ConcreteType)
            if isinstance(item, ast.Name):
                previous = environment.get(item.id)
                merged = _common(previous, element) if previous is not None else element
                if merged is None:
                    raise self.error(item, f"local '{item.id}' has incompatible assignment types")
                environment[item.id] = merged
            elif isinstance(item, (ast.Tuple, ast.List)):
                self._bind_destructuring(item, element, environment)
            else:
                raise self.error(item, "Tuple destructuring targets must be local names")

    def _expression(self, node: ast.expr, environment: dict[str, InferenceType]) -> InferenceType:
        result = self._infer_expression(node, environment)
        operation = self._operation(node)
        self.expression_records[id(node)] = (node, result, operation)
        if isinstance(result, ConcreteType) and isinstance(node, ast.BinOp):
            self._constrain_literal(node.left, element_type(result))
            self._constrain_literal(node.right, element_type(result))
        return result

    def _infer_expression(self, node: ast.expr, environment: dict[str, InferenceType]) -> InferenceType:
        if isinstance(node, ast.Name):
            if node.id not in environment:
                raise self.error(node, f"cannot infer unknown value '{node.id}'")
            return environment[node.id]
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (bool, int, float)):
                return literal(node.value)
            if isinstance(node.value, str) and self.host:
                return ConcreteType("str", "str")
        if self.host and isinstance(node, ast.JoinedStr):
            for part in node.values:
                if isinstance(part, ast.FormattedValue):
                    value = self._expression(part.value, environment)
                    self._constrain_literal(part.value, default_type(value))
            return ConcreteType("str", "str")
        if self.host and isinstance(node, (ast.List, ast.Dict)):
            kind, empty = ("list", "[]") if isinstance(node, ast.List) else ("dict", "{}")
            raise self.error(
                node,
                f"a {kind} is not passed as a value; {empty} declares a {kind} local "
                f"(x: {'list[T]' if kind == 'list' else 'dict[K, V]'} = {empty}) or empties a {kind} field or local",
            )
        if isinstance(node, ast.BinOp):
            left = self._expression(node.left, environment)
            right = self._expression(node.right, environment)
            common = _common(left, right, division=isinstance(node.op, ast.Div))
            if common is None:
                raise self.error(node, f"no safe common type for {describe(left)} and {describe(right)}")
            return common
        if isinstance(node, ast.UnaryOp):
            operand = self._expression(node.operand, environment)
            if isinstance(node.op, ast.Not):
                self._require_bool(node.operand, operand, "not operand must be a bool Value")
                return _scalar("bool")
            return operand
        if isinstance(node, ast.BoolOp):
            for value in node.values:
                self._require_bool(value, self._expression(value, environment), "and/or operands must be bool Values")
            return _scalar("bool")
        if isinstance(node, ast.IfExp):
            self._require_bool(
                node.test,
                self._expression(node.test, environment),
                "conditional expression condition must be a bool Value",
            )
            then_type = self._expression(node.body, environment)
            else_type = self._expression(node.orelse, environment)
            result = _common(then_type, else_type)
            if result is None:
                raise self.error(node, "conditional expression branches have incompatible types")
            result = default_type(result)
            self._constrain_literal(node.body, result)
            self._constrain_literal(node.orelse, result)
            return result
        if self.host and isinstance(node, ast.Compare) and isinstance(node.ops[0], (ast.In, ast.NotIn)):
            container = self._expression(node.comparators[0], environment)
            if len(node.ops) != 1 or not isinstance(container, ConcreteType) or container.kind != "dict":
                raise self.error(node, "'in' tests whether a dict has a key")
            self._dict_key(node.left, container, environment)
            return _scalar("bool")
        if isinstance(node, ast.Compare):
            left = self._expression(node.left, environment)
            result = _scalar("bool")
            for value in node.comparators:
                right = self._expression(value, environment)
                common = _common(left, right)
                if common is None:
                    raise self.error(node, f"no safe common type for {describe(left)} and {describe(right)}")
                self._constrain_literal(node.left, common)
                self._constrain_literal(value, common)
                left = right
                common = default_type(common)
                # Tensors compare element by element, as in numpy, WGSL and HLSL.
                if isinstance(common, ConcreteType) and common.kind == "tensor":
                    if len(node.ops) != 1:
                        raise self.error(node, "Tensor comparisons cannot be chained")
                    result = ConcreteType("tensor", "Tensor", (_scalar("bool"), *common.arguments[1:]))
            return result
        if isinstance(node, ast.Tuple):
            if not node.elts:
                raise self.error(node, "Tuple values must contain at least one element")
            return ConcreteType(
                "tuple",
                "Tuple",
                tuple(default_type(self._expression(element, environment)) for element in node.elts),
            )
        if isinstance(node, ast.Subscript):
            value_type = self._expression(node.value, environment)
            if isinstance(value_type, ConcreteType) and value_type.kind == "tuple":
                if (
                    not isinstance(node.slice, ast.Constant)
                    or not isinstance(node.slice.value, int)
                    or isinstance(node.slice.value, bool)
                ):
                    raise self.error(node.slice, "Tuple indexing requires an integer literal")
                index = node.slice.value
                if index < 0:
                    index += len(value_type.arguments)
                if index < 0 or index >= len(value_type.arguments):
                    raise self.error(node.slice, "Tuple index is out of bounds")
                element = value_type.arguments[index]
                assert isinstance(element, ConcreteType)
                return element
            if isinstance(value_type, ConcreteType) and value_type.kind == "dict":
                self._dict_key(node.slice, value_type, environment)
                element = value_type.arguments[1]
                assert isinstance(element, ConcreteType)
                return element
            if isinstance(value_type, ConcreteType) and value_type.kind == "storage":
                return self._storage_element(node, value_type, environment)
            if isinstance(value_type, ConcreteType) and value_type.kind == "tensor":
                indices = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
                rank = len(value_type.arguments) - 1
                if is_slicing(indices, rank) or len(indices) > rank:
                    return self._selection_type(node, indices, value_type, environment)
            self._infer_indices(node.slice, environment)
            if (
                isinstance(value_type, ConcreteType)
                and value_type.kind == "tensor_view"
                and value_type.arguments[2] == "write"
            ):
                raise self.error(node, "cannot load through a write-only TensorView")
            if isinstance(value_type, ConcreteType) and value_type.kind in {"tensor", "tensor_view", "list"}:
                if value_type.kind == "list" and isinstance(node.slice, ast.Tuple):
                    raise self.error(node.slice, "a list takes one index")
                element = value_type.arguments[0]
                assert isinstance(element, ConcreteType)
                return element
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in self.enums
            and node.value.id not in environment
        ):
            if node.attr not in self.enums[node.value.id]:
                raise self.error(node, f"enum '{node.value.id}' has no member '{node.attr}'")
            return ConcreteType("enum", node.value.id)
        if isinstance(node, ast.Attribute):
            value_type = self._expression(node.value, environment)
            if isinstance(value_type, ConcreteType) and value_type.kind == "storage":
                if node.attr != "shape":
                    raise self.error(node, f"TensorStorage has no attribute '{node.attr}'")
                extents = value_type.arguments[1]
                assert isinstance(extents, tuple)
                return ConcreteType("tuple", "Tuple", (_scalar("i32"),) * len(extents))
            if node.attr == "shape":
                rank = _shape_rank(value_type)
                if rank is None:
                    raise self.error(node, "shape requires a Tensor or TensorView")
                if rank < 1:
                    raise self.error(node, "shape requires rank >= 1")
                return ConcreteType("tensor", "Tensor", (_scalar("u32"), rank))
            if (
                isinstance(value_type, ConcreteType)
                and value_type.kind == "tensor"
                and set(node.attr) <= set("xyzwrgba")
            ):
                element = _element(value_type)
                return element if len(node.attr) == 1 else ConcreteType("tensor", "Tensor", (element, len(node.attr)))
            if isinstance(value_type, ConcreteType) and value_type.kind == "struct":
                fields = dict(self.structs.get(value_type.name, ()))
                if node.attr in fields:
                    return fields[node.attr]
        if isinstance(node, ast.Call):
            return self._call(node, environment)
        raise self.error(node, f"cannot infer expression syntax: {type(node).__name__}")

    def _storage_element(
        self, node: ast.Subscript, storage: ConcreteType, environment: dict[str, InferenceType]
    ) -> ConcreteType:
        """The element `storage[i, j, ...]` reads or writes: one integer index per axis, as numpy indexes an
        element."""

        element, extents = storage.arguments
        assert isinstance(element, ConcreteType) and isinstance(extents, tuple)
        indices = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
        if any(isinstance(index, ast.Starred) for index in indices):
            raise self.error(node.slice, f"{storage.spelling} takes one index or slice per axis")
        if is_slicing(indices, len(extents)) or len(indices) > len(extents):
            return self._selection_type(node, indices, storage, environment)
        self._infer_indices(node.slice, environment)
        return element

    def _selection_type(
        self,
        node: ast.Subscript,
        indices: list[ast.expr],
        container: ConcreteType,
        environment: dict[str, InferenceType],
    ) -> ConcreteType:
        """The type `v[...]` selects from a Tensor value or a TensorStorage with slices among its indices: the
        element for no kept axis, otherwise a Tensor of the kept lengths."""

        element = container.arguments[0]
        assert isinstance(element, ConcreteType)
        extents = (
            [int(extent) for extent in container.arguments[1:]]  # type: ignore[arg-type]
            if container.kind == "tensor"
            else [extent if isinstance(extent, int) else None for extent in container.arguments[1]]  # type: ignore[union-attr]
        )
        if element.kind != "scalar":
            raise self.error(node, f"slicing selects scalar elements; {container.spelling} has {element.spelling}")
        shape = selection_shape(self._selection_plan(node, indices, extents, environment))
        return ConcreteType("tensor", "Tensor", (element, *shape)) if shape else element

    def _selection_plan(
        self,
        node: ast.Subscript,
        indices: list[ast.expr],
        extents: list[int | None],
        environment: dict[str, InferenceType],
    ) -> list[AxisSelection]:
        try:
            plan = plan_selection(node, indices, extents)
        except SelectionError as error:
            raise self.error(error.node, str(error)) from None
        for axis, extent in zip(plan, extents, strict=True):
            if isinstance(axis, AxisRange):
                continue
            index_node = axis.node if isinstance(axis, AxisIndex) else axis.start
            if isinstance(axis, AxisWindow) and any(isinstance(part, ast.Call) for part in ast.walk(index_node)):
                # Python evaluates the start twice, once in each bound, so it must not call anything.
                raise self.error(index_node, "a window start has no calls; bind the start to a local first")
            if not default_type(self._expression(index_node, environment)).is_integer:
                raise self.error(index_node, "Tensor indices are integers")
            literal = constant_integer(index_node) if isinstance(axis, AxisIndex) else None
            if literal is not None and extent is not None and not -extent <= literal < extent:
                raise self.error(index_node, f"index {literal} is outside an axis of extent {extent}")
        starts = {id(axis.start) for axis in plan if isinstance(axis, AxisWindow)}
        for index in indices:
            if isinstance(index, ast.Slice):
                for bound in (index.lower, index.upper, index.step):
                    if bound is not None and id(bound) not in starts:
                        self._expression(bound, environment)
        return plan

    def _require_block(self, node: ast.expr, value: InferenceType, block: ConcreteType, what: str) -> None:
        """A value written into a selection of type `block`: a scalar for every element, or a Tensor of the
        selection's shape."""

        element = block.arguments[0] if block.kind == "tensor" else block
        assert isinstance(element, ConcreteType)
        if block.kind == "tensor" and isinstance(value, ConcreteType) and value.kind == "tensor":
            if value.arguments[1:] != block.arguments[1:]:
                raise self.error(node, f"cannot {what} {describe(value)} into a selection of {block.spelling}")
            value_element = value.arguments[0]
            assert isinstance(value_element, ConcreteType)
            if not can_convert(value_element, element):
                raise self.error(node, f"cannot {what} {describe(value)} into a selection of {block.spelling}")
            return
        if not can_convert(value, element):
            raise self.error(node, f"cannot {what} {describe(value)} into {block.spelling}")
        self._constrain_literal(node, element)

    def _storage_creation(self, node: ast.Call, environment: dict[str, InferenceType]) -> ConcreteType | None:
        """The type of `zeros(dtype=T, shape=(...))`, `empty(...)`, `zeros_like(s)`, `empty_like(s)` or
        `from_values(t, dtype=T)`, spelled through `vd`, `vd.storage` or `vd.TensorStorage`; None when `node` is none
        of them."""

        name = (dotted_name(node.func) or "").split(".")[-1]
        if name not in {"zeros", "empty", "zeros_like", "empty_like", "from_values"}:
            return None
        if isinstance(node.func, ast.Attribute) and self._receiver(node.func.value, environment):
            return None
        if name == "from_values":
            if len(node.args) != 1 or [keyword.arg for keyword in node.keywords] != ["dtype"]:
                raise self.error(node, "from_values takes a Tensor and dtype=, as in from_values(t, dtype=f32)")
            element = self.parse_type(node.keywords[0].value)
            values = self._expression(node.args[0], environment)
            if not isinstance(values, ConcreteType) or values.kind != "tensor" or values.arguments[0] != element:
                raise self.error(
                    node.args[0],
                    f"from_values takes a Tensor of {element.spelling} elements; {describe(values)} is not",
                )
            if element.name not in HOST_STORAGE_ELEMENTS:
                raise self.error(
                    node.keywords[0].value,
                    f"Host TensorStorage elements are bool, i32, u32, f32 or f64; {element.spelling} is not one",
                )
            return ConcreteType("storage", "TensorStorage", (element, tuple(values.arguments[1:])))
        if name.endswith("_like"):
            if len(node.args) != 1 or node.keywords:
                raise self.error(node, f"{name} takes one TensorStorage")
            source = self._expression(node.args[0], environment)
            if not isinstance(source, ConcreteType) or source.kind != "storage":
                raise self.error(node.args[0], f"{name} takes a TensorStorage; {describe(source)} is not one")
            return source
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        if node.args or set(keywords) != {"dtype", "shape"}:
            raise self.error(
                node, f"{name} takes only the keywords dtype= and shape=, as in {name}(dtype=f32, shape=(n,))"
            )
        element = self.parse_type(keywords["dtype"])
        if element.kind != "scalar" or element.name not in HOST_STORAGE_ELEMENTS:
            raise self.error(
                keywords["dtype"],
                f"Host TensorStorage elements are bool, i32, u32, f32 or f64; {element.spelling} is not one",
            )
        shape = keywords["shape"]
        extents: list[int | str] = []
        if isinstance(shape, ast.Tuple):
            for extent in shape.elts:
                extent_type = self._expression(extent, environment)
                if not default_type(extent_type).is_integer:
                    raise self.error(extent, "TensorStorage extents are integers")
                self._constrain_literal(extent, default_type(extent_type))
                if (
                    isinstance(extent, ast.UnaryOp)
                    and isinstance(extent.op, ast.USub)
                    and isinstance(extent.operand, ast.Constant)
                    and extent.operand.value != 0
                ):
                    raise self.error(extent, "TensorStorage extents cannot be negative")
                static = isinstance(extent, ast.Constant) and isinstance(extent.value, int) and extent.value > 0
                extents.append(extent.value if static else "?")  # type: ignore[union-attr]
        else:
            shape_type = self._expression(shape, environment)
            if (
                not isinstance(shape_type, ConcreteType)
                or shape_type.kind != "tuple"
                or not all(isinstance(item, ConcreteType) and item.is_integer for item in shape_type.arguments)
            ):
                raise self.error(shape, "shape is a tuple of integers, such as (h, w) or another storage's shape")
            extents = ["?"] * len(shape_type.arguments)
        if not 1 <= len(extents) <= HOST_STORAGE_MAX_RANK:
            raise self.error(shape, f"a Host TensorStorage has 1 to {HOST_STORAGE_MAX_RANK} axes")
        return ConcreteType("storage", "TensorStorage", (element, tuple(extents)))

    def _storage_call(
        self, node: ast.Call, receiver: ConcreteType, environment: dict[str, InferenceType]
    ) -> InferenceType:
        assert isinstance(node.func, ast.Attribute)
        method = node.func.attr
        element, extents = receiver.arguments
        assert isinstance(element, ConcreteType) and isinstance(extents, tuple)
        arities = {"fill": 1, "copy_from_values": 1, "to_values": 0, "copy": 0}
        if method not in arities:
            raise self.error(node.func, f"TensorStorage has no method '{method}' in Host code")
        if len(node.args) != arities[method] or node.keywords:
            count = arities[method]
            raise self.error(node, f"TensorStorage.{method} takes {count} argument{'s' if count != 1 else ''}")
        if method == "copy":
            return receiver
        if method == "to_values":
            if "?" in extents:
                raise self.error(
                    node, f"to_values returns a Tensor, whose shape is static; {receiver.spelling} has dyn extents"
                )
            return ConcreteType("tensor", "Tensor", (element, *extents))
        value = self._expression(node.args[0], environment)
        if method == "fill":
            if not can_convert(value, element):
                raise self.error(node.args[0], f"cannot fill {receiver.spelling} with {describe(value)}")
            self._constrain_literal(node.args[0], element)
            return ConcreteType("void", "void")
        if (
            not isinstance(value, ConcreteType)
            or value.kind != "tensor"
            or value.arguments[0] != element
            or len(value.arguments) - 1 != len(extents)
            or any(wanted not in {"?", given} for given, wanted in zip(value.arguments[1:], extents, strict=True))
        ):
            raise self.error(
                node.args[0],
                f"copy_from_values into {receiver.spelling} takes a Tensor of its element type and shape; "
                f"{describe(value)} is not one",
            )
        return ConcreteType("void", "void")

    def _infer_indices(self, node: ast.expr, environment: dict[str, InferenceType]) -> None:
        indices = list(node.elts) if isinstance(node, ast.Tuple) else [node]
        for index in indices:
            if isinstance(index, ast.Slice):
                raise self.error(index, "slices select from a Tensor value or a TensorStorage")
            index_type = self._expression(index, environment)
            concrete = default_type(index_type)
            if not concrete.is_integer:
                raise self.error(index, "index must be an integer")

    @staticmethod
    def _operation(node: ast.expr) -> str | None:
        if isinstance(node, ast.BinOp):
            return {
                ast.Add: "add",
                ast.Sub: "sub",
                ast.Mult: "mul",
                ast.Div: "div",
                ast.FloorDiv: "floordiv",
                ast.Mod: "mod",
                ast.Pow: "pow",
                ast.BitAnd: "bitand",
                ast.BitOr: "bitor",
                ast.BitXor: "bitxor",
                ast.LShift: "lshift",
                ast.RShift: "rshift",
            }.get(type(node.op))
        if isinstance(node, ast.UnaryOp):
            return {ast.Not: "not", ast.USub: "neg", ast.Invert: "invert"}.get(type(node.op), "identity")
        if isinstance(node, ast.BoolOp):
            return "and" if isinstance(node.op, ast.And) else "or"
        if isinstance(node, ast.IfExp):
            return "conditional"
        if isinstance(node, ast.Compare):
            return type(node.ops[0]).__name__.lower() if node.ops else "compare"
        if isinstance(node, ast.Subscript):
            return "index"
        if isinstance(node, ast.Attribute):
            return "attribute"
        if isinstance(node, ast.Call):
            return (dotted_name(node.func) or "").split(".")[-1]
        if isinstance(node, ast.Tuple):
            return "tuple"
        if isinstance(node, ast.Constant):
            return "constant"
        if isinstance(node, ast.Name):
            return "name"
        return None

    def _constrain_literal(self, node: ast.expr, expected: InferenceType) -> None:
        if not isinstance(expected, ConcreteType):
            return
        tuple_elements = (
            node.elts
            if isinstance(node, ast.Tuple)
            else node.args
            if isinstance(node, ast.Call) and (dotted_name(node.func) or "").split(".")[-1] == "Tuple"
            else None
        )
        if tuple_elements is not None and expected.kind == "tuple" and len(tuple_elements) == len(expected.arguments):
            for element, element_expected in zip(tuple_elements, expected.arguments, strict=True):
                assert isinstance(element_expected, ConcreteType)
                self._constrain_literal(element, element_expected)
            self.expression_records[id(node)] = (node, expected, "Tuple" if isinstance(node, ast.Call) else "tuple")
            return
        record = self.expression_records.get(id(node))
        if record is None:
            return
        source, inferred, operation = record
        if not isinstance(inferred, ConcreteType):
            concrete = contextualize(inferred, expected)
            self.expression_records[id(node)] = (source, concrete, operation)
            # A literal-typed operator takes its type from its literal operands, which lowering emits with their
            # own records, so the context has to reach them too.
            if isinstance(node, ast.BinOp):
                self._constrain_literal(node.left, element_type(concrete))
                self._constrain_literal(node.right, element_type(concrete))
            elif isinstance(node, ast.UnaryOp) and not isinstance(node.op, ast.Not):
                self._constrain_literal(node.operand, concrete)

    def _receiver(self, node: ast.expr, environment: dict[str, InferenceType]) -> bool:
        """Whether `node` is a value expression rather than a module path such as `vd.texture_sample`."""
        while isinstance(node, ast.Attribute):
            node = node.value
        return not isinstance(node, ast.Name) or node.id in environment

    def _resolve_method(self, node: ast.Call, environment: dict[str, InferenceType]) -> bool:
        if not isinstance(node.func, ast.Attribute) or not self._receiver(node.func.value, environment):
            return False
        receiver = self._expression(node.func.value, environment)
        if not isinstance(receiver, ConcreteType) or receiver.kind not in {"struct", "handle"}:
            return False
        # A handle's methods are the externs its class body declares, which take it first.
        helper = self.methods.get(receiver.name, {}).get(node.func.attr)
        if helper is None:
            chain = " -> ".join(key[0] for key in self.active)
            suffix = f" (in helper specialization {chain})" if chain else ""
            raise self.error(node.func, f"'{receiver.name}' has no method '{node.func.attr}'{suffix}")
        # A generic extern method was specialized for its type argument, which the call then drops.
        specialized = getattr(node, "_vernon_method_generics", {}).get(helper)
        if specialized is not None:
            helper, position = specialized
            del node.args[position]
        node.args = [node.func.value, *node.args]
        node.func = copy_location(ast.Name(id=helper, ctx=ast.Load()), node.func)
        return True

    def _list_call(self, node: ast.Call, environment: dict[str, InferenceType]) -> InferenceType | None:
        """The type of `len(x)` or of a method call on a list or dict `x`; None when `node` is neither."""

        if dotted_name(node.func) == "len" and len(node.args) == 1 and not node.keywords:
            argument = self._expression(node.args[0], environment)
            if isinstance(argument, ConcreteType) and argument.kind in {"list", "str", "dict", "storage"}:
                return _scalar("i32")
            raise self.error(node.args[0], "len requires a list, str, dict or TensorStorage")
        if (created := self._storage_creation(node, environment)) is not None:
            return created
        if dotted_name(node.func) in {"deepcopy", "copy.deepcopy"} and "deepcopy" not in environment:
            if len(node.args) != 1 or node.keywords:
                raise self.error(node, "deepcopy takes one class instance, list, dict or TensorStorage")
            source = self._expression(node.args[0], environment)
            copyable = isinstance(source, ConcreteType) and (
                source.kind in {"list", "dict", "storage"}
                or (source.kind == "struct" and source.name in self.host_classes)
            )
            if not copyable:
                raise self.error(
                    node.args[0],
                    f"deepcopy copies a class instance, list, dict or TensorStorage; {describe(source)} is a value, "
                    "which needs no copy",
                )
            return source
        if not isinstance(node.func, ast.Attribute) or not self._receiver(node.func.value, environment):
            return None
        receiver = self._expression(node.func.value, environment)
        if isinstance(receiver, ConcreteType) and receiver.kind == "dict":
            return self._dict_call(node, receiver, environment)
        if isinstance(receiver, ConcreteType) and receiver.kind == "storage":
            return self._storage_call(node, receiver, environment)
        if not isinstance(receiver, ConcreteType) or receiver.kind != "list":
            return None
        element = receiver.arguments[0]
        assert isinstance(element, ConcreteType)
        arity = {"append": 1, "pop": 0, "clear": 0, "copy": 0}.get(node.func.attr)
        if arity is None:
            raise self.error(node.func, f"list has no method '{node.func.attr}'")
        if len(node.args) != arity or node.keywords:
            raise self.error(node, f"list.{node.func.attr} takes {arity} argument{'s' if arity != 1 else ''}")
        if node.func.attr == "copy":
            return receiver
        if node.func.attr == "append":
            value = self._expression(node.args[0], environment)
            if not can_convert(value, element):
                raise self.error(node.args[0], f"cannot append {describe(value)} to {receiver.spelling}")
            self._constrain_literal(node.args[0], element)
            return ConcreteType("void", "void")
        return element if node.func.attr == "pop" else ConcreteType("void", "void")

    def _dict_call(
        self, node: ast.Call, receiver: ConcreteType, environment: dict[str, InferenceType]
    ) -> InferenceType:
        assert isinstance(node.func, ast.Attribute)
        method = node.func.attr
        value = receiver.arguments[1]
        assert isinstance(value, ConcreteType)
        arities = {"get": (2, 2), "pop": (1, 2), "clear": (0, 0), "copy": (0, 0)}
        if method in {"keys", "values", "items"}:
            raise self.error(node, f"dict.{method}() is iterated by a for loop; it is not a value")
        if method not in arities:
            raise self.error(node.func, f"dict has no method '{method}'")
        low, high = arities[method]
        if not low <= len(node.args) <= high or node.keywords:
            expected = str(low) if low == high else f"{low} or {high}"
            raise self.error(node, f"dict.{method} takes {expected} argument{'s' if high != 1 else ''}")
        if method == "clear":
            return ConcreteType("void", "void")
        if method == "copy":
            return receiver
        self._dict_key(node.args[0], receiver, environment)
        if len(node.args) == 2:
            default = self._expression(node.args[1], environment)
            if not can_convert(default, value):
                raise self.error(node.args[1], f"the default {describe(default)} is not a {value.spelling}")
            self._constrain_literal(node.args[1], value)
        return value

    def _reject_value_write(self, target: ast.Subscript, container: InferenceType) -> None:
        if isinstance(container, ConcreteType) and container.kind == "tensor":
            name = ast.unparse(target.value)
            index = ast.unparse(target.slice)
            raise self.error(
                target,
                f"{name} is a Tensor value, which is never written into; build a new one with "
                f"{name} = {name}.at[{index}].set(x), or hold the elements in a TensorStorage",
            )

    def _tensor_update(self, node: ast.Call, environment: dict[str, InferenceType]) -> InferenceType | None:
        """The type of `v.at[...].set(x)` or `.add(x)`, a new Tensor with the selected elements replaced; None when
        `node` is not one. The `at` attribute and its subscript are syntax only and get no type of their own."""

        update = tensor_update_parts(node)
        if update is None:
            return None
        tensor_node, index_nodes, method = update
        tensor = self._expression(tensor_node, environment)
        if not isinstance(tensor, ConcreteType) or tensor.kind != "tensor":
            raise self.error(tensor_node, f"only a Tensor value has .at[...]; {describe(tensor)} is not one")
        if method not in {"set", "add"}:
            raise self.error(node.func, f".at[...] has set(x) and add(x), not {method}")
        if len(node.args) != 1 or node.keywords:
            raise self.error(node, f".at[...].{method} takes one value")
        element = tensor.arguments[0]
        assert isinstance(element, ConcreteType)
        if element.kind != "scalar":
            raise self.error(tensor_node, f".at[...] updates scalar elements; {tensor.spelling} has {element.spelling}")
        if method == "add" and element.name == "bool":
            raise self.error(node.func, ".at[...].add needs numeric elements")
        assert isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Subscript)
        extents = [int(extent) for extent in tensor.arguments[1:]]  # type: ignore[arg-type]
        shape = selection_shape(self._selection_plan(node.func.value, index_nodes, extents, environment))
        block = ConcreteType("tensor", "Tensor", (element, *shape)) if shape else element
        self._require_block(node.args[0], self._expression(node.args[0], environment), block, method)
        return tensor

    def _struct_replace(self, node: ast.Call, environment: dict[str, InferenceType]) -> InferenceType:
        if len(node.args) != 1 or not node.keywords or any(keyword.arg is None for keyword in node.keywords):
            raise self.error(node, "replace takes a Struct and the fields to change, as in replace(p, x=1.0)")
        struct = self._expression(node.args[0], environment)
        if not isinstance(struct, ConcreteType) or struct.kind != "struct" or struct.name not in self.structs:
            raise self.error(node.args[0], f"replace takes a Struct value; {describe(struct)} is not one")
        fields = dict(self.structs[struct.name])
        for keyword in node.keywords:
            expected = fields.get(keyword.arg or "")
            if expected is None:
                raise self.error(keyword.value, f"{struct.name} has no field '{keyword.arg}'")
            value = self._expression(keyword.value, environment)
            if not can_convert(value, expected):
                raise self.error(keyword.value, f"cannot pass {describe(value)} as {expected.spelling}")
            self._constrain_literal(keyword.value, expected)
        return struct

    def _field_keywords(
        self, node: ast.Call, name: str, environment: dict[str, InferenceType], positional: set[str]
    ) -> set[str]:
        """Types the `field=value` keywords of a constructor of `name`; `positional` are the fields set before."""

        fields = dict(self.structs[name])
        named: set[str] = set()
        for keyword in node.keywords:
            if keyword.arg is None:
                raise self.error(keyword.value, f"{name}(...) names each field it sets")
            expected = fields.get(keyword.arg)
            if expected is None:
                raise self.error(keyword.value, f"{name} has no field '{keyword.arg}'")
            if keyword.arg in positional or keyword.arg in named:
                raise self.error(keyword.value, f"{name}(...) sets field '{keyword.arg}' twice")
            named.add(keyword.arg)
            value = self._expression(keyword.value, environment)
            if not can_convert(value, expected):
                raise self.error(keyword.value, f"cannot pass {describe(value)} as {expected.spelling}")
            self._constrain_literal(keyword.value, expected)
        return named

    def _call(self, node: ast.Call, environment: dict[str, InferenceType]) -> InferenceType:
        if (update := self._tensor_update(node, environment)) is not None:
            return update
        if self.host and (container := self._list_call(node, environment)) is not None:
            return container
        self._resolve_method(node, environment)
        resolved_name = (dotted_name(node.func) or "").split(".")[-1]
        name = getattr(node, "_vernon_generic_name", resolved_name)
        if name in {"Tensor", "Vector", "Matrix"}:
            return self._aggregate(node, environment, name)
        if name == "workgroup_storage":
            if len(node.args) != 1 or len(node.keywords) != 1 or node.keywords[0].arg != "shape":
                raise self.error(node, "workgroup_storage requires an element type and shape=(...)")
            element = self.parse_type(node.args[0])
            if not is_abi_stable_value(element, lambda struct: tuple(value for _, value in self.structs[struct])):
                raise self.error(node.args[0], "workgroup_storage element must be an ABI-stable Value")
            shape_node = node.keywords[0].value
            if not isinstance(shape_node, ast.Tuple):
                raise self.error(shape_node, "workgroup_storage shape must be a tuple")
            shape: list[int] = []
            for extent in shape_node.elts:
                if (
                    not isinstance(extent, ast.Constant)
                    or not isinstance(extent.value, int)
                    or isinstance(extent.value, bool)
                    or extent.value <= 0
                ):
                    raise self.error(extent, "workgroup_storage dimensions must be positive compile-time integers")
                shape.append(extent.value)

            shape_tuple = tuple(shape)
            footprint = workgroup_physical_bytes(element, shape_tuple, self.structs.__getitem__)
            if footprint > 16 * 1024:
                raise self.error(node, "combined workgroup storage exceeds the portable 16 KiB workgroup storage limit")
            if self.workgroup_storage_bytes > 16 * 1024 - footprint:
                raise self.error(node, "combined workgroup storage exceeds the portable 16 KiB workgroup storage limit")
            self.workgroup_storage_bytes += footprint
            return ConcreteType("tensor_view", "TensorView", (element, shape_tuple, "read_write", "workgroup"))
        if name in ATOMIC_OPERATION_NAMES:
            if len(node.args) != 3 or node.keywords:
                raise self.error(node, f"{name} requires storage, index, and value")
            if not isinstance(node.args[0], ast.Name):
                raise self.error(node.args[0], f"{name} requires a named storage owner")
            storage = self._expression(node.args[0], environment)
            index = default_type(self._expression(node.args[1], environment))
            value = self._expression(node.args[2], environment)
            if not isinstance(storage, ConcreteType) or storage.kind != "tensor_view":
                raise self.error(node.args[0], f"{name} requires a writable TensorView")
            element, shape, _, _ = storage.arguments
            assert isinstance(element, ConcreteType)
            assert isinstance(shape, tuple)
            if storage.arguments[2] == "read":
                raise self.error(node.args[0], f"{name} requires a writable TensorView")
            supported_elements = {"i32", "u32", "f32", "f64"} if name == "atomic_add" else {"i32", "u32"}
            if element.kind != "scalar" or element.name not in supported_elements:
                expected = "i32, u32, f32, or f64" if name == "atomic_add" else "i32 or u32"
                raise self.error(node.args[0], f"{name} requires {expected} storage elements")
            index_nodes = list(node.args[1].elts) if isinstance(node.args[1], ast.Tuple) else [node.args[1]]
            if len(shape) == 1:
                if len(index_nodes) != 1 or not index.is_integer:
                    raise self.error(node.args[1], f"{name} rank-one index must be an integer")
            elif not isinstance(node.args[1], ast.Tuple) or len(index_nodes) != len(shape):
                raise self.error(node.args[1], f"{name} requires one tuple index per TensorView dimension")
            else:
                for index_node in index_nodes:
                    if not default_type(self._expression(index_node, environment)).is_integer:
                        raise self.error(index_node, f"{name} indices must be integers")
            if not can_convert(value, element):
                raise self.error(node.args[2], f"{name} value must be {element.spelling}")
            self._constrain_literal(node.args[2], element)
            return element
        if name in {"workgroup_barrier", "storage_barrier", FRAGMENT_DISCARD_OPERATION}:
            if node.args or node.keywords:
                raise self.error(node, f"{name} does not accept arguments")
            return ConcreteType("void", "void")
        arguments = [self._expression(argument, environment) for argument in node.args]
        if name == "Tuple":
            if not arguments:
                raise self.error(node, "Tuple values must contain at least one element")
            return ConcreteType("tuple", "Tuple", tuple(default_type(argument) for argument in arguments))
        if (
            isinstance(node.func, ast.Attribute)
            and name in {"norm", "normalize"}
            and (not isinstance(node.func.value, ast.Name) or node.func.value.id in environment)
        ):
            arguments.insert(0, self._expression(node.func.value, environment))
        if name in self.value_generics:
            index = self.value_generics[name]
            if index >= len(arguments):
                raise self.error(node, f"'{name}' takes a @struct value at position {index}")
            value = arguments[index]
            if (
                not isinstance(value, ConcreteType)
                or value.kind != "struct"
                or f"{name}__{value.name}" not in self.functions
            ):
                raise self.error(node.args[index], f"'{name}' takes a @struct value, not {describe(value)}")
            name = f"{name}__{value.name}"
            node.func = copy_location(ast.Name(id=name, ctx=ast.Load()), node.func)
        if name in self.generic:
            symbol, result = self._specialize(name, tuple(arguments), node)
            node._vernon_generic_name = name
            node.func = copy_location(ast.Name(id=symbol, ctx=ast.Load()), node.func)
            if result is None:
                return ConcreteType("void", "void")
            return result
        if name in self.specialized_signatures:
            parameter_types, result = self.specialized_signatures[name]
            if len(arguments) != len(parameter_types):
                raise self.error(node, f"function '{name}' expects {len(parameter_types)} arguments")
            for source, inferred, expected in zip(node.args, arguments, parameter_types, strict=True):
                if not can_convert(inferred, expected):
                    raise self.error(node, f"cannot pass {describe(inferred)} as {expected.spelling}")
                self._constrain_literal(source, expected)
            return ConcreteType("void", "void") if result is None else result
        if name in self.functions:
            function = self.functions[name]
            if len(arguments) != len(function.args.args):
                raise self.error(node, f"function '{name}' expects {len(function.args.args)} arguments")
            for source, argument, parameter in zip(node.args, arguments, function.args.args, strict=True):
                if parameter.annotation is None:
                    raise self.error(
                        parameter, f"function '{name}' parameter {parameter.arg!r} requires a type annotation"
                    )
                expected = self.parse_type(parameter.annotation)
                if not can_convert(argument, expected):
                    raise self.error(node, f"cannot pass {describe(argument)} as {expected.spelling}")
                self._constrain_literal(source, expected)
            result = function.returns
            return (
                ConcreteType("void", "void")
                if result is None or (isinstance(result, ast.Constant) and result.value is None)
                else self.parse_type(result)
            )
        if name in self.handles:
            if arguments:
                raise self.error(node, f"handle '{name}' constructs only the null handle and takes no arguments")
            return ConcreteType("handle", name)
        if name in {"int", "i32"}:
            return _scalar("i32")
        if (
            self.host
            and name in {"i64", "u64", "u32"}
            and len(arguments) == 1
            and isinstance(arguments[0], LiteralType)
        ):
            # Host literals beyond i32 are only writable as the argument of a wider cast.
            self._constrain_literal(node.args[0], _scalar(name))
        if name == "u32" or (self.host and name in {"i64", "u64", "bool"}):
            return _scalar(name)
        if name in {"float", "f32"}:
            return _scalar("f32")
        if name in {"f16", "f64"}:
            return _scalar(name)
        if name in GENERATED_INTERFACE_CONTRACTS:
            if arguments:
                raise self.error(node, f"{name} does not accept arguments")
            return _contract_type(GENERATED_INTERFACE_CONTRACTS[name].type)
        if name == "replace" and name not in self.structs:
            return self._struct_replace(node, environment)
        if name in self.host_classes:
            if arguments:
                raise self.error(
                    node, f"{name}() names the fields it sets, as in {name}(field=value); the rest take their defaults"
                )
            self._field_keywords(node, name, environment, set())
            return ConcreteType("struct", name)
        if name in self.structs:
            fields = self.structs[name]
            if len(arguments) > len(fields):
                raise self.error(node, f"{name} constructor requires {len(fields)} arguments")
            for source, inferred, (_, expected) in zip(node.args, arguments, fields, strict=False):
                if not can_convert(inferred, expected):
                    raise self.error(node, f"cannot pass {describe(inferred)} as {expected.spelling}")
                self._constrain_literal(source, expected)
            if node.keywords and not self.host:
                raise self.error(node, f"device code constructs {name} from positional field values")
            positional = {field for field, _ in fields[: len(arguments)]}
            named = self._field_keywords(node, name, environment, positional)
            missing = [field for field, _ in fields if field not in positional and field not in named]
            if missing:
                raise self.error(node, f"{name} constructor requires {len(fields)} arguments; '{missing[0]}' is unset")
            return ConcreteType("struct", name)
        if name == "matmul":
            if len(arguments) != 2:
                raise self.error(node, "matmul requires two arguments")
            left, right = arguments
            if not isinstance(left, ConcreteType) or left.kind != "tensor":
                raise self.error(node, "matmul left operand must be a non-scalar Tensor")
            if not isinstance(right, ConcreteType) or right.kind != "tensor":
                raise self.error(node, "matmul right operand must be a non-scalar Tensor")
            result_shape = matmul_shape(left.arguments[1:], right.arguments[1:])
            element = common_type(_element(left), _element(right))
            if result_shape is None:
                raise self.error(node, "matmul operands have incompatible core or batch dimensions")
            if not isinstance(element, ConcreteType) or element.kind != "scalar":
                raise self.error(node, "matmul operands have incompatible element types")
            if result_shape:
                return ConcreteType("tensor", "Tensor", (element, *result_shape))
            return element
        if name in {"texture_load", "texture_store"}:
            expected_arguments = 2 if name == "texture_load" else 3
            if len(arguments) != expected_arguments or not isinstance(arguments[0], ConcreteType):
                raise self.error(node, f"{name} requires a texture and coordinates")
            texture = arguments[0]
            if texture.kind != "texture":
                raise self.error(node.args[0], f"{name} requires a Texture")
            dimension, element, _, access = texture.arguments
            if name == "texture_store" and access == "sampled":
                raise self.error(node.args[0], "texture_store requires a storage Texture")
            rank = TEXEL_COORDINATE_RANK.get(str(dimension))
            if rank is None:
                raise self.error(node.args[0], f"texture_load cannot address texels of a {dimension} texture")
            coordinates = arguments[1]
            coordinate_element = (
                coordinates.arguments[0]
                if isinstance(coordinates, ConcreteType) and coordinates.kind == "tensor"
                else None
            )
            if (
                not isinstance(coordinates, ConcreteType)
                or coordinates.kind != "tensor"
                or coordinates.arguments[1:] != (rank,)
                or not isinstance(coordinate_element, ConcreteType)
                or not coordinate_element.is_integer
            ):
                raise self.error(
                    node.args[1],
                    f"{name} coordinates for a {dimension} texture must be a {rank}-component integer vector",
                )
            assert isinstance(element, ConcreteType)
            texel = ConcreteType("tensor", "Tensor", (element, 4))
            if name == "texture_load":
                if access == "write":
                    raise self.error(node.args[0], "texture_load requires read or read_write access")
                return texel
            if access == "read":
                raise self.error(node.args[0], "texture_store requires write or read_write access")
            if not can_convert(arguments[2], texel):
                raise self.error(node.args[2], f"texture_store value must be {texel.spelling}")
            return ConcreteType("void", "void")
        if name == "texture_sample":
            if len(arguments) not in {2, 3, 4} or not isinstance(arguments[0], ConcreteType):
                raise self.error(node, "texture_sample requires a texture and coordinates")
            texture = arguments[0]
            if texture.kind != "texture" or texture.arguments[3] != "sampled":
                raise self.error(node, "texture_sample requires a texture and coordinates")
            coordinate_index = (
                2
                if len(arguments) >= 3 and isinstance(arguments[1], ConcreteType) and arguments[1].kind == "sampler"
                else 1
            )
            if coordinate_index >= len(arguments) or not isinstance(arguments[coordinate_index], ConcreteType):
                raise self.error(node, "texture_sample requires coordinates")
            element = texture.arguments[1]
            assert isinstance(element, ConcreteType)
            if element.is_integer:
                raise self.error(node.args[0], "integer textures are read with texture_load, not texture_sample")
            dimension = texture.arguments[0]
            rank = SAMPLE_COORDINATE_RANK.get(str(dimension))
            coordinates = arguments[coordinate_index]
            if rank is None or coordinates != ConcreteType("tensor", "Tensor", (element, rank)):
                raise self.error(
                    node,
                    f"texture_sample coordinates for a {dimension} texture "
                    f"must be a {rank}-component floating-point vector",
                )
            return ConcreteType("tensor", "Tensor", (element, 4))
        if name == "texture_sample_bias":
            if (
                len(arguments) != 4
                or not isinstance(arguments[0], ConcreteType)
                or arguments[0].kind != "texture"
                or arguments[0].arguments[3] != "sampled"
                or not isinstance(arguments[1], ConcreteType)
                or arguments[1].kind != "sampler"
            ):
                raise self.error(node, "texture_sample_bias requires a sampled texture, sampler, coordinates, and bias")
            texture = arguments[0]
            element = texture.arguments[1]
            assert isinstance(element, ConcreteType)
            if element.is_integer:
                raise self.error(node.args[0], "integer textures are read with texture_load, not texture_sample_bias")
            dimension = texture.arguments[0]
            rank = SAMPLE_COORDINATE_RANK.get(str(dimension))
            if rank is None or arguments[2] != ConcreteType("tensor", "Tensor", (element, rank)):
                raise self.error(
                    node.args[2],
                    f"texture_sample_bias coordinates for a {dimension} texture "
                    f"must be a {rank}-component floating-point vector",
                )
            bias = arguments[3]
            if isinstance(bias, ConcreteType) and (bias.kind != "scalar" or not bias.is_float):
                raise self.error(node.args[3], "texture_sample_bias bias must be a floating-point scalar")
            return ConcreteType("tensor", "Tensor", (element, 4))
        if name == "texture_size":
            if not arguments or not isinstance(arguments[0], ConcreteType) or arguments[0].kind != "texture":
                raise self.error(node, "texture_size requires a texture and optional lod")
            rank = SIZE_RANK[str(arguments[0].arguments[0])]
            return ConcreteType("tensor", "Tensor", (_scalar("u32"), rank))
        if name in {"any", "all"}:
            if len(arguments) != 1 or default_type(_element(arguments[0])) != _scalar("bool"):
                raise self.error(node, f"{name} requires one bool or bool Tensor argument")
            return _scalar("bool")
        if name == "select":
            if len(arguments) != 3:
                raise self.error(node, "select requires a condition and two values")
            condition = default_type(arguments[0])
            if default_type(_element(condition)) != _scalar("bool"):
                raise self.error(node.args[0], "select condition must be a bool or bool Tensor")
            result = _common(arguments[1], arguments[2])
            if result is None:
                raise self.error(node, "select values have incompatible types")
            result = default_type(result)
            if isinstance(condition, ConcreteType) and condition.kind == "tensor":
                shape = condition.arguments[1:]
                if isinstance(result, ConcreteType) and result.kind == "scalar":
                    result = ConcreteType("tensor", "Tensor", (result, *shape))
                elif not isinstance(result, ConcreteType) or result.kind != "tensor" or result.arguments[1:] != shape:
                    raise self.error(node, "select with a bool Tensor condition needs values of its shape")
            self._constrain_literal(node.args[1], _element(result))
            self._constrain_literal(node.args[2], _element(result))
            return result
        if name in {
            "sin",
            "cos",
            "acos",
            "atan2",
            "exp",
            "log",
            "sqrt",
            "floor",
            "abs",
            "normalize",
            "reflect",
            "min",
            "max",
            "pow",
            "clamp",
        }:
            if not arguments:
                raise self.error(node, f"{name} requires arguments")
            result = arguments[0]
            for argument in arguments[1:]:
                common = _common(result, argument)
                if common is None:
                    raise self.error(node, f"{name} arguments have incompatible types")
                result = common
            return result
        if name in {"dot", "norm"}:
            if not arguments:
                raise self.error(node, f"{name} requires arguments")
            return default_type(_element(arguments[0]))
        if name == "cross":
            return arguments[0]
        if name in self.enums:
            raise self.error(node, f"an enum is not constructed from an integer; name a member, {name}.<member>")
        raise self.error(node, f"cannot infer call to '{name}'")

    def _aggregate(
        self,
        node: ast.Call,
        environment: dict[str, InferenceType],
        name: str,
    ) -> ConcreteType:
        if len(node.args) != 1:
            raise self.error(node, f"{name} requires one sequence literal")
        if name == "Vector":
            sequence = node.args[0]
            if not isinstance(sequence, (ast.List, ast.Tuple)) or not sequence.elts:
                raise self.error(node, "Vector requires a non-empty sequence literal")
            values = [default_type(self._expression(value, environment)) for value in sequence.elts]
            if any(
                not isinstance(value, ConcreteType)
                or (value.kind != "scalar" and not (value.kind == "tensor" and len(value.arguments) == 2))
                for value in values
            ):
                raise self.error(node, "Vector elements must be scalars or rank-one Tensor Values")
            element = _element(values[0])
            count = 0
            for value in values:
                common = _common(element, _element(value))
                if common is None:
                    raise self.error(node, "Vector elements have incompatible types")
                element = common
                count += 1 if value.kind == "scalar" else int(value.arguments[1])
            element = default_type(element)
            for source, value in zip(sequence.elts, values, strict=True):
                expected = (
                    element
                    if value.kind == "scalar"
                    else ConcreteType("tensor", "Tensor", (element, value.arguments[1]))
                )
                self._constrain_literal(source, expected)
            return ConcreteType("tensor", "Tensor", (element, count))
        if name == "Tensor" and not isinstance(node.args[0], (ast.List, ast.Tuple)):
            element = default_type(self._expression(node.args[0], environment))
            if not is_abi_stable_value(element):
                raise self.error(node, "Tensor element has an incompatible type")
            self._constrain_literal(node.args[0], element)
            return ConcreteType("tensor", "Tensor", (element,))
        literal = rectangular_literal(node.args[0])
        if literal is None:
            raise self.error(node, f"{name} requires a non-empty rectangular sequence literal")
        elements, shape = literal
        expected_rank = {"Vector": 1, "Matrix": 2}.get(name)
        if expected_rank is not None and len(shape) != expected_rank:
            raise self.error(node, f"{name} requires a rank-{expected_rank} sequence literal")
        element = self._expression(elements[0], environment)
        for value in elements[1:]:
            common = _common(element, self._expression(value, environment))
            if common is None:
                raise self.error(node, f"{name} elements have incompatible types")
            element = common
        element = default_type(element)
        if not is_abi_stable_value(element):
            raise self.error(node, f"{name} elements have incompatible types")
        for value in elements:
            self._constrain_literal(value, element)
        return ConcreteType("tensor", "Tensor", (element, *shape))

    def _specialize(
        self,
        name: str,
        call_types: tuple[InferenceType, ...],
        call: ast.Call,
    ) -> tuple[str, ConcreteType | None]:
        function = self.functions[name]
        if len(call_types) != len(function.args.args):
            raise self.error(call, f"function '{name}' expects {len(function.args.args)} arguments")
        resolved_arguments: list[ConcreteType] = []
        for argument, inferred in zip(function.args.args, call_types, strict=True):
            expected = self.parse_type(argument.annotation) if argument.annotation is not None else None
            concrete = contextualize(inferred, expected)
            if expected is not None:
                if not can_convert(concrete, expected):
                    raise self.error(call, f"cannot pass {describe(inferred)} as {expected.spelling}")
                concrete = expected
            resolved_arguments.append(concrete)
        argument_types = tuple(resolved_arguments)
        for source, argument_type in zip(call.args, argument_types, strict=True):
            self._constrain_literal(source, argument_type)
        key = (name, argument_types, self.enabled_features)
        if key in self.active:
            raise self.error(call, f"recursive helper specialization for '{name}'")
        existing = self.instances.get(key)
        if existing is not None:
            return existing.name, self.instance_results[key]
        digest = hashlib.sha256(
            repr((name, tuple(value.mlir for value in argument_types), self.enabled_features)).encode("utf-8")
        ).hexdigest()[:12]
        clone = copy.deepcopy(function)
        clone.name = f"{name}__{digest}"
        clone._vernon_qualified_name = name
        clone.decorator_list = [ast.Name(id="func", ctx=ast.Load())]
        for argument, value_type in zip(clone.args.args, argument_types, strict=True):
            argument.annotation = copy_location(_annotation(value_type), argument)
        self.instances[key] = clone
        self.active.append(key)
        environment = {
            argument.arg: value_type for argument, value_type in zip(clone.args.args, argument_types, strict=True)
        }
        returns = self._statements(clone.body, environment)
        self.active.pop()
        if clone.returns is not None:
            result = (
                None
                if isinstance(clone.returns, ast.Constant) and clone.returns.value is None
                else self.parse_type(clone.returns)
            )
            if result is not None:
                for inferred in returns:
                    if inferred is None:
                        raise self.error(function, f"helper '{name}' mixes value and empty returns")
                    if not can_convert(inferred, result):
                        raise self.error(
                            function,
                            f"helper '{name}' cannot return {describe(inferred)} as {result.spelling}",
                        )
        elif not returns or all(value is None for value in returns):
            result = None
            clone.returns = ast.Constant(value=None)
        elif any(value is None for value in returns):
            raise self.error(function, f"helper '{name}' mixes value and empty returns")
        else:
            result = returns[0]
            assert result is not None
            for value in returns[1:]:
                assert value is not None
                common = _common(result, value)
                if common is None:
                    raise self.error(function, f"helper '{name}' has incompatible return types")
                result = common
            result = default_type(result)
            clone.returns = copy_location(_annotation(result), function)
        self.instance_results[key] = result
        self.specialized_signatures[clone.name] = (argument_types, result)
        return clone.name, result


def infer_and_monomorphize_helpers(
    module: ast.Module,
    parse_type: ParseType,
    parse_interface: ParseInterface,
    error: Error,
    enabled_features: tuple[str, ...],
    *,
    host: bool = False,
) -> ast.Module:
    return _Inference(module, parse_type, parse_interface, error, enabled_features, host=host).run()
