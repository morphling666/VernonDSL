"""Ownership of Host class instances, checked so compiled code and Python mode mean the same.

Python shares one object between every name and container slot that refers to it; compiled code holds an instance
in exactly one place and refers to it through pointers. The two agree when:

- a name bound to an existing place refers to it (a borrow), as Python's names do;
- a new instance (`A()`, a call returning one, `pop`) is moved into the one place that keeps it, and an owned local
  moved out is not used again, because Python would still share the object there;
- no place is assigned from another, which Python would make two names for one object;
- nothing a borrow refers into is replaced, grown or shrunk while the borrow is in use, because compiled code would
  then read freed or moved memory where Python still holds the old object.

A borrow is a class parameter, a local bound to a field, a `for` loop's element, or a call's argument. Places are
named by paths: the root name, then field names, with `*` for an element of a list or dict, since an index is not
known statically. A structural operation (replacing a class place, or appending to, popping from, clearing,
assigning or deleting an element of a list or dict of classes) at path T invalidates every borrow whose path starts
with T. A call may perform them through its class arguments, which the per-function summaries record.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass, field

from ..diagnostics import CompileError
from ..frontend.model import ConcreteType
from .model import HostFunctionInfo, HostProgram
from .values import element_of

Path = tuple[str, ...]
ELEMENT = "*"
# The kinds of class locals: an instance the function owns, a name for a field reached from a parameter, and a
# loop's element.
OWNED, BORROWED, ELEMENT_LOCAL = "owned", "borrowed", "element"


def _prefix(head: Path, path: Path) -> bool:
    return len(head) <= len(path) and path[: len(head)] == head


def _spell(path: Path) -> str:
    text = path[0]
    for step in path[1:]:
        text += "[...]" if step == ELEMENT else f".{step}"
    return text


def _function_name(symbol: str) -> str:
    """How source code names a function: a method's symbol `Class__method` as `Class.method`."""

    owner, separator, method = symbol.partition("__")
    return f"{owner}.{method}" if separator and owner and method and not method.startswith("_") else symbol


@dataclass
class _Op:
    """A structural operation at `path`: what replacing, growing or shrinking there invalidates."""

    path: Path
    node: ast.AST
    text: str


@dataclass
class _Call:
    callee: str
    # (parameter index, argument path or None for a new instance, argument node)
    operands: list[tuple[int, Path | None, ast.expr]]
    node: ast.Call


@dataclass
class FunctionOwnership:
    info: HostFunctionInfo
    ops: list[_Op] = field(default_factory=list)
    calls: list[_Call] = field(default_factory=list)
    # The container path each loop over a list or dict of classes iterates.
    loops: list[tuple[Path, ast.For]] = field(default_factory=list)
    borrowed: dict[str, tuple[Path, ast.AST]] = field(default_factory=dict)


class OwnershipChecker:
    def __init__(self, program: HostProgram, error: Callable[[ast.AST, str], CompileError]):
        self.program = program
        self.error = error
        self.functions: dict[str, FunctionOwnership] = {}

    # Types.

    def is_class(self, value_type: ConcreteType) -> bool:
        return self.program.is_place(value_type)

    def holds_classes(self, value_type: ConcreteType) -> bool:
        return value_type.kind in {"list", "dict"} and self.is_class(element_of(value_type))

    # Per function.

    def check(self, info: HostFunctionInfo, parents: dict[int, ast.AST]) -> None:
        self.info = info
        self.parents = parents
        self.record = FunctionOwnership(info)
        self.functions[info.symbol] = self.record
        self.params = {
            name: index for index, (name, value_type) in enumerate(info.parameters) if self.is_class(value_type)
        }
        self.body = [node for statement in info.source.body for node in ast.walk(statement)]
        typed_class = [
            node
            for node in self.body
            if isinstance(node, ast.expr) and id(node) in info.expressions and self.is_class(info.expression_type(node))
        ]
        if info.kind != "host":
            if typed_class:
                raise self.error(
                    typed_class[0], f"@func helper '{info.symbol}' is shared with device code; it uses no Host class"
                )
            return
        self.name_paths: dict[int, Path] = {}
        self.classify_locals()
        self.map_elements()
        for node in typed_class:
            if not isinstance(getattr(node, "ctx", None), (ast.Store, ast.Del)):
                self.check_position(node)
        self.collect()
        self.check_element_scopes()
        _MoveFlow(self).run(info.source.body)

    def local_kind(self, name: str) -> str | None:
        return self.info.class_locals.get(name)

    def classify_locals(self) -> None:
        info = self.info
        kinds = info.class_locals
        for node in self.body:
            if isinstance(node, ast.For) and isinstance(node.target, (ast.Name, ast.Tuple)):
                targets = node.target.elts if isinstance(node.target, ast.Tuple) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and self.is_class(info.local_types.get(target.id, _VOID)):
                        kinds[target.id] = ELEMENT_LOCAL
        for node in self.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
                continue
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                if not isinstance(target, ast.Name) or not self.is_class(info.local_types.get(target.id, _VOID)):
                    continue
                if target.id in self.params:
                    raise self.error(node, f"'{target.id}' is a parameter, a borrowed instance; it cannot be rebound")
                kind = OWNED if self.creates(node.value) or self.owned_name(node.value) else BORROWED
                previous = kinds.setdefault(target.id, kind)
                if previous == ELEMENT_LOCAL:
                    raise self.error(node, f"'{target.id}' is a loop's element; bind the instance to another name")
                if previous != kind:
                    raise self.error(
                        node,
                        f"'{target.id}' owns a new instance in one assignment and names an existing one in another; "
                        "use two names",
                    )
                if kind == BORROWED:
                    self.bind_borrow(target.id, node)

    def map_elements(self) -> None:
        """Gives each use of a loop's element the path `container[*]`. `ast.walk` reaches an outer loop before the
        loops in its body, so an inner loop over an element's container finds that element's path already."""

        for node in self.body:
            if not isinstance(node, ast.For) or id(_iterated(node.iter)) not in self.info.expressions:
                continue
            iterated = _iterated(node.iter)
            if not self.holds_classes(self.info.expression_type(iterated)):
                continue
            container = self.path(iterated)
            if container is None:
                continue
            target = node.target.elts[-1] if isinstance(node.target, ast.Tuple) else node.target
            if not isinstance(target, ast.Name):
                continue
            for statement in node.body:
                for use in ast.walk(statement):
                    if isinstance(use, ast.Name) and use.id == target.id:
                        self.name_paths[id(use)] = (*container, ELEMENT)

    def bind_borrow(self, name: str, node: ast.Assign | ast.AnnAssign) -> None:
        assert node.value is not None
        if name in self.record.borrowed:
            raise self.error(
                node, f"'{name}' already names {_spell(self.record.borrowed[name][0])}; a borrowed name is bound once"
            )
        root = node.value
        while isinstance(root, (ast.Attribute, ast.Subscript)):
            root = root.value
        if isinstance(root, ast.Name) and self.local_kind(root.id) == ELEMENT_LOCAL:
            raise self.error(
                node.value,
                f"'{name}' would name a loop's element, which is borrowed only by its loop; use "
                f"{ast.unparse(node.value)} directly",
            )
        path = self.path(node.value)
        if path is None:
            raise self.error(node.value, f"'{ast.unparse(node.value)}' is neither a new instance nor a place to name")
        if path[0] not in self.params:
            raise self.error(
                node.value,
                f"'{name}' would name a part of the local '{path[0]}'; use {ast.unparse(node.value)} directly",
            )
        if ELEMENT in path:
            raise self.error(
                node.value,
                f"'{name}' would name an element of a list or dict, which moves when the container changes; use "
                f"{ast.unparse(node.value)} directly, or iterate the container",
            )
        self.record.borrowed[name] = (path, node)

    # Expressions.

    def creates(self, node: ast.expr) -> bool:
        """Whether `node` evaluates to a new instance: a constructor, a call returning an instance, a pop, or a
        deepcopy."""

        if not isinstance(node, ast.Call) or not self.is_class(self.info.expression_type(node)):
            return False
        if isinstance(node.func, ast.Attribute):
            return node.func.attr in {"pop", "deepcopy"}
        return True

    def owned_name(self, node: ast.expr) -> bool:
        return isinstance(node, ast.Name) and self.local_kind(node.id) == OWNED

    def path(self, node: ast.expr) -> Path | None:
        """The path of a place expression, None for anything else."""

        if isinstance(node, ast.Name):
            if node.id in self.record.borrowed:
                return self.record.borrowed[node.id][0]
            if id(node) in self.name_paths:
                return self.name_paths[id(node)]
            local = self.info.local_types.get(node.id)
            if node.id in self.params or (local is not None and (self.is_class(local) or local.kind in _CONTAINERS)):
                return (node.id,)
            return None
        if isinstance(node, ast.Attribute):
            base = self.path(node.value) if id(node.value) in self.info.expressions else None
            if base is None or self.info.expression_type(node.value).kind != "struct":
                return None
            if not self.is_class(self.info.expression_type(node.value)):
                return None
            return (*base, node.attr)
        if isinstance(node, ast.Subscript) and id(node.value) in self.info.expressions:
            if self.holds_classes(self.info.expression_type(node.value)):
                base = self.path(node.value)
                return None if base is None else (*base, ELEMENT)
        return None

    def check_position(self, node: ast.expr) -> None:
        parent = self.parents.get(id(node))
        spelled = ast.unparse(node)
        created = self.creates(node)
        moved = self.owned_name(node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            raise self.error(node, "get would copy a class instance; test 'key in d' and index it, or pop it")
        if not created and not moved and self.path(node) is None:
            raise self.error(node, f"'{spelled}' is a Host class instance; name a place or create one")
        if isinstance(parent, ast.Attribute) and parent.value is node:
            if created:
                raise self.error(node, f"'{spelled}' creates an instance; bind it to a local before reading it")
            return
        if isinstance(parent, ast.Call) and node in parent.args:
            if isinstance(parent.func, ast.Attribute) and parent.func.attr == "append":
                self.require_new(node, f"appended to {ast.unparse(parent.func.value)}")
            elif self.taken(parent, node):
                # A specialization `name__Class` is spelled as the extern's name.
                self.require_new(node, f"moved into {parent.func.id.split('__')[0]}")  # type: ignore[attr-defined]
            return
        if isinstance(parent, (ast.Assign, ast.AnnAssign)) and parent.value is node:
            targets = parent.targets if isinstance(parent, ast.Assign) else [parent.target]
            if all(isinstance(target, ast.Name) for target in targets):
                if moved and any(target.id == node.id for target in targets):  # type: ignore[attr-defined]
                    raise self.error(parent, f"'{spelled}' is assigned to itself")
                return
            self.require_new(node, f"stored in {ast.unparse(targets[0])}")
            return
        if isinstance(parent, ast.Return) and parent.value is node:
            self.require_new(node, "returned")
            return
        if isinstance(parent, ast.Expr) and created:
            return
        if isinstance(parent, ast.keyword):
            self.require_new(node, f"moved into field '{parent.arg}'")
            return
        raise self.error(
            node,
            f"'{spelled}' is a Host class instance; read its fields, call its methods, pass it to a @host function, "
            "or move a new one into a place",
        )

    def taken(self, call: ast.Call, argument: ast.expr) -> bool:
        """Whether the call is to an extern taking the instance `argument` (an extern type variable bound to
        `vd.HostClass`), which moves it."""

        if not isinstance(call.func, ast.Name) or call.func.id not in self.program.externs:
            return False
        parameters = self.program.externs[call.func.id].parameters
        index = next(position for position, item in enumerate(call.args) if item is argument)
        return index < len(parameters) and self.is_class(parameters[index][1])

    def require_new(self, node: ast.expr, where: str) -> None:
        if self.creates(node) or self.owned_name(node):
            return
        raise self.error(
            node,
            f"'{ast.unparse(node)}' names an existing instance, which Python would share rather than copy once "
            f"{where}; store a new instance (a constructor, a call returning one, a pop or copy.deepcopy(...)) or "
            "an owned local",
        )

    # Structural operations and calls.

    def collect(self) -> None:
        info = self.info
        for node in self.body:
            if isinstance(node, ast.Call):
                self.collect_call(node)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                    self.collect_store(target, node)
            elif isinstance(node, ast.Delete):
                for target in node.targets:
                    if isinstance(target, ast.Subscript) and self.holds_classes(info.expression_type(target.value)):
                        self.op(target.value, node, f"del {ast.unparse(target)}")
            elif isinstance(node, ast.For) and id(_iterated(node.iter)) in info.expressions:
                iterated = _iterated(node.iter)
                if self.holds_classes(info.expression_type(iterated)):
                    path = self.path(iterated)
                    assert path is not None
                    self.record.loops.append((path, node))

    def op(self, place: ast.expr, node: ast.AST, text: str) -> None:
        path = self.path(place)
        if path is not None:
            self.record.ops.append(_Op(path, node, text))

    def collect_store(self, target: ast.expr, statement: ast.AST) -> None:
        info = self.info
        text = ast.unparse(statement).splitlines()[0]
        if isinstance(target, ast.Name):
            local = info.local_types.get(target.id)
            if local is not None and (self.local_kind(target.id) == OWNED or self.holds_classes(local)):
                self.record.ops.append(_Op((target.id,), statement, text))
            return
        if not isinstance(target, (ast.Attribute, ast.Subscript)):
            return
        if isinstance(target, ast.Subscript):
            base = info.expression_type(target.value)
            if self.holds_classes(base):
                self.op(target, statement, text)
            return
        owner = info.expression_type(target.value)
        if not self.is_class(owner):
            return
        field_type = dict(self.program.struct_fields(owner.name))[target.attr]
        if self.is_class(field_type) or self.holds_classes(field_type):
            self.op(target, statement, text)

    def collect_call(self, node: ast.Call) -> None:
        info = self.info
        if isinstance(node.func, ast.Attribute) and id(node.func.value) in info.expressions:
            receiver = info.expression_type(node.func.value)
            if self.holds_classes(receiver):
                if node.func.attr == "get":
                    raise self.error(node, "get would copy a class instance; test 'key in d' and index it, or pop it")
                if node.func.attr == "copy":
                    container = ast.unparse(node.func.value)
                    raise self.error(
                        node,
                        f"{container}.copy() would share its instances in Python; copy.deepcopy({container}) copies "
                        "them",
                    )
                if node.func.attr in {"append", "pop", "clear"}:
                    self.op(node.func.value, node, ast.unparse(node))
            return
        if not isinstance(node.func, ast.Name) or node.func.id not in self.program.functions:
            return
        callee = self.program.functions[node.func.id]
        operands = [
            (index, None if self.creates(argument) else self.path(argument), argument)
            for index, (argument, (_, parameter)) in enumerate(zip(node.args, callee.parameters, strict=True))
            if self.is_class(parameter)
        ]
        if operands:
            self.record.calls.append(_Call(callee.symbol, operands, node))

    def check_element_scopes(self) -> None:
        """A loop's element is a borrow only while its loop runs; Python keeps the last one bound afterwards."""

        inside: set[int] = set()
        for _, loop in self.record.loops:
            names = loop.target.elts if isinstance(loop.target, ast.Tuple) else [loop.target]
            for name in names:
                if isinstance(name, ast.Name) and self.local_kind(name.id) == ELEMENT_LOCAL:
                    inside |= {
                        id(node)
                        for statement in loop.body
                        for node in ast.walk(statement)
                        if isinstance(node, ast.Name) and node.id == name.id
                    }
        for node in self.body:
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and self.local_kind(node.id) == ELEMENT_LOCAL
                and id(node) not in inside
            ):
                raise self.error(node, f"'{node.id}' is a loop's element, borrowed only inside the loop body")

    # The whole program.

    def finish(self) -> None:
        restructures = self.summaries()
        for record in self.functions.values():
            self.check_borrows(record, restructures)

    def summaries(self) -> dict[str, set[int]]:
        """The class parameters through which each function may replace, grow or shrink something: directly, or by
        passing a path under the parameter to a callee that may."""

        result: dict[str, set[int]] = {}
        for symbol, record in self.functions.items():
            roots = {name: index for index, (name, _) in enumerate(record.info.parameters)}
            result[symbol] = {roots[op.path[0]] for op in record.ops if op.path[0] in roots}
        changed = True
        while changed:
            changed = False
            for symbol, record in self.functions.items():
                roots = {name: index for index, (name, _) in enumerate(record.info.parameters)}
                for call in record.calls:
                    for index, path, _ in call.operands:
                        if path is None or path[0] not in roots or index not in result.get(call.callee, set()):
                            continue
                        if roots[path[0]] not in result[symbol]:
                            result[symbol].add(roots[path[0]])
                            changed = True
        return result

    def check_borrows(self, record: FunctionOwnership, restructures: dict[str, set[int]]) -> None:
        def changing(call: _Call) -> list[tuple[Path, ast.expr]]:
            reach = restructures.get(call.callee, set())
            return [(path, argument) for index, path, argument in call.operands if path is not None and index in reach]

        # Arguments of one call must not overlap where the callee may restructure through one of them.
        for call in record.calls:
            reach = restructures.get(call.callee, set())
            placed = [(index, path, argument) for index, path, argument in call.operands if path is not None]
            for position, (first, first_path, first_node) in enumerate(placed):
                for second, second_path, second_node in placed[position + 1 :]:
                    assert first_path is not None and second_path is not None
                    if (first in reach or second in reach) and (
                        _prefix(first_path, second_path) or _prefix(second_path, first_path)
                    ):
                        raise self.error(
                            call.node,
                            f"'{ast.unparse(first_node)}' and '{ast.unparse(second_node)}' may be one instance or "
                            f"contain each other, and '{_function_name(call.callee)}' may replace, grow or shrink what "
                            "one holds",
                        )
            # An element argument is reached before later arguments run; they must not invalidate it.
            for _, path, argument in call.operands:
                if path is None or ELEMENT not in path:
                    continue
                later = [node for other in call.node.args if other is not argument for node in ast.walk(other)]
                self.check_region(record, path, later, changing, f"'{ast.unparse(argument)}'")
        for container, loop in record.loops:
            region = [node for statement in loop.body for node in ast.walk(statement)]
            self.check_region(record, (*container, ELEMENT), region, changing, f"the loop over {_spell(container)}")
        for name, (path, _) in record.borrowed.items():
            self.check_region(record, path, self.body_of(record), changing, f"'{name}', which names {_spell(path)},")

    @staticmethod
    def body_of(record: FunctionOwnership) -> list[ast.AST]:
        return [node for statement in record.info.source.body for node in ast.walk(statement)]

    def check_region(
        self,
        record: FunctionOwnership,
        borrow: Path,
        region: list[ast.AST],
        changing: Callable[[_Call], list[tuple[Path, ast.expr]]],
        what: str,
    ) -> None:
        nodes = {id(node) for node in region}
        for op in record.ops:
            if id(op.node) in nodes and _prefix(op.path, borrow):
                raise self.error(
                    op.node,
                    f"{what} borrows {_spell(borrow)}, and '{op.text}' replaces, grows or shrinks what holds it",
                )
        for call in record.calls:
            if id(call.node) not in nodes:
                continue
            for path, argument in changing(call):
                if _prefix(path, borrow) and path != borrow:
                    raise self.error(
                        call.node,
                        f"{what} borrows {_spell(borrow)}, and '{_function_name(call.callee)}' may replace, grow or "
                        "shrink what "
                        f"'{ast.unparse(argument)}' holds",
                    )


_VOID = ConcreteType("void", "void")
_CONTAINERS = frozenset({"list", "dict"})


def _iterated(node: ast.expr) -> ast.expr:
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"keys", "values", "items"}
    ):
        return node.func.value
    return node


class _MoveFlow:
    """Rejects a use of an owned local after it was moved, on any path: the state maps each local that may have been
    moved to the statement moving it."""

    def __init__(self, checker: OwnershipChecker):
        self.checker = checker
        self.breaks: list[list[dict[str, ast.AST]]] = []
        self.continues: list[list[dict[str, ast.AST]]] = []

    def run(self, body: list[ast.stmt]) -> None:
        self.block(body, {})

    def block(self, body: list[ast.stmt], state: dict[str, ast.AST] | None) -> dict[str, ast.AST] | None:
        for statement in body:
            if state is None:
                return None
            state = self.statement(statement, state)
        return state

    @staticmethod
    def join(*states: dict[str, ast.AST] | None) -> dict[str, ast.AST] | None:
        live = [state for state in states if state is not None]
        if not live:
            return None
        joined: dict[str, ast.AST] = {}
        for state in live:
            for name, node in state.items():
                joined.setdefault(name, node)
        return joined

    def uses(self, nodes: list[ast.AST], state: dict[str, ast.AST]) -> None:
        checker = self.checker
        for node in nodes:
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load) and child.id in state:
                    moved = state[child.id]
                    raise checker.error(
                        child,
                        f"'{child.id}' was moved at line {getattr(moved, 'lineno', '?')}; Python would still share "
                        "that instance, so bind a new one to the name before using it again",
                    )

    def moves(self, statement: ast.stmt) -> list[str]:
        checker = self.checker
        values: list[ast.expr] = []
        if isinstance(statement, (ast.Assign, ast.AnnAssign)) and statement.value is not None:
            values.append(statement.value)
        elif isinstance(statement, ast.Return) and statement.value is not None:
            values.append(statement.value)
        for node in ast.walk(statement):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "append":
                values.extend(node.args)
            elif isinstance(node, ast.Call):
                values.extend(argument for argument in node.args if checker.taken(node, argument))
                # Only a constructor takes keywords, and it moves an instance into its field.
                values.extend(keyword.value for keyword in node.keywords)
        return [value.id for value in values if checker.owned_name(value)]  # type: ignore[attr-defined]

    def statement(self, statement: ast.stmt, state: dict[str, ast.AST]) -> dict[str, ast.AST] | None:
        if isinstance(statement, ast.If):
            self.uses([statement.test], state)
            return self.join(self.block(statement.body, dict(state)), self.block(statement.orelse, dict(state)))
        if isinstance(statement, (ast.While, ast.For)):
            self.uses([statement.test if isinstance(statement, ast.While) else statement.iter], state)
            entry = dict(state)
            while True:
                self.breaks.append([])
                self.continues.append([])
                end = self.block(statement.body, dict(entry))
                breaks, continues = self.breaks.pop(), self.continues.pop()
                after = self.join(entry, end, *continues)
                assert after is not None
                if after == entry:
                    break
                entry = after
            finished = self.block(statement.orelse, dict(entry))
            return self.join(finished, *breaks)
        if isinstance(statement, ast.With):
            state = dict(state)
            for item in statement.items:
                self.uses([item.context_expr], state)
                for name in self.moves(ast.Expr(value=item.context_expr)):
                    state[name] = statement
                if isinstance(item.optional_vars, ast.Name):
                    state.pop(item.optional_vars.id, None)
            return self.block(statement.body, state)
        if isinstance(statement, ast.Break):
            self.breaks[-1].append(dict(state))
            return None
        if isinstance(statement, ast.Continue):
            self.continues[-1].append(dict(state))
            return None
        self.uses([statement], state)
        moved = self.moves(statement)
        if isinstance(statement, ast.Return):
            return None
        state = dict(state)
        for target in statement.targets if isinstance(statement, ast.Assign) else []:
            if isinstance(target, ast.Name):
                state.pop(target.id, None)
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            state.pop(statement.target.id, None)
        for name in moved:
            state[name] = statement
        return state
