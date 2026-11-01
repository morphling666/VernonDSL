"""Host class values: instances passed and returned, owned and borrowed locals, moves, lists and dicts of classes and
recursive types run alike in Python mode and AOT, release every instance, and reject Python's aliasing."""

from __future__ import annotations

import ctypes
import gc
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest
from host_aot import LinkedHost, toolchain
from test_host_strs import _CountingAllocator
from vernon_dsl.diagnostics import CompileError
from vernon_dsl.hostlang import ExternRegistry, HostModule, load_host_module

CLASS_VALUES = Path(__file__).parent / "host_programs" / "class_values.py"

# fill(4) makes items of weights 1..4 with tags [2w], named[7] of weight 70 and named[8] a new Item.
# bump adds 1 to each item, then 5 to items[0] and 1 to named[7]; swap_best moves the weight-5 item into best;
# take(7) moves named[7] into items; drop removes named[8] and that item again; grow builds a tree of depth 4.
_SCRIPT: list[tuple[str, tuple[int, ...], int]] = [
    ("fill", (4,), 6),
    ("total", (), 512),
    ("bump", (), 83),
    ("swap_best", (), 8),
    ("take", (7,), 71),
    ("drop", (), 3),
    ("grow", (), 104),
    ("alias", (), 105),
    ("local_items", (3,), 32),
    ("reset", (), 1),
]


def _native_available() -> bool:
    try:
        from vernon_dsl import _native

        return hasattr(_native, "_host_runtime_symbols")
    except ImportError:
        return False


def test_python_mode_runs_class_values_and_releases_every_instance() -> None:
    module = HostModule(CLASS_VALUES, ExternRegistry(), eager=False)
    counting = _CountingAllocator()
    module.runtime.allocator.value = ctypes.addressof(counting.record)
    bag = module.namespace.Bag()
    outcomes = [getattr(bag, name)(*arguments) for name, arguments, _ in _SCRIPT]
    bag.grow()
    assert [child.value for child in bag.tree.children] == [1, 2, 1, 2]
    live_before_release = len(counting.live)
    del bag
    gc.collect()
    module.runtime.allocator.value = None

    assert outcomes == [expected for _, _, expected in _SCRIPT]
    assert live_before_release > 4
    assert counting.live == {}, "moves leave nothing behind, and fini releases every element, value and child"
    assert counting.mismatched == []


def test_json_rebuilds_lists_and_dicts_of_classes_from_their_defaults() -> None:
    module = HostModule(CLASS_VALUES, ExternRegistry(), eager=False)
    counting = _CountingAllocator()
    module.runtime.allocator.value = ctypes.addressof(counting.record)
    bag = module.namespace.Bag()
    bag.fill(2)
    bag.grow()
    module.from_json(
        bag,
        {
            "items": [{"weight": 5}, {"tags": [1, 2]}],
            "named": {"3": {"label": "three"}},
            "tree": {"children": [{"value": 9, "children": [{"value": 8}]}]},
        },
    )
    written = module.to_json(bag)
    del bag
    gc.collect()
    module.runtime.allocator.value = None

    assert written["items"] == [
        {"weight": 5, "tags": [], "label": "item"},
        {"weight": 1, "tags": [1, 2], "label": "item"},
    ], "an element the JSON names only some fields of starts from the class's defaults"
    assert written["named"] == {"3": {"weight": 1, "tags": [], "label": "three"}}
    assert written["tree"] == {"value": 0, "children": [{"value": 9, "children": [{"value": 8, "children": []}]}]}
    assert counting.live == {}, "the replaced elements, values and children are released"
    assert counting.mismatched == []


def test_python_callers_move_new_instances_into_places() -> None:
    namespace = HostModule(CLASS_VALUES, ExternRegistry(), eager=False).namespace
    bag = namespace.Bag()
    item = namespace.make_item(9)
    bag.best = item
    assert bag.best.weight == 9 and list(bag.best.tags) == [18]
    assert item.weight == 0 and len(item.tags) == 0, "the moved-from instance is left empty"
    bag.items.append(namespace.Item())
    with pytest.raises(TypeError, match="viewing another place"):
        bag.best = bag.items[0]
    assert namespace.weigh(bag.best, 1) == 12
    popped = bag.items.pop()
    assert popped.weight == 1 and len(bag.items) == 0


@pytest.mark.skipif(not _native_available(), reason="vernon_dsl._native was built without Host AOT")
@pytest.mark.skipif(toolchain() is None, reason="no C toolchain links Host objects")
def test_class_values_run_alike_and_release_every_instance(tmp_path: Path) -> None:
    from vernon_dsl.hostlang.aot import load_programs

    library = LinkedHost(load_programs(CLASS_VALUES), tmp_path, ExternRegistry())
    table = library.table
    counting = _CountingAllocator()
    table.allocator[0] = ctypes.addressof(counting.record)
    descriptor = next(
        table.types[index].contents for index in range(table.type_count) if table.types[index].contents.name == b"Bag"
    )
    instance = ctypes.create_string_buffer(descriptor.size)
    address = ctypes.addressof(instance)
    library.call("Bag__init", address)
    outcomes = [library.call(f"Bag_{name}", address, *arguments) for name, arguments, _ in _SCRIPT]
    live_before_fini = len(counting.live)
    ctypes.CFUNCTYPE(None, ctypes.c_void_p)(descriptor.fini)(address)
    table.allocator[0] = None

    assert outcomes == [expected for _, _, expected in _SCRIPT]
    assert live_before_fini > 4, "the items, their tags and the tree own memory"
    assert counting.live == {}, "fini releases every element, value and child"
    assert counting.mismatched == []


_TREE_CONSUMER = """
#include "class_values.h"

static int32_t sum(const CvNode &node) {
    int32_t total = node.value;
    for (const CvNode &child : node.children) {
        total += sum(child);
    }
    return total;
}

int32_t touch(CvBag *bag) {
    int32_t total = sum(bag->tree);
    for (CvItem &item : bag->items) {
        item.weight += 1;
        total += item.weight + static_cast<int32_t>(item.tags.size());
    }
    if (CvItem *named = bag->named.find(7)) {
        total += named->weight;
    }
    return total + bag->best.weight;
}
"""


def test_header_declares_recursive_classes_for_c_and_cpp(tmp_path: Path) -> None:
    from vernon_dsl.hostlang.aot import compile_host
    from vernon_dsl.hostlang.reflect import include_dir

    header = compile_host(CLASS_VALUES, prefix="cv_").header
    assert (
        "typedef struct CvNode CvNode;" in header
        and "struct CvNode { int32_t value; VH_LIST(CvNode) children; };" in header
    )
    assert "VH_LIST(CvItem) items;" in header and "VH_DICT(int32_t, CvItem) named;" in header
    found = toolchain()
    if found is None:
        pytest.skip("no C toolchain")
    kind, compiler, environment = found
    (tmp_path / "class_values.h").write_text(header, encoding="utf-8")
    (tmp_path / "records.c").write_text(
        '#include "class_values.h"\nuint64_t count(const CvNode *node) { return node->children.size; }\n',
        encoding="utf-8",
    )
    (tmp_path / "typed.cpp").write_text(_TREE_CONSUMER, encoding="utf-8")
    runtime_include = Path(__file__).resolve().parents[2] / "source" / "include"
    includes = [str(tmp_path), str(include_dir()), str(runtime_include)]
    for source, standard in (("records.c", "c11"), ("typed.cpp", "c++17")):
        if kind == "msvc":
            command = [compiler, "/nologo", "/W4", "/WX", f"/std:{standard}", "/EHsc", "/c"]
            command += [f"/I{path}" for path in includes] + [str(tmp_path / source), f"/Fo{tmp_path}{os.sep}"]
        else:
            driver = compiler if standard == "c11" else (shutil.which("c++") or compiler)
            command = [driver, f"-std={standard}", "-Wall", "-Werror", "-c"]
            command += [f"-I{path}" for path in includes] + [str(tmp_path / source), "-o", str(tmp_path / "out.o")]
        result = subprocess.run(command, capture_output=True, text=True, errors="replace", env=environment, check=False)
        assert result.returncode == 0, f"{source}:\n{result.stdout}{result.stderr}"


_PRELUDE = """
from __future__ import annotations

import vernon_dsl as vd
from vernon_dsl import i32


@vd.host
class Item:
    weight: i32 = 1
    tags: list[i32]


@vd.host
class Bag:
    items: list[Item]
    best: Item

    @vd.host(effects="pure")
    def reset(self) -> None:
        self.items.clear()
"""

_GRAB = """

@vd.host(effects="pure")
def grab(bag: Bag, item: Item) -> i32:
    bag.items.clear()
    return item.weight
"""


@pytest.mark.parametrize(
    ("method", "message"),
    [
        (
            """
            def f(self) -> i32:
                item = Item()
                self.best = item
                return item.weight
            """,
            "'item' was moved at line",
        ),
        (
            """
            def f(self) -> None:
                for item in self.items:
                    self.items.append(Item())
            """,
            "the loop over self.items borrows self.items[...], and 'self.items.append(Item())' replaces",
        ),
        (
            """
            def f(self) -> None:
                for item in self.items:
                    self.reset()
            """,
            "'Bag.reset' may replace, grow or shrink what 'self' holds",
        ),
        (
            """
            def f(self) -> i32:
                return grab(self, self.items[0])
            """,
            "'self' and 'self.items[0]' may be one instance or contain each other, and 'grab' may replace",
        ),
        (
            """
            def f(self) -> i32:
                best = self.best
                self.best = Item()
                return best.weight
            """,
            "'best', which names self.best, borrows self.best, and 'self.best = Item()' replaces",
        ),
        (
            """
            def f(self) -> i32:
                return Item().weight
            """,
            "creates an instance; bind it to a local before reading it",
        ),
        (
            """
            def f(self) -> None:
                self.best = self.items[0]
            """,
            "which Python would share rather than copy once stored in self.best",
        ),
        (
            """
            def f(self) -> None:
                best = self.items[0]
            """,
            "would name an element of a list or dict, which moves when the container changes",
        ),
        (
            """
            def f(self, item: Item) -> Item:
                return item
            """,
            "which Python would share rather than copy once returned",
        ),
    ],
    ids=[
        "use-after-move",
        "loop-invalidated",
        "loop-call-invalidates",
        "overlapping-arguments",
        "borrow-invalidated",
        "temporary-read",
        "place-to-place",
        "borrowed-element",
        "returned-parameter",
    ],
)
def test_ownership_diagnostics(tmp_path: Path, method: str, message: str) -> None:
    source = tmp_path / "broken.py"
    source.write_text(_PRELUDE + textwrap.indent(textwrap.dedent(method), "    ") + _GRAB, encoding="utf-8")
    with pytest.raises(CompileError) as error:
        load_host_module(source)
    assert message in str(error.value)
