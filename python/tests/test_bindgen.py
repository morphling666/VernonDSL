"""Level 1 bindings (vernon_dsl.bindgen) over the tally sample library in python/tests/bindgen_sample.

The sample's nanobind module, tally_api, is built by CMake with BUILD_TESTING; tests that call it skip without it.
"""

from __future__ import annotations

import gc
import importlib
import shutil
import subprocess
from pathlib import Path
from types import ModuleType

import pytest
from host_aot import LinkedHost, toolchain
from vernon_dsl.bindgen import BindgenError, check, generate
from vernon_dsl.bindgen.__main__ import main
from vernon_dsl.bindgen.generator import read_library
from vernon_dsl.frontend.model import ConcreteType
from vernon_dsl.hostlang import ExternRegistry, HostExecutionError, library_identity
from vernon_dsl.hostlang.aot import compile_programs, load_programs
from vernon_dsl.hostlang.reflect import VhType, layout_hash

SAMPLE = Path(__file__).parent / "bindgen_sample"
GENERATED = SAMPLE / "generated"
INCLUDE = Path(__file__).parents[1] / "vernon_dsl" / "hostlang" / "include"
# A packaged build links VernonRuntime statically; the consumer compiles in the Host container functions it needs.
RUNTIME_CONTAINERS = Path(__file__).parents[2] / "source" / "lib" / "runtime" / "host_containers.cpp"
RUNTIME_INCLUDE = Path(__file__).parents[2] / "source" / "include"


@pytest.fixture
def sample_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(SAMPLE))


@pytest.fixture
def tally_api(sample_path: None) -> ModuleType:
    try:
        return importlib.import_module("tally_api")
    except ImportError:
        pytest.skip("the tally_api sample module is not built (CMake BUILD_TESTING)")


@pytest.fixture
def tally_user(sample_path: None) -> ModuleType:
    return importlib.import_module("tally_user")


def test_committed_outputs_are_current() -> None:
    assert check(SAMPLE / "tally.py", GENERATED) == []


def test_check_reports_drift(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["generate", str(SAMPLE / "tally.py"), "--output-dir", str(tmp_path)]) == 0
    assert main(["check", str(SAMPLE / "tally.py"), "--output-dir", str(tmp_path)]) == 0
    header = tmp_path / "tally_api.h"
    header.write_text(header.read_text(encoding="utf-8") + "/* edited */\n", encoding="utf-8")
    (tmp_path / "tally_api.pyi").unlink()
    capsys.readouterr()
    assert main(["check", str(SAMPLE / "tally.py"), "--output-dir", str(tmp_path)]) == 1
    reported = capsys.readouterr().err
    assert "tally_api.h is out of date" in reported and "tally_api.pyi is out of date" in reported


def test_stub_goes_where_pyi_says(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stub = tmp_path / "package" / "tally.pyi"
    arguments = [str(SAMPLE / "tally.py"), "--output-dir", str(tmp_path / "generated"), "--pyi", str(stub)]
    assert main(["generate", *arguments]) == 0
    assert stub.read_text(encoding="utf-8") == generate(SAMPLE / "tally.py")["tally_api.pyi"]
    assert not (tmp_path / "generated" / "tally_api.pyi").exists()
    assert main(["check", *arguments]) == 0
    stub.write_text("", encoding="utf-8")
    capsys.readouterr()
    assert main(["check", *arguments]) == 1
    assert f"{stub} is out of date" in capsys.readouterr().err


def test_outputs_follow_the_conventions() -> None:
    outputs = generate(SAMPLE / "tally.py")
    identity = library_identity(SAMPLE / "tally.py", "tally", 1)
    header = outputs["tally_api.h"]
    assert f'#define TALLY_V1_SCHEMA "{identity}"' in header
    assert "TALLY_API int32_t tally_v1_counter_create(TallyMode mode, TallyCounter *out);" in header
    assert "TALLY_API int32_t tally_v1_counter_label(TallyCounter counter, VhStrArg label, int32_t *out);" in header
    assert "TALLY_API float tally_v1_scaled_total(const TallyStats *stats, float factor);" in header
    implementation = outputs["tally_api_impl.h"]
    assert "vh::Result<std::unique_ptr<tally::Counter>> counter_create(TallyMode mode);" in implementation
    assert "vh::Status counter_destroy(std::unique_ptr<tally::Counter> counter);" in implementation
    assert "vh::Status counter_add(tally::Counter &counter, int32_t amount);" in implementation
    assert "vh::Result<TallyStats> counter_stats(const tally::Counter &counter);" in implementation
    assert "vh::Result<int32_t> counter_label(tally::Counter &counter, std::string_view label);" in implementation
    assert "// Counts amount; a negative amount fails with status 10." in implementation
    assert "def counter_stats(counter: Counter) -> Stats: ..." in outputs["tally_api.pyi"]

    # Spans and owned results: VhSpan and VhList in C, C++17 views and owning values in the implementation.
    assert "TALLY_API int32_t tally_v1_counter_add_all(TallyCounter counter, VhSpan amounts);" in header
    assert (
        "TALLY_API int32_t tally_v1_counter_histogram(TallyCounter counter, VhList *out, const VhAllocator *allocator);"
        in header
    )
    assert "vh::Status counter_add_all(tally::Counter &counter, vh::Span<int32_t> amounts);" in implementation
    assert (
        "vh::Result<float> counter_add_weighted(tally::Counter &counter, "
        "vh::Span<vh::Entry<int32_t, tally::Sample>> samples);"
    ) in implementation
    assert "vh::Result<std::string> counter_name(const tally::Counter &counter);" in implementation
    assert (
        "vh::Result<std::vector<vh::Entry<int32_t, int32_t>>> counter_histogram(const tally::Counter &counter);"
        in implementation
    )
    assert "std::span" not in implementation + outputs["tally_api_thunks.cpp"]

    # Generic externs take the Struct's descriptor and its bytes.
    assert (
        "TALLY_API int32_t tally_v1_counter_store(TallyCounter counter, const VhType *type, const void *value);"
        in header
    )
    assert "vh::Status counter_load(const tally::Counter &counter, const VhType &type, void *out);" in implementation
    assert "vh::Result<bool> counter_has(const tally::Counter &counter, const VhType &type);" in implementation
    stubs = outputs["tally_api.pyi"]
    assert "def counter_store(counter: Counter, value: T) -> None: ..." in stubs
    assert "def counter_load(counter: Counter, type: type[T]) -> T: ..." in stubs
    assert "from vernon_dsl.hostlang import HostExecutionError as HostExecutionError" in stubs
    assert "    def __init__(self, amount: int, weight: float) -> None: ..." in stubs

    # The effect check: declared once per object it is asked about, called by every non-pure status thunk.
    assert "vh::Status allow(const VhCall &call);" in implementation
    assert "vh::Status allow(const VhCall &call, const tally::Counter &object);" in implementation
    assert "vh::Status allow(const VhCall &call, const tally::Cursor &object);" in implementation
    assert (
        'const VhCall call_counter_label = {sizeof(VhCall), 2, "tally_v1_counter_label", nullptr, '
        "call_counter_label_meta, 1};"
    ) in outputs["tally_api_thunks.cpp"]
    assert "call_counter_stats = {sizeof(VhCall), 1," in outputs["tally_api_thunks.cpp"]
    assert "call_scaled_total" not in outputs["tally_api_thunks.cpp"]
    assert "call_live_counters" not in outputs["tally_api_thunks.cpp"]

    # Structs with cpp=: Sample is cast after static_asserts, Range converts through declared functions.
    thunks = outputs["tally_api_thunks.cpp"]
    assert "static_assert(sizeof(tally::Sample) == sizeof(TallySample)" in thunks
    assert "tally::Range" not in thunks.split("namespace {")[0]
    assert "tally::Range convert(const TallyRange &value);" in implementation
    assert "TallyRange convert(const tally::Range &value);" in implementation
    assert "vh::Result<tally::Range> counter_range(const tally::Counter &counter);" in implementation
    assert '#include "tally/counter.h"' in implementation

    # Converted items convert one by one; a handle with convert= is a C++ value without a table.
    assert (
        "vh::Result<std::vector<tally::Range>> counter_clamp(const tally::Counter &counter, "
        "vh::Span<tally::Range> ranges);"
    ) in implementation
    assert "tally::Ticket convert(TallyTicket value);" in implementation
    assert "TallyTicket convert(const tally::Ticket &value);" in implementation
    assert "int32_t ticket_position(tally::Ticket ticket);" in implementation
    assert "HandleTable<tally::Ticket>" not in thunks

    # A dependent handle records its parent, and destroying the parent destroys it first.
    assert "out->bits = cursor_table().insert(std::move(returned.value()), counter.bits);" in thunks
    assert "vh::Status dependents = destroy_counter_dependents(counter.bits);" in thunks
    assert "release_cursor(child.first, std::move(child.second));" in thunks
    assert "vh::Status destroyed = tally::api::cursor_destroy(std::move(object));" in thunks
    assert "TALLY_API const VhType *const *tally_v1_types(uint32_t *count);" in header

    # The library's own code reaches the objects behind handles; an extension can host the nanobind module.
    assert "tally::Counter *find(TallyCounter handle);" in implementation
    assert "template <> vh::Status destroy_all<TallyCounter>();" in implementation
    nanobind = outputs["tally_api_nanobind.cpp"]
    assert "void tally_v1_nanobind(nb::module_ &m) {" in nanobind
    assert "#ifndef TALLY_NANOBIND_HOSTED\nNB_MODULE(tally_api, m) { tally_v1_nanobind(m); }\n#endif" in nanobind


def test_nanobind_module_calls_the_library(tally_api: ModuleType) -> None:
    assert tally_api.vh_schema == library_identity(SAMPLE / "tally.py", "tally", 1)
    counter = tally_api.counter_create(tally_api.Mode.SUM)
    tally_api.counter_add(counter, 5)
    tally_api.counter_add(counter, 2)
    stats = tally_api.counter_stats(counter)
    assert (stats.count, stats.total, list(stats.recent)) == (2, 7, [2.0, 5.0, 0.0])
    assert tally_api.scaled_total(stats, 2.0) == 14.0
    assert tally_api.counter_label(counter, "größe") == len("größe".encode())
    with pytest.raises(HostExecutionError, match="amount -1 is negative") as failed:
        tally_api.counter_add(counter, -1)
    assert failed.value.status == 10
    tally_api.counter_destroy(counter)
    with pytest.raises(HostExecutionError, match="counter is not a live Counter") as stale:
        tally_api.counter_stats(counter)
    assert stale.value.status == 2


def test_nanobind_module_passes_spans_and_returns_owned_results(tally_api: ModuleType) -> None:
    counter = tally_api.counter_create(tally_api.Mode.SUM)
    with pytest.raises(HostExecutionError, match="no label") as nameless:
        tally_api.counter_name(counter)
    assert nameless.value.status == 11
    tally_api.counter_add_all(counter, [3, 1, 3])
    with pytest.raises(HostExecutionError, match="amount -2 is negative"):
        tally_api.counter_add_all(counter, [5, -2])
    weighted = tally_api.counter_add_weighted(counter, {2: tally_api.Sample(4, 0.5), 1: tally_api.Sample(6, 1.0)})
    assert weighted == 2 * 0.5 + 1 * 1.0 * 2
    tally_api.counter_label(counter, "größe")
    assert tally_api.counter_name(counter) == "größe"
    assert tally_api.counter_history(counter) == [3, 1, 3, 4, 6]
    histogram = tally_api.counter_histogram(counter)
    assert histogram == {3: 2, 1: 1, 4: 1, 6: 1} and list(histogram) == [3, 1, 4, 6]
    samples = tally_api.counter_samples(counter)
    assert [(sample.amount, sample.weight) for sample in samples] == [(3, 1.5), (1, 0.5), (3, 1.5), (4, 2.0), (6, 3.0)]
    bounds = tally_api.counter_range(counter)
    assert (bounds.low, bounds.high) == (1, 6)
    assert tally_api.counter_in_range(counter, tally_api.Range(3, 4)) == 3

    # A generic function takes the bound Struct's class, whose descriptor the library describes, or a value of it.
    assert not tally_api.counter_has(counter, tally_api.Sample)
    tally_api.counter_store(counter, tally_api.Sample(3, 0.25))
    assert tally_api.counter_has(counter, tally_api.Sample)
    # A Struct of no fields is constructed without arguments.
    tally_api.counter_store(counter, tally_api.Marker())
    assert tally_api.counter_has(counter, tally_api.Marker)
    loaded = tally_api.counter_load(counter, tally_api.Sample)
    assert isinstance(loaded, tally_api.Sample) and (loaded.amount, loaded.weight) == (3, 0.25)
    with pytest.raises(TypeError, match="value is not a Struct of this module"):
        tally_api.counter_store(counter, 3)
    with pytest.raises(TypeError, match="type is not a Struct class of this module"):
        tally_api.counter_has(counter, int)
    tally_api.counter_destroy(counter)


def test_effect_check_refuses_before_the_implementation(tally_api: ModuleType) -> None:
    assert tally_api.HostExecutionError is HostExecutionError
    counter = tally_api.counter_create(tally_api.Mode.SUM)
    cursor = tally_api.cursor_create(counter)
    tally_api.counter_add(counter, 4)
    tally_api.counter_label(counter, "frozen")
    with pytest.raises(HostExecutionError, match="tally_v1_counter_add: the counter is frozen") as refused:
        tally_api.counter_add(counter, 1)
    assert refused.value.status == 16
    with pytest.raises(HostExecutionError, match="the counter is frozen"):
        tally_api.counter_store(counter, tally_api.Sample(1, 1.0))
    with pytest.raises(HostExecutionError, match="tally_v1_cursor_next: the counter is frozen"):
        tally_api.cursor_next(cursor)
    # Reads pass, and so do the exemptions: a call with `unfreezes` meta, and storing a Marker, seen through the
    # call's type.
    assert tally_api.counter_history(counter) == [4]
    tally_api.counter_store(counter, tally_api.Marker())
    assert tally_api.counter_has(counter, tally_api.Marker) and not tally_api.counter_has(counter, tally_api.Sample)
    # A refused destroy leaves the counter and its cursor alive.
    with pytest.raises(HostExecutionError, match="tally_v1_counter_destroy: the counter is frozen"):
        tally_api.counter_destroy(counter)
    assert tally_api.counter_exists(counter.bits) and tally_api.counter_history(counter) == [4]
    tally_api.counter_label(counter, "thawed")
    assert tally_api.cursor_next(cursor) == 4
    tally_api.counter_destroy(counter)
    assert not tally_api.counter_exists(counter.bits)


def test_library_descriptors_match_host_layouts(tally_api: ModuleType) -> None:
    program = read_library(SAMPLE / "tally.py").program
    for name in ("Stats", "Sample", "Range"):
        descriptor = VhType.from_address(getattr(tally_api, name).vh_type)
        assert descriptor.name.decode() == name
        assert descriptor.layout_hash == layout_hash(ConcreteType("struct", name), [program])
    stats = VhType.from_address(tally_api.Stats.vh_type)
    assert [stats.fields[index].name.decode() for index in range(stats.field_count)] == ["count", "total", "recent"]
    assert stats.fields[2].offset == 16 and stats.fields[2].type.contents.rank == 1


def test_value_handles_and_converted_items(tally_api: ModuleType) -> None:
    counter = tally_api.counter_create(tally_api.Mode.SUM)
    tally_api.counter_add_all(counter, [4, 9, 2])
    clamped = tally_api.counter_clamp(counter, [tally_api.Range(0, 5), tally_api.Range(3, 20)])
    assert [(bounds.low, bounds.high) for bounds in clamped] == [(2, 5), (3, 9)]
    ticket = tally_api.counter_ticket(counter)
    assert isinstance(ticket, tally_api.Ticket) and tally_api.ticket_position(ticket) == 2
    tickets = tally_api.counter_tickets(counter)
    assert tickets[2] == ticket and len({*tickets}) == 3
    assert tally_api.counter_redeem(counter, tickets[:2]) == 13
    other = tally_api.counter_create(tally_api.Mode.SUM)
    with pytest.raises(HostExecutionError, match="ticket 2 is not this counter's") as foreign:
        tally_api.counter_redeem(other, [ticket])
    assert foreign.value.status == 14


def test_owned_handles_and_dependents(tally_api: ModuleType) -> None:
    # Objects earlier tests created go first, so the counts below are this test's.
    gc.collect()
    counters, cursors = tally_api.live_counters(), tally_api.live_cursors()
    counter = tally_api.counter_create(tally_api.Mode.SUM)
    assert isinstance(counter, tally_api.Counter.Owned) and isinstance(counter, tally_api.Counter)
    tally_api.counter_add_all(counter, [7, 8])
    cursor = tally_api.cursor_create(counter)
    assert tally_api.cursor_next(cursor) == 7
    # A plain handle of the same bits names the object without owning it.
    borrowed = tally_api.Counter(counter.bits)
    assert borrowed == counter and hash(borrowed) == hash(counter)
    assert counter != None and counter != 0  # noqa: E711
    del borrowed
    gc.collect()
    assert tally_api.live_counters() == counters + 1

    # The cursor keeps its counter's Python object alive; both go with the last reference.
    del counter
    gc.collect()
    assert tally_api.live_counters() == counters + 1
    assert tally_api.cursor_next(cursor) == 8
    del cursor
    gc.collect()
    assert (tally_api.live_counters(), tally_api.live_cursors()) == (counters, cursors)

    # Destroying a counter destroys its cursors first; their objects then name nothing.
    counter = tally_api.counter_create(tally_api.Mode.SUM)
    first, second = tally_api.cursor_create(counter), tally_api.cursor_create(counter)
    assert tally_api.live_cursors() == cursors + 2
    tally_api.counter_destroy(counter)
    assert not counter and (tally_api.live_counters(), tally_api.live_cursors()) == (counters, cursors)
    with pytest.raises(HostExecutionError, match="cursor is not a live Cursor"):
        tally_api.cursor_next(first)
    del first, second
    gc.collect()
    assert tally_api.live_cursors() == cursors


def test_library_code_finds_and_destroys_handles(tally_api: ModuleType) -> None:
    counter = tally_api.counter_create(tally_api.Mode.SUM)
    cursor = tally_api.cursor_create(counter)
    assert tally_api.counter_exists(counter.bits) and not tally_api.counter_exists(0)

    # Destroying every counter goes through counter_destroy's cascade: the cursors go first.
    tally_api.counter_destroy_all()
    assert (tally_api.live_counters(), tally_api.live_cursors()) == (0, 0)
    assert not tally_api.counter_exists(counter.bits)
    with pytest.raises(HostExecutionError, match="cursor is not a live Cursor"):
        tally_api.cursor_next(cursor)
    del cursor, counter
    gc.collect()


def _python_mode(tally_user: ModuleType, registry: ExternRegistry, name: str, *args: object) -> object:
    try:
        return getattr(tally_user, name).call(*args, registry=registry)
    except HostExecutionError as error:
        return ("status", error.status)


def _linked(linked: LinkedHost, name: str, *args: object) -> object:
    try:
        return linked.call(name, *args)
    except HostExecutionError as error:
        return ("status", error.status)


def test_python_mode_and_linked_aot_agree(tally_api: ModuleType, tally_user: ModuleType, tmp_path: Path) -> None:
    if toolchain() is None:
        pytest.skip("no C toolchain links Host objects on this machine")
    registry = ExternRegistry().bind_module(tally_api)
    linked = LinkedHost(load_programs(SAMPLE / "tally_user.py"), tmp_path, registry, prefix="tally_user_")
    for name, args, expected in [
        ("run", (4,), 10046),
        ("peak", (5,), 18.0),
        ("negative", (), ("status", 10)),
        ("stale", (), ("status", 2)),
        ("spans", (4,), 878.0),
        ("owned", (7,), 507030706),
        ("nameless", (), ("status", 11)),
        ("ranged", (), 10903),
        ("stored", (7,), 1117.5),
    ]:
        assert _python_mode(tally_user, registry, name, *args) == expected, name
        assert _linked(linked, name, *args) == expected, name

    with pytest.raises(HostExecutionError, match="'tally_v1_counter_add' failed: counter_add: amount -4 is negative"):
        tally_user.negative.call(registry=registry)
    with pytest.raises(HostExecutionError):
        linked.call("negative")
    assert tally_api.vh_last_error() == "counter_add: amount -4 is negative"

    # Both modes reach the one library: a run destroys its Counter, and a failed one leaks it.
    alive = linked.call("alive")
    assert _python_mode(tally_user, registry, "alive") == alive
    linked.call("run", 3)
    tally_user.run.call(3, registry=registry)
    assert linked.call("alive") == alive
    assert _linked(linked, "negative") == ("status", 10)
    assert tally_api.live_counters() == alive + 1


def _build_consumer(directory: Path) -> Path:
    found = toolchain()
    if found is None:
        pytest.skip("no C toolchain links Host objects on this machine")
    kind, compiler, environment = found
    artifacts = compile_programs(load_programs(SAMPLE / "tally_user.py"), prefix="tally_user_")
    paths = artifacts.write(directory, "tally_user")
    sources = [
        SAMPLE / "consumer.c",
        GENERATED / "tally_api_thunks.cpp",
        SAMPLE / "cpp" / "tally_api.cpp",
        SAMPLE / "cpp" / "counter.cpp",
        RUNTIME_CONTAINERS,
    ]
    includes = [directory, GENERATED, SAMPLE / "cpp", INCLUDE, RUNTIME_INCLUDE]
    compiler_cpp = compiler if kind == "msvc" else shutil.which("c++")
    if compiler_cpp is None:
        pytest.skip("no C++ compiler")

    def run(command: list[str]) -> None:
        built = subprocess.run(command, capture_output=True, text=True, errors="replace", env=environment, check=False)
        assert built.returncode == 0, f"{built.stdout}{built.stderr}"

    # C and C++ sources compile separately: MSVC takes one language standard per invocation.
    objects = []
    for source in sources:
        is_c = source.suffix == ".c"
        if kind == "msvc":
            output = directory / f"{source.stem}.obj"
            standard = "/std:c11" if is_c else "/std:c++17"
            includes_flags = [f"/I{path}" for path in includes]
            run(
                [
                    compiler,
                    "/nologo",
                    "/c",
                    "/EHsc",
                    "/DVERNON_RUNTIME_STATIC",
                    standard,
                    *includes_flags,
                    str(source),
                    f"/Fo{output}",
                ]
            )
        else:
            output = directory / f"{source.stem}.o"
            standard = "-std=c11" if is_c else "-std=c++17"
            tool = compiler if is_c else compiler_cpp
            flags = [standard, "-DVERNON_RUNTIME_STATIC", *(f"-I{path}" for path in includes)]
            run([tool, *flags, "-c", str(source), "-o", str(output)])
        objects.append(str(output))
    if kind == "msvc":
        program = directory / "consumer.exe"
        run([compiler, "/nologo", *objects, str(paths["object"]), f"/Fe{program}"])
    else:
        program = directory / "consumer"
        run([compiler_cpp, *objects, str(paths["object"]), "-lm", "-o", str(program)])
    return program


def test_linked_c_consumer(tmp_path: Path) -> None:
    program = _build_consumer(tmp_path)
    ran = subprocess.run([str(program)], capture_output=True, text=True, check=True)
    identity = library_identity(SAMPLE / "tally.py", "tally", 1)
    assert ran.stdout.splitlines() == [
        "run 0 10046",
        "peak 0 18.0",
        "stale 2 tally_v1_counter_add: counter is not a live Counter",
        "spans 0 878.0",
        "owned 0 507030706",
        "nameless 11 counter_name: the counter has no label",
        "ranged 0 10903",
        "stored 0 1117.5",
        "alive 0 1",
        f"schema {identity}",
    ]


_HEADER = """
import vernon_dsl as vd
from vernon_dsl import i32

vd.library("bad", version=0{meta})


@vd.handle(meta=vd.meta(cpp="bad::Thing"))
class Thing:
    pass


@vd.handle
class Id:
    pass


@vd.struct(shared=True, meta=vd.meta(cpp="bad::Pair", convert="convert"))
class Pair:
    first: i32
    second: i32
"""


@pytest.mark.parametrize(
    ("declaration", "message"),
    [
        ('@vd.extern(effects="read")\ndef use(thing: Thing) -> i32: ...', "so it is abi='status'"),
        ('@vd.extern(effects="io")\ndef make() -> Thing: ...', "creates a Thing handle, so it is abi='status'"),
        (
            '@vd.extern(effects="io", abi="status", meta=vd.meta(destroys="id"))\ndef drop(id: Id) -> None: ...',
            "names no parameter holding a handle with cpp=",
        ),
        (
            '@vd.extern(effects="read", abi="status")\ndef count(things: list[Thing]) -> i32: ...',
            "holds Thing handles, each of which would need a table lookup",
        ),
        (
            '@vd.handle(meta=vd.meta(convert="convert"))\nclass Loose:\n    pass',
            "handle Loose: convert names an unqualified C\\+\\+ function, and needs cpp=",
        ),
        (
            '@vd.handle(meta=vd.meta(cpp="bad::Part", depends_on="Id"))\nclass Part:\n    pass',
            "handle Part: depends_on names a handle, and both need cpp= without convert=",
        ),
        (
            '@vd.handle(meta=vd.meta(cpp="bad::Part", depends_on="Part"))\nclass Part:\n    pass',
            "handle Part: depends_on forms a cycle through Part",
        ),
        (
            '@vd.handle(meta=vd.meta(cpp="bad::Part", depends_on="Thing"))\nclass Part:\n    pass\n\n\n'
            '@vd.extern(effects="io", abi="status")\ndef part() -> Part: ...',
            "extern part creates a Part, which depends on Thing, so it takes one Thing",
        ),
        (
            '@vd.handle(meta=vd.meta(cpp="bad::Part", depends_on="Thing"))\nclass Part:\n    pass\n\n\n'
            '@vd.extern(effects="io", abi="status", meta=vd.meta(destroys="part"))\n'
            "def drop(part: Part, thing: Thing) -> None: ...",
            "extern drop: destroying Thing destroys its Part objects through it, so it takes only the Part",
        ),
        (
            '@vd.struct(shared=True, meta=vd.meta(convert="convert"))\nclass Loose:\n    value: i32\n\n\n'
            '@vd.extern(effects="pure")\ndef loose(value: Loose) -> i32: ...',
            "struct Loose: convert names an unqualified C\\+\\+ function, and needs cpp=",
        ),
    ],
)
def test_generator_rejects(tmp_path: Path, declaration: str, message: str) -> None:
    path = tmp_path / "bad_binding.py"
    path.write_text(_HEADER.format(meta=', meta=vd.meta(cpp_namespace="bad::api")') + "\n\n" + declaration + "\n")
    with pytest.raises(BindgenError, match=message):
        generate(path)


def test_generator_needs_a_cpp_namespace(tmp_path: Path) -> None:
    path = tmp_path / "bare_binding.py"
    path.write_text(_HEADER.format(meta=""))
    with pytest.raises(BindgenError, match="cpp_namespace"):
        generate(path)


def test_generator_needs_an_unqualified_effect_check(tmp_path: Path) -> None:
    path = tmp_path / "checked_binding.py"
    path.write_text(_HEADER.format(meta=', meta=vd.meta(cpp_namespace="bad::api", effect_check="bad::allow")'))
    with pytest.raises(BindgenError, match="effect_check names an unqualified C\\+\\+ function"):
        generate(path)
