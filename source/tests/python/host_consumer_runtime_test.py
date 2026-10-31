from __future__ import annotations

import ctypes
import difflib
import sys
from pathlib import Path

ASSET_ID = "demo/systems"


def main() -> None:
    fixture_dir, python_dir, asset_source = sys.argv[1:4]
    sys.path[:0] = [fixture_dir, python_dir, str(Path(asset_source).parent)]

    import systems_asset
    import vernon_host_consumer_fixture as fixture
    from vernon_dsl.hostlang import HostModule
    from vernon_dsl.hostlang.externs import ExternRegistry

    # The cooked object registered its table under the asset id from its static constructor.
    linked = fixture.find(ASSET_ID)
    assert linked == fixture.aot_table(), "the linked object should register itself under the asset id"
    fixture.reset()
    aot = fixture.drive(ASSET_ID)

    # Python mode registers its table under the same id, shadowing the linked one until it withdraws.
    registry = ExternRegistry().bind_symbols(fixture.extern_symbols())
    module = HostModule.from_asset(systems_asset.asset, registry)
    table = ctypes.addressof(module.table)
    assert module.asset_id == ASSET_ID
    assert fixture.register_table(ASSET_ID, table) == 0
    assert fixture.find(ASSET_ID) == table
    fixture.reset()
    python = fixture.drive(ASSET_ID)
    assert module.take_error() is None, "a Host status reaches C as the status, not as a kept exception"
    assert fixture.unregister_table(ASSET_ID, table) == 0
    assert fixture.find(ASSET_ID) == linked, "withdrawing the Python-mode table should restore the linked one"

    if python != aot:
        sys.exit("\n".join(difflib.unified_diff(aot, python, "aot", "python", lineterm="")))
    for expected in (
        "mover_init=0 speed_default=2",
        "update status=0 updates=3 height=3",
        "height_after status=0 value=9",
        'regen status=0 healed=5.5 stored={"current":4.5,"maximum":5}',
        "regen_missing status=7",
        "module=demo/systems types=8 functions=1",
        "trail_init=0 cells kind=16 element=f32 rank=2 shape=0,0 fini=1 marks=2",
        "spread status=0 rows=2 cell=12 marks=2,3 missing=1",
        "trail_total status=0 value=136",
        "view rank=2 shape=2,3 strides=12,4 bytes=24 shared=1",
        "trail_resized=1 count=2 c_count=2 total=5",
        'trail_moved={"cells":[[2.5,2.5]],"marks":[2,3]}',
        'trail_left={"cells":[],"marks":[]}',
        "poison_timer_type=Timer same_timer=1",
        "poison status=0 drained=0 current=4.5 ticks=0",
        "poison status=0 drained=0.25 current=4.25 ticks=1",
        "drains kind=13 element=f32 fini=1 name_kind=14",
        "hits kind=15 key=Entity element=f32 size=48",
        "typed name=poison drains=0.25 hit=0.25 missing=1",
        "typed_edit=1",
        "total_drained status=0 value=0.75",
        "strengthen status=0 level=1",
        "strengthen status=0 level=2",
        "typed levels=venom=9 strong poison=2 ",
        'poison_moved={"rate":1,"timer":{"period":0.5,"elapsed":0,"fired":1},"ticks":1,"name":"strong poison",'
        '"drains":[0.25,0.5],"hits":{"3":0.25},"levels":{"venom":9,"strong poison":2}}',
        'poison_left={"rate":1,"timer":{"period":0.5,"elapsed":0,"fired":1},"ticks":1,"name":"","drains":[],'
        '"hits":{},"levels":{}}',
        "allocator used=1 balanced=1",
        "scaled status=0 value=4.5",
        "    call=demo_read effect=1 type=Health",
    ):
        assert expected in aot, f"missing '{expected}' in:\n" + "\n".join(aot)
    print("\n".join(aot))


if __name__ == "__main__":
    main()
