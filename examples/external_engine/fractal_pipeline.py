import vernon_dsl as vd

from examples.fractal import paint

asset = vd.program_asset(
    id="examples/external_engine/fractal",
    program=paint,
    variants=((),),
)
