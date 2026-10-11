from __future__ import annotations

import vernon_dsl as vd
from type_specialization_fixture_materials import Glossy, Matte
from type_specialization_fixture_template import MATERIAL, shade

asset = vd.program_asset(
    id="compute/type-specialization",
    program=shade,
    variants=({MATERIAL: Matte}, {MATERIAL: Glossy}),
)
