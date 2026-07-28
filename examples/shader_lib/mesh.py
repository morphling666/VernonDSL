from typing import Annotated

import vernon_dsl as vd


@vd.kernel(workgroup_size=(64, 1, 1))
def expand_indexed_mesh(
    positions: vd.TensorView[vd.f32, 2, vd.read],
    normals: vd.TensorView[vd.f32, 2, vd.read],
    colors: vd.TensorView[vd.f32, 2, vd.read],
    materials: vd.TensorView[vd.f32, 2, vd.read],
    indices: vd.TensorView[vd.u32, 1, vd.read],
    draw_positions: vd.TensorView[vd.f32, 2, vd.write],
    draw_normals: vd.TensorView[vd.f32, 2, vd.write],
    draw_colors: vd.TensorView[vd.f32, 2, vd.write],
    draw_materials: vd.TensorView[vd.f32, 2, vd.write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    draw_index = gid[0]
    source_index = indices[draw_index]
    draw_positions[draw_index, 0] = positions[source_index, 0]
    draw_positions[draw_index, 1] = positions[source_index, 1]
    draw_positions[draw_index, 2] = positions[source_index, 2]
    draw_normals[draw_index, 0] = normals[source_index, 0]
    draw_normals[draw_index, 1] = normals[source_index, 1]
    draw_normals[draw_index, 2] = normals[source_index, 2]
    draw_colors[draw_index, 0] = colors[source_index, 0]
    draw_colors[draw_index, 1] = colors[source_index, 1]
    draw_colors[draw_index, 2] = colors[source_index, 2]
    draw_materials[draw_index, 0] = materials[source_index, 0]
    draw_materials[draw_index, 1] = materials[source_index, 1]
    draw_materials[draw_index, 2] = materials[source_index, 2]
