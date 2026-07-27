from typing import Annotated

import vernon_dsl as vd


@vd.struct(shared=True)
class AggregateVertexPayload:
    object_id: vd.i32
    uv: vd.Vector[vd.f32, 2]
    weights: vd.Tensor[vd.f32, (2,)]


@vd.struct(shared=True)
class ComplexAggregateVertex:
    position: vd.Vector[vd.f32, 2]
    payload: AggregateVertexPayload
    basis: vd.Tensor[vd.f32, (2, 2)]


@vd.struct(shared=True)
class SmallAggregateVertex:
    position: vd.Vector[vd.f32, 2]
    scale: vd.f32


@vd.kernel(workgroup_size=(4, 1, 1))
def copy_complex_aggregate_tensor_view(
    output: vd.TensorView[ComplexAggregateVertex, 1, vd.write],
    source: vd.TensorView[ComplexAggregateVertex, 1, vd.read],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    output[gid[0]] = source[gid[0]]


@vd.func
def inspect_multidimensional_aggregate_tensor(
    values: vd.Tensor[ComplexAggregateVertex, (2, 3, 4)],
) -> vd.Tensor[vd.f32, (6,)]:
    vertex = values[1, 2, 3]
    return vd.Tensor(
        [
            vertex.position[0],
            vertex.position[1],
            vd.f32(vertex.payload.object_id),
            vertex.payload.uv[1],
            vertex.payload.weights[0],
            vertex.basis[1, 0],
        ]
    )


@vd.kernel
def inspect_multidimensional_aggregate_tensor_value(
    output: vd.TensorView[vd.f32, 1, vd.write],
    values: vd.Tensor[ComplexAggregateVertex, (2, 3, 4)],
) -> None:
    inspected = inspect_multidimensional_aggregate_tensor(values)
    output[0] = inspected[0]
    output[1] = inspected[1]
    output[2] = inspected[2]
    output[3] = inspected[3]
    output[4] = inspected[4]
    output[5] = inspected[5]


@vd.vertex
def aggregate_triangle_vertex(
    vertex: Annotated[ComplexAggregateVertex, vd.attribute()],
    aggregate: Annotated[vd.Tensor[ComplexAggregateVertex, (2, 3, 4)], vd.uniform()],
    vertex_index: Annotated[vd.u32, vd.builtin("vertex_index")],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    inspected = inspect_multidimensional_aggregate_tensor(aggregate)
    dynamic_vertex = aggregate[0, 0, vertex_index]
    object_id = vd.f32(vertex.payload.object_id)
    inspected_scale = (inspected[0] + inspected[1] + inspected[2] + inspected[3] + inspected[4] + inspected[5]) / 191.25
    dynamic_scale = (
        dynamic_vertex.position[0]
        + dynamic_vertex.position[1]
        + vd.f32(dynamic_vertex.payload.object_id)
        + dynamic_vertex.payload.uv[1]
        + dynamic_vertex.payload.weights[0]
        + dynamic_vertex.basis[1, 0]
    ) / (8.0 * vd.f32(vertex_index) + 7.25)
    attribute_scale = (
        (vertex.payload.uv[0] + vertex.payload.weights[0] + vertex.basis[0, 0]) * object_id / (2.0 * object_id)
    )
    scale = inspected_scale * dynamic_scale * attribute_scale
    return vd.Vector([vertex.position[0] * scale, vertex.position[1] * scale, 0.0, 1.0])


@vd.vertex
def small_multidimensional_aggregate_attribute_vertex(
    values: Annotated[vd.Tensor[SmallAggregateVertex, (2, 2, 2)], vd.attribute()],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    selected = values[1, 1, 1]
    return vd.Vector(
        [
            selected.position[0] * selected.scale / 2.0,
            selected.position[1] * selected.scale / 2.0,
            0.0,
            1.0,
        ]
    )


@vd.vertex
def oversized_aggregate_attribute_vertex(
    values: Annotated[vd.Tensor[ComplexAggregateVertex, (2, 2, 2)], vd.attribute()],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    selected = values[1, 1, 1]
    return vd.Vector([selected.position[0], selected.position[1], 0.0, 1.0])
