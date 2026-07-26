from typing import Annotated

import vernon_dsl as vd


@vd.kernel(workgroup_size=(2, 1, 1))
def translate_vertices(
    position: vd.TensorView[vd.f32, 2, vd.read_write],
    offset: vd.f32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    component = gid[0]
    vertex = gid[1]
    if component == 0:
        position[vertex, component] = position[vertex, component] + offset


@vd.kernel(workgroup_size=(4, 1, 1))
def copy_static_tensor_value(
    output: vd.TensorView[vd.f32, 1, vd.write],
    singleton: vd.Tensor[vd.f32, (1,)],
    quad: vd.Tensor[vd.f32, (4,)],
    weights: vd.Tensor[vd.f32, (2, 2, 2)],
    large: vd.Tensor[vd.f32, (2, 3, 5)],
    matrix_left: vd.Tensor[vd.f32, (2, 2)],
    matrix_right: vd.Tensor[vd.f32, (2, 2)],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    doubled = weights + weights
    matrix_product = matrix_left * matrix_right
    invocation = gid[0]
    if invocation == 0:
        output[0] = singleton[0]
    if invocation == 1:
        output[1] = weights[0, 0, 0]
    if invocation == 2:
        output[2] = weights[1, 0, 1]
    if invocation == 3:
        output[3] = weights[1, 1, 1]
    if invocation == 4:
        output[4] = doubled[1, 1, 0]
    if invocation == 5:
        output[5] = large[1, 2, 4]
    if invocation == 6:
        output[6] = quad[3]
    if invocation == 7:
        output[7] = matrix_product[0, 0]
    if invocation == 8:
        output[8] = matrix_product[0, 1]
    if invocation == 9:
        output[9] = matrix_product[1, 0]
    if invocation == 10:
        output[10] = matrix_product[1, 1]


@vd.vertex
def triangle_vertex(
    position: Annotated[vd.Vector[vd.f32, 2], vd.attribute()],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    return vd.Vector([position, 0.0, 1.0])


@vd.vertex
def rank_three_tensor_attribute_vertex(
    value: Annotated[vd.Tensor[vd.f32, (2, 2, 3)], vd.attribute()],
    projection: Annotated[vd.Tensor[vd.f32, (1, 3, 2)], vd.attribute()],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    added = value + value
    subtracted = added - value
    multiplied = subtracted * 6.0
    divided = multiplied / 6.0
    projected = vd.matmul(divided, projection)
    return vd.Vector([projected[0, 0, 0], projected[0, 0, 1], 0.0, 1.0])


@vd.vertex
def instanced_tensor_transform_vertex(
    position: Annotated[vd.Vector[vd.f32, 2], vd.attribute()],
    transform: Annotated[vd.Tensor[vd.f32, (4, 4)], vd.attribute(divisor=1)],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    return vd.matmul(transform, vd.Vector([position, 0.0, 1.0]))


@vd.vertex
def i32_attribute_vertex(
    position: Annotated[vd.Vector[vd.f32, 2], vd.attribute()],
    value: Annotated[vd.Tensor[vd.i32, (2,)], vd.attribute()],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    return vd.Vector([position, 0.0, 1.0])


@vd.vertex
def u32_attribute_vertex(
    position: Annotated[vd.Vector[vd.f32, 2], vd.attribute()],
    value: Annotated[vd.Tensor[vd.u32, (2,)], vd.attribute()],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    return vd.Vector([position, 0.0, 1.0])


@vd.vertex
def f16_attribute_vertex(
    position: Annotated[vd.Vector[vd.f32, 2], vd.attribute()],
    value: Annotated[vd.Tensor[vd.f16, (2,)], vd.attribute()],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    return vd.Vector([position, 0.0, 1.0])


@vd.vertex
def f32_attribute_vertex(
    position: Annotated[vd.Vector[vd.f32, 2], vd.attribute()],
    value: Annotated[vd.Tensor[vd.f32, (2,)], vd.attribute()],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    return vd.Vector([position, 0.0, 1.0])


@vd.vertex
def f64_attribute_vertex(
    position: Annotated[vd.Vector[vd.f32, 2], vd.attribute()],
    value: Annotated[vd.Tensor[vd.f64, (2,)], vd.attribute()],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    return vd.Vector([position, 0.0, 1.0])


@vd.vertex
def non_square_attribute_vertex(
    position: Annotated[vd.Vector[vd.f32, 2], vd.attribute()],
    value: Annotated[vd.Tensor[vd.f32, (2, 3)], vd.attribute()],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    return vd.Vector([position.x + value[0, 0], position.y + value[1, 2], 0.0, 1.0])


@vd.vertex
def divisor_two_attribute_vertex(
    position: Annotated[vd.Vector[vd.f32, 2], vd.attribute()],
    offset: Annotated[vd.Vector[vd.f32, 2], vd.attribute(divisor=2)],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    return vd.Vector([position + offset, 0.0, 1.0])


@vd.vertex
def oversized_tensor_attribute_vertex(
    value: Annotated[vd.Tensor[vd.f32, (8, 8, 4)], vd.attribute()],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    return vd.Vector([value[0, 0, 0], value[0, 0, 1], 0.0, 1.0])


@vd.vertex
def translated_vertex(
    position: Annotated[vd.Vector[vd.f32, 2], vd.attribute()],
    offset: Annotated[vd.Vector[vd.f32, 2], vd.uniform()],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    return vd.Vector([position + offset, 0.0, 1.0])


@vd.vertex
def matrix_vertex(
    position: Annotated[vd.Vector[vd.f32, 2], vd.attribute()],
    transform: Annotated[vd.Matrix[vd.f32, 4, 4], vd.uniform()],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    return vd.matmul(transform, vd.Vector([position, 0.0, 1.0]))


@vd.vertex
def mat2_vertex(
    position: Annotated[vd.Vector[vd.f32, 2], vd.attribute()],
    transform: Annotated[vd.Matrix[vd.f32, 2, 2], vd.uniform()],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    return vd.Vector([vd.matmul(transform, position), 0.0, 1.0])


@vd.fragment
def solid_fragment() -> vd.Vector[vd.f32, 4]:
    return vd.Vector([1.0, 0.25, 0.0, 1.0])


@vd.fragment
def colored_fragment(
    color: Annotated[vd.Vector[vd.f32, 4], vd.uniform()],
) -> vd.Vector[vd.f32, 4]:
    return color


@vd.fragment
def static_tensor_fragment(
    weights: Annotated[vd.Tensor[vd.f32, (2, 2, 2)], vd.uniform(set=0, binding=3)],
) -> vd.Vector[vd.f32, 4]:
    return vd.Vector([weights[1, 0, 1], 0.0, 0.0, 1.0])


@vd.fragment
def matrix_elementwise_fragment(
    left: Annotated[vd.Tensor[vd.f32, (2, 2)], vd.uniform()],
    right: Annotated[vd.Tensor[vd.f32, (2, 2)], vd.uniform()],
) -> vd.Vector[vd.f32, 4]:
    product = left * right
    return vd.Vector([product[0, 0], product[0, 1], product[1, 0], product[1, 1]])


@vd.vertex
def numpy_tensor_vertex(
    position: Annotated[vd.Vector[vd.f32, 2], vd.attribute()],
    offset_left: Annotated[vd.Tensor[vd.f32, (8, 8, 4)], vd.uniform()],
    offset_right: Annotated[vd.Tensor[vd.f32, (1, 8, 1)], vd.uniform()],
    transform: Annotated[vd.Tensor[vd.f32, (8, 8, 4)], vd.uniform()],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    offsets = offset_left + offset_right
    expanded_position = vd.Vector([position, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    transformed = vd.matmul(expanded_position, transform)
    offset = vd.Vector([offsets[7, 7, 0], offsets[0, 0, 1]])
    transformed_position = vd.Vector([transformed[7, 0], transformed[7, 1]])
    return vd.Vector([transformed_position + offset, 0.0, 1.0])


@vd.fragment
def numpy_tensor_fragment(
    broadcast_left: Annotated[vd.Tensor[vd.f32, (8, 8, 4)], vd.uniform()],
    broadcast_right: Annotated[vd.Tensor[vd.f32, (1, 8, 1)], vd.uniform()],
    matmul_left: Annotated[vd.Tensor[vd.f32, (8, 8, 4)], vd.uniform()],
    matmul_right: Annotated[vd.Tensor[vd.f32, (1, 4, 2)], vd.uniform()],
) -> vd.Vector[vd.f32, 4]:
    broadcasted = broadcast_left + broadcast_right
    product = vd.matmul(matmul_left, matmul_right)
    return vd.Vector([broadcasted[0, 0, 0], broadcasted[7, 7, 3], product[0, 0, 1], product[7, 7, 0]])


@vd.struct
class DepthVertexOutput:
    position: Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]
    color: vd.Vector[vd.f32, 4]


@vd.vertex
def depth_vertex(
    position: Annotated[vd.Vector[vd.f32, 4], vd.attribute()],
    color: Annotated[vd.Vector[vd.f32, 4], vd.attribute()],
) -> DepthVertexOutput:
    return DepthVertexOutput(position, color)


@vd.fragment
def depth_fragment(
    color: Annotated[vd.Vector[vd.f32, 4], vd.varying()],
) -> vd.Vector[vd.f32, 4]:
    return color
