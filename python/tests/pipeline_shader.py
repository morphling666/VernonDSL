from typing import Annotated

import vernon_dsl as vd

OPTIONAL_IMAGE = vd.feature("OPTIONAL_IMAGE")


@vd.kernel(workgroup_size=(2, 1, 1))
def translate_vertices(
    position: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.read_write],
    offset: vd.f32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    component = gid[0]
    vertex = gid[1]
    if component == 0:
        position[vertex, component] = position[vertex, component] + offset


@vd.kernel(workgroup_size=(4, 1, 1))
def copy_static_tensor_value(
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
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
        output[invocation] = singleton[0]
    if invocation == 1:
        output[invocation] = weights[0, 0, 0]
    if invocation == 2:
        output[invocation] = weights[1, 0, 1]
    if invocation == 3:
        output[invocation] = weights[1, 1, 1]
    if invocation == 4:
        output[invocation] = doubled[1, 1, 0]
    if invocation == 5:
        output[invocation] = large[1, 2, 4]
    if invocation == 6:
        output[invocation] = quad[3]
    if invocation == 7:
        output[invocation] = matrix_product[0, 0]
    if invocation == 8:
        output[invocation] = matrix_product[0, 1]
    if invocation == 9:
        output[invocation] = matrix_product[1, 0]
    if invocation == 10:
        output[invocation] = matrix_product[1, 1]


@vd.vertex
def triangle_vertex(
    position: Annotated[vd.Vector[vd.f32, 2], vd.attribute()],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    return vd.Vector([position, 0.0, 1.0])


@vd.struct
class HeightmapVertexOutput:
    position: Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]
    color: vd.Vector[vd.f32, 4]


@vd.vertex
def heightmap_vertex(
    position: Annotated[vd.Vector[vd.f32, 2], vd.attribute()],
    uv: Annotated[vd.Vector[vd.f32, 2], vd.attribute()],
    heightmap: Annotated[vd.Texture["2d", vd.f32], vd.resource(set=0, binding=0)],  # noqa: F722, F821
    heightmap_sampler: Annotated[vd.Sampler, vd.resource(set=0, binding=1)],
) -> HeightmapVertexOutput:
    sample = vd.texture_sample(heightmap, heightmap_sampler, uv, 0.0)
    return HeightmapVertexOutput(vd.Vector([position, sample.x * 0.5, 1.0]), sample)


@vd.fragment
def heightmap_fragment(
    color: Annotated[vd.Vector[vd.f32, 4], vd.varying()],
) -> vd.Vector[vd.f32, 4]:
    return color


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
def spilled_matrix_vertex(
    position: Annotated[vd.Vector[vd.f32, 2], vd.attribute()],
    projection: Annotated[vd.Matrix[vd.f32, 4, 4], vd.uniform()],
    view: Annotated[vd.Matrix[vd.f32, 4, 4], vd.uniform()],
    model: Annotated[vd.Matrix[vd.f32, 4, 4], vd.uniform()],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    return vd.matmul(projection, vd.matmul(view, vd.matmul(model, vd.Vector([position, 0.0, 1.0]))))


@vd.struct(shared=True)
class Placement:
    scale: vd.f32
    transform: vd.Matrix[vd.f32, 4, 4]


@vd.vertex
def struct_matrix_vertex(
    position: Annotated[vd.Vector[vd.f32, 2], vd.attribute()],
    placement: Annotated[Placement, vd.uniform()],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    return vd.matmul(placement.transform, vd.Vector([position * placement.scale, 0.0, 1.0]))


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
def mixed_uniform_fragment(
    first: Annotated[vd.Vector[vd.f32, 3], vd.uniform()],
    second: Annotated[vd.Vector[vd.f32, 3], vd.uniform()],
    scale: Annotated[vd.f32, vd.uniform()],
    bias: Annotated[vd.f32, vd.uniform()],
    uv: Annotated[vd.Vector[vd.f32, 2], vd.uniform()],
    texel: Annotated[vd.Vector[vd.f32, 2], vd.uniform()],
) -> vd.Vector[vd.f32, 4]:
    return vd.Vector([first.x + second.x, first.y + scale, first.z + bias, uv.x + texel.x])


@vd.fragment
def cube_direction_fragment(
    direction: Annotated[vd.Vector[vd.f32, 3], vd.uniform()],
    image: Annotated[vd.Texture["cube", vd.f32], vd.resource(set=0, binding=0)],  # noqa: F722, F821
    sampler: Annotated[vd.Sampler, vd.resource(set=0, binding=1)],
) -> vd.Vector[vd.f32, 4]:
    return vd.texture_sample(image, sampler, direction)


@vd.fragment
def volume_coordinate_fragment(
    coordinate: Annotated[vd.Vector[vd.f32, 3], vd.uniform()],
    image: Annotated[vd.Texture["3d", vd.f32], vd.resource(set=0, binding=0)],  # noqa: F722, F821
    sampler: Annotated[vd.Sampler, vd.resource(set=0, binding=1)],
) -> vd.Vector[vd.f32, 4]:
    return vd.texture_sample(image, sampler, coordinate)


@vd.fragment
def array_layer_fragment(
    coordinate: Annotated[vd.Vector[vd.f32, 3], vd.uniform()],
    image: Annotated[vd.Texture["2d_array", vd.f32], vd.resource(set=0, binding=0)],  # noqa: F722, F821
    sampler: Annotated[vd.Sampler, vd.resource(set=0, binding=1)],
) -> vd.Vector[vd.f32, 4]:
    return vd.texture_sample(image, sampler, coordinate)


@vd.fragment
def cube_array_fragment(
    direction: Annotated[vd.Vector[vd.f32, 4], vd.uniform()],
    image: Annotated[vd.Texture["cube_array", vd.f32], vd.resource(set=0, binding=0)],  # noqa: F722, F821
    sampler: Annotated[vd.Sampler, vd.resource(set=0, binding=1)],
) -> vd.Vector[vd.f32, 4]:
    return vd.texture_sample(image, sampler, direction)


@vd.fragment
def flat_coordinate_fragment(
    uv: Annotated[vd.Vector[vd.f32, 2], vd.uniform()],
    image: Annotated[vd.Texture["2d", vd.f32], vd.resource(set=0, binding=0)],  # noqa: F722, F821
    sampler: Annotated[vd.Sampler, vd.resource(set=0, binding=1)],
) -> vd.Vector[vd.f32, 4]:
    return vd.texture_sample(image, sampler, uv)


@vd.fragment
def shared_sampler_fragment(
    uv: Annotated[vd.Vector[vd.f32, 2], vd.uniform()],
    base: vd.Texture["2d", vd.f32],  # noqa: F722, F821
    detail: vd.Texture["2d", vd.f32],  # noqa: F722, F821
    sampler: vd.Sampler,
) -> vd.Vector[vd.f32, 4]:
    return vd.texture_sample(base, sampler, uv) + vd.texture_sample(detail, sampler, uv)


@vd.fragment
def identifier_fetch_fragment(
    identifiers: Annotated[vd.Texture["2d", vd.u32], vd.resource(set=0, binding=0)],  # noqa: F722, F821
) -> vd.u32:
    pixel = vd.fragment_coord()
    identifier = vd.texture_load(identifiers, vd.Vector([vd.i32(pixel.x), vd.i32(pixel.y)]))[0]
    return identifier // vd.u32(7) + identifier % vd.u32(7) * vd.u32(0x20000000)


@vd.struct
class IdentifierVaryings:
    position: Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]
    base: vd.u32


@vd.vertex
def identifier_vertex(
    position: Annotated[vd.Vector[vd.f32, 2], vd.attribute()],
    base: Annotated[vd.u32, vd.attribute()],
) -> IdentifierVaryings:
    return IdentifierVaryings(vd.Vector([position, 0.0, 1.0]), base)


@vd.struct
class IdentifierTargets:
    color: vd.Vector[vd.f32, 4]
    identifier: vd.u32


@vd.fragment
def identifier_mrt_fragment(
    base: Annotated[vd.u32, vd.varying()],
) -> IdentifierTargets:
    pixel = vd.fragment_coord()
    identifier = base + vd.u32(pixel.x) * vd.u32(0x40000000) + vd.u32(pixel.y)
    return IdentifierTargets(vd.Vector([1.0, 0.25, 0.0, 1.0]), identifier)


@vd.fragment
def identifier_pair_fragment(
    base: Annotated[vd.u32, vd.varying()],
) -> vd.Vector[vd.u32, 2]:
    pixel = vd.fragment_coord()
    column = vd.u32(pixel.x)
    row = vd.u32(pixel.y)
    return vd.Vector([base + column * vd.u32(0x40000000) + row, vd.u32(0xFFFFFFFF) - column * vd.u32(0x01010101) - row])


@vd.struct
class InterpolatedVaryings:
    position: Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]
    perspective: vd.f32
    linear: Annotated[vd.f32, vd.varying(vd.Interpolation.LINEAR)]
    flat: Annotated[vd.f32, vd.varying(vd.Interpolation.FLAT)]


@vd.vertex
def interpolation_vertex(
    position: Annotated[vd.Vector[vd.f32, 4], vd.attribute()],
    value: Annotated[vd.f32, vd.attribute()],
) -> InterpolatedVaryings:
    return InterpolatedVaryings(position, value, value, value)


@vd.fragment
def interpolation_fragment(
    perspective: Annotated[vd.f32, vd.varying()],
    linear: Annotated[vd.f32, vd.varying(vd.Interpolation.LINEAR)],
    flat: Annotated[vd.f32, vd.varying(vd.Interpolation.FLAT)],
) -> vd.Vector[vd.f32, 4]:
    return vd.Vector([perspective, linear, flat, 1.0])


@vd.fragment
def optional_texture_fragment(
    base_image: Annotated[vd.Texture["2d", vd.f32], vd.resource(set=0, binding=0)],  # noqa: F722, F821
    base_sampler: Annotated[vd.Sampler, vd.resource(set=0, binding=1)],
    optional_image: Annotated[vd.Texture["2d", vd.f32], vd.resource(set=0, binding=2)],  # noqa: F722, F821
    optional_sampler: Annotated[vd.Sampler, vd.resource(set=0, binding=3)],
) -> vd.Vector[vd.f32, 4]:
    result = vd.texture_sample(base_image, base_sampler, vd.Vector([0.5, 0.5]))
    optional = vd.texture_sample(optional_image, optional_sampler, vd.Vector([0.5, 0.5]))
    if OPTIONAL_IMAGE:
        result = optional
    return result


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


@vd.fragment
def array_uniform_fragment(
    grid: Annotated[vd.Tensor[vd.f32, (5, 6)], vd.uniform()],
    ramp: Annotated[vd.Tensor[vd.f32, (8,)], vd.uniform()],
    scale: Annotated[vd.f32, vd.uniform()],
) -> vd.Vector[vd.f32, 4]:
    return vd.Vector([grid[0, 1] * scale, grid[4, 5], ramp[6], ramp[7]])


@vd.struct(shared=True)
class CascadeTable:
    # Starts on a 16-byte boundary, so runtime reads index the uniform block.
    matrices: vd.Tensor[vd.f32, (2, 4, 4)]
    splits: vd.Vector[vd.f32, 4]
    bias: vd.f32
    # Starts off a 16-byte boundary, so runtime reads index a copy.
    weights: vd.Vector[vd.f32, 4]
    selector: vd.u32


@vd.fragment
def runtime_indexed_uniform_fragment(
    table: Annotated[CascadeTable, vd.uniform()],
) -> vd.Vector[vd.f32, 4]:
    cascade = vd.i32(table.selector)
    local = vd.Vector([table.bias, table.splits[0], table.splits[3]])
    return vd.Vector(
        [
            table.matrices[cascade, 2, 3],
            table.splits[cascade + 1],
            table.weights[cascade + 2],
            local[cascade],
        ]
    )


@vd.fragment
def mixed_inline_uniform_fragment(
    bias: Annotated[vd.f32, vd.uniform()],
    tint: Annotated[vd.Vector[vd.f32, 3], vd.uniform()],
    basis: Annotated[vd.Matrix[vd.f32, 2, 2], vd.uniform()],
    gain: Annotated[vd.f32, vd.uniform()],
    shift: Annotated[vd.Vector[vd.f32, 2], vd.uniform()],
) -> vd.Vector[vd.f32, 4]:
    mixed = vd.matmul(basis, shift)
    return vd.Vector([bias + tint[0], tint[2] * gain, mixed[0], mixed[1]])


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
