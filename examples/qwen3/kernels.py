"""VernonDSL kernels for single-token (decode) inference of a Qwen3 dense transformer.

Weights, activations and the KV cache are f32: GPU targets do not support f16 yet (PROGRAM_GPU_F16_UNSUPPORTED).
Weights of one kind are stacked over the layers and read at the layer a kernel is given.

Every ordinary store must be owned by one invocation (specs/compiler/invocation_index_ownership.md):

- element-wise kernels store at indices affine in the global invocation id;
- a reduction runs in one workgroup through workgroup storage, and only lane 0 stores its result;
- the KV cache is stored at the current position, which no invocation id determines, so it is written with
  `atomic_add` into a zeroed cache: each element is added to exactly once per generation.

Reductions are written out in each kernel: a helper cannot take workgroup storage. Each is a tree over a
power-of-two number of lanes whose last barrier also orders every lane's read of the total before any later write.

`step` holds the decode state the host writes before each step: `step[0]` the position of the token, `step[1]` the
token id. It is a TensorView rather than kernel scalars, because a Module keeps its scalar arguments from its first
invocation.
"""

from typing import Annotated

import vernon_dsl as vd

# Lanes of a reducing workgroup; a power of two for the tree reduction.
LANES = 256
# Lanes of a per-head workgroup: one per element of a head. Qwen3 heads are 128 wide.
HEAD_DIM = 128


@vd.func
def silu(value: vd.f32) -> vd.f32:
    return value / (1.0 + vd.exp(-value))


@vd.kernel(workgroup_size=(HEAD_DIM, 1, 1))
def embed(
    table: vd.TensorView[vd.f32, (1, vd.dyn, vd.dyn), vd.read],
    step: vd.TensorView[vd.i32, (2,), vd.read],
    hidden: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    """hidden = table[0, token]."""
    column = gid[0]
    if column < hidden.shape[0]:
        hidden[column] = table[0, step[1], column]


@vd.kernel(workgroup_size=(LANES, 1, 1))
def rms_norm(
    x: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    weight: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.read],
    out: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    layer: vd.i32,
    eps: vd.f32,
    lane_id: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("local_invocation_id")],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    """out = x / rms(x) * weight[layer]. Each workgroup reduces all of x and normalizes LANES elements of it."""
    partial = vd.workgroup_storage(vd.f32, shape=(LANES,))
    lane = lane_id[0]
    size = vd.i32(x.shape[0])
    squares = 0.0
    for k in range(vd.i32(lane), size, LANES):
        squares += x[k] * x[k]
    partial[lane] = squares
    vd.workgroup_barrier()
    stride = vd.u32(LANES // 2)
    while stride > 0:
        if lane < stride:
            partial[lane] = partial[lane] + partial[lane + stride]
        vd.workgroup_barrier()
        stride = stride // 2
    scale = 1.0 / vd.sqrt(partial[0] / vd.f32(size) + eps)
    column = gid[0]
    if column < x.shape[0]:
        out[column] = x[column] * scale * weight[layer, column]


@vd.kernel(workgroup_size=(LANES, 1, 1))
def gemv(
    weight: vd.TensorView[vd.f32, (vd.dyn, vd.dyn, vd.dyn), vd.read],
    x: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    out: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    layer: vd.i32,
    lane_id: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("local_invocation_id")],
    group_id: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("workgroup_id")],
) -> None:
    """out[n] = weight[layer, n] . x, one workgroup per row."""
    partial = vd.workgroup_storage(vd.f32, shape=(LANES,))
    lane = lane_id[0]
    row = group_id[0]
    columns = vd.i32(x.shape[0])
    total = 0.0
    for k in range(vd.i32(lane), columns, LANES):
        total += weight[layer, row, k] * x[k]
    partial[lane] = total
    vd.workgroup_barrier()
    stride = vd.u32(LANES // 2)
    while stride > 0:
        if lane < stride:
            partial[lane] = partial[lane] + partial[lane + stride]
        vd.workgroup_barrier()
        stride = stride // 2
    if lane == 0:
        out[row] = partial[0]


@vd.kernel(workgroup_size=(LANES, 1, 1))
def gemv_residual(
    weight: vd.TensorView[vd.f32, (vd.dyn, vd.dyn, vd.dyn), vd.read],
    x: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    hidden: vd.TensorView[vd.f32, (vd.dyn,), vd.read_write],
    layer: vd.i32,
    lane_id: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("local_invocation_id")],
    group_id: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("workgroup_id")],
) -> None:
    """hidden[n] += weight[layer, n] . x, one workgroup per row."""
    partial = vd.workgroup_storage(vd.f32, shape=(LANES,))
    lane = lane_id[0]
    row = group_id[0]
    columns = vd.i32(x.shape[0])
    total = 0.0
    for k in range(vd.i32(lane), columns, LANES):
        total += weight[layer, row, k] * x[k]
    partial[lane] = total
    vd.workgroup_barrier()
    stride = vd.u32(LANES // 2)
    while stride > 0:
        if lane < stride:
            partial[lane] = partial[lane] + partial[lane + stride]
        vd.workgroup_barrier()
        stride = stride // 2
    if lane == 0:
        hidden[row] = hidden[row] + partial[0]


@vd.kernel(workgroup_size=(LANES, 1, 1))
def gemv_swiglu_residual(
    weight: vd.TensorView[vd.f32, (vd.dyn, vd.dyn, vd.dyn), vd.read],
    gate_up: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    hidden: vd.TensorView[vd.f32, (vd.dyn,), vd.read_write],
    layer: vd.i32,
    lane_id: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("local_invocation_id")],
    group_id: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("workgroup_id")],
) -> None:
    """hidden[n] += weight[layer, n] . (silu(gate) * up), where gate_up is gate followed by up."""
    partial = vd.workgroup_storage(vd.f32, shape=(LANES,))
    lane = lane_id[0]
    row = group_id[0]
    columns = vd.i32(weight.shape[2])
    total = 0.0
    for k in range(vd.i32(lane), columns, LANES):
        total += weight[layer, row, k] * (silu(gate_up[k]) * gate_up[columns + k])
    partial[lane] = total
    vd.workgroup_barrier()
    stride = vd.u32(LANES // 2)
    while stride > 0:
        if lane < stride:
            partial[lane] = partial[lane] + partial[lane + stride]
        vd.workgroup_barrier()
        stride = stride // 2
    if lane == 0:
        hidden[row] = hidden[row] + partial[0]


@vd.kernel(workgroup_size=(HEAD_DIM, 1, 1))
def qk_norm_rope_store(
    qkv: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    q_norm: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.read],
    k_norm: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.read],
    cos: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.read],
    sin: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.read],
    step: vd.TensorView[vd.i32, (2,), vd.read],
    q: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.write],
    k_cache: vd.TensorView[vd.f32, (vd.dyn, vd.dyn, vd.dyn), vd.read_write],
    v_cache: vd.TensorView[vd.f32, (vd.dyn, vd.dyn, vd.dyn), vd.read_write],
    layer: vd.i32,
    eps: vd.f32,
    lane_id: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("local_invocation_id")],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    """Per-head RMSNorm and RoPE of the query and key heads; keys and values are appended to the caches.

    One workgroup per query or key head of qkv, whose layout is query heads, key heads, value heads. Workgroups of
    query heads write q, scaled by 1/sqrt(HEAD_DIM); those of key heads append the key and value of their head.
    """
    partial = vd.workgroup_storage(vd.f32, shape=(HEAD_DIM,))
    lane = lane_id[0]
    head = gid[1]
    query_heads = q.shape[0]
    pos = step[0]
    value = qkv[head * HEAD_DIM + lane]
    partial[lane] = value * value
    vd.workgroup_barrier()
    stride = vd.u32(HEAD_DIM // 2)
    while stride > 0:
        if lane < stride:
            partial[lane] = partial[lane] + partial[lane + stride]
        vd.workgroup_barrier()
        stride = stride // 2
    scale = 1.0 / vd.sqrt(partial[0] / vd.f32(HEAD_DIM) + eps)
    half = vd.u32(HEAD_DIM // 2)
    # rotate_half pairs element i with i + HEAD_DIM / 2.
    partner_lane = lane + half
    frequency = lane
    sign = -1.0
    if lane >= half:
        partner_lane = lane - half
        frequency = lane - half
        sign = 1.0
    partner = qkv[head * HEAD_DIM + partner_lane]
    if head < query_heads:
        normed = value * scale * q_norm[layer, lane]
        normed_partner = partner * scale * q_norm[layer, partner_lane]
        rotated = normed * cos[pos, frequency] + sign * normed_partner * sin[pos, frequency]
        q[head, gid[0]] = rotated / vd.sqrt(vd.f32(HEAD_DIM))
    else:
        kv_heads = k_cache.shape[0] // k_norm.shape[0]
        kv_head = head - query_heads
        normed = value * scale * k_norm[layer, lane]
        normed_partner = partner * scale * k_norm[layer, partner_lane]
        rotated = normed * cos[pos, frequency] + sign * normed_partner * sin[pos, frequency]
        slot = vd.i32(vd.u32(layer) * kv_heads + kv_head)
        vd.atomic_add(k_cache, (slot, pos, vd.i32(lane)), rotated)
        stored = qkv[(head + kv_heads) * HEAD_DIM + lane]
        vd.atomic_add(v_cache, (slot, pos, vd.i32(lane)), stored)


@vd.kernel(workgroup_size=(HEAD_DIM, 1, 1))
def attention_scores(
    q: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.read],
    k_cache: vd.TensorView[vd.f32, (vd.dyn, vd.dyn, vd.dyn), vd.read],
    step: vd.TensorView[vd.i32, (2,), vd.read],
    scores: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.write],
    layer: vd.i32,
    kv_heads: vd.i32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    """scores[h, t] = q[h] . k[t] for every cached position t up to the current one."""
    position = gid[0]
    head = gid[1]
    if vd.i32(position) <= step[0] and head < q.shape[0]:
        group = q.shape[0] // vd.u32(kv_heads)
        slot = vd.u32(layer * kv_heads) + head // group
        total = 0.0
        for d in range(HEAD_DIM):
            total += q[head, d] * k_cache[slot, position, d]
        scores[head, position] = total


@vd.kernel(workgroup_size=(HEAD_DIM, 1, 1))
def attention_values(
    scores: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.read],
    v_cache: vd.TensorView[vd.f32, (vd.dyn, vd.dyn, vd.dyn), vd.read],
    step: vd.TensorView[vd.i32, (2,), vd.read],
    out: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    layer: vd.i32,
    kv_heads: vd.i32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    """out[h * HEAD_DIM + d] = softmax(scores[h]) . v[:, d], with an online softmax over the cached positions."""
    lane = gid[0]
    head = gid[1]
    group = scores.shape[0] // vd.u32(kv_heads)
    slot = vd.u32(layer * kv_heads) + head // group
    running_max = -3.0e38
    running_sum = 0.0
    total = 0.0
    for position in range(step[0] + 1):
        score = scores[head, position]
        updated_max = vd.max(running_max, score)
        correction = vd.exp(running_max - updated_max)
        weight = vd.exp(score - updated_max)
        running_sum = running_sum * correction + weight
        total = total * correction + weight * v_cache[slot, position, lane]
        running_max = updated_max
    out[lane + HEAD_DIM * head] = total / running_sum


@vd.kernel(workgroup_size=(HEAD_DIM, 1, 1))
def clear_cache(
    cache: vd.TensorView[vd.f32, (vd.dyn, vd.dyn, vd.dyn), vd.write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    cache[gid[2], gid[1], gid[0]] = 0.0
