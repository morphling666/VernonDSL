import vernon_dsl as vd


@vd.struct(shared=True)
class DirectAggregateParameters:
    scale: vd.f32
    bias: vd.f32


@vd.struct
class DirectAggregatePair:
    first: vd.f32
    second: vd.f32


@vd.struct
class DirectAggregateResult:
    pair: DirectAggregatePair
    tuple_values: vd.Tuple[vd.f32, vd.f32]
    values: vd.Tensor[vd.f32, (2,)]
    tag: vd.i32


@vd.kernel
def direct_aggregate_objective(
    value: vd.Tensor[vd.f32, (2,)],
    parameters: DirectAggregateParameters,
) -> DirectAggregateResult:
    scaled = value * parameters.scale + parameters.bias
    return DirectAggregateResult(
        DirectAggregatePair(scaled[0], scaled[1]),
        (scaled[0], scaled[1]),
        scaled,
        vd.i32(7),
    )
