from vernon_dsl import *


def shadow_visibility(
    sampled_depth: f32,
    fragment_depth: f32,
    bias: f32,
    softness: f32,
) -> f32:
    distance = sampled_depth - fragment_depth + bias
    return clamp(distance * softness, 0.0, 1.0)
