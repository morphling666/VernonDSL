"""Static Python DSL surface for the Vernon compiler.

The compiler reads source through :mod:`ast`; it never imports or executes the
input module. These Python objects exist for editor completion and type syntax.
"""

from typing import Annotated

from .compiler import Compiler, compile_file, compile_source
from .decorators import compute, fragment, struct, vertex
from .diagnostics import CompileError
from .intrinsics import clamp, cross, dot, matmul, max, min, normalize, pow, reflect, texture_sample
from .types import (
    Array,
    Buffer,
    Sampler,
    Tensor,
    Texture,
    bool,
    builtin,
    f16,
    f32,
    f64,
    i32,
    instance,
    location,
    mat,
    mat2,
    mat3,
    mat4,
    resource,
    u32,
    uniform,
    varying,
    vec,
    vec2,
    vec3,
    vec4,
)

__all__ = [
    "Annotated",
    "Array",
    "Buffer",
    "CompileError",
    "Compiler",
    "Sampler",
    "Tensor",
    "Texture",
    "bool",
    "builtin",
    "clamp",
    "compile_file",
    "compile_source",
    "compute",
    "cross",
    "dot",
    "f16",
    "f32",
    "f64",
    "fragment",
    "i32",
    "instance",
    "location",
    "matmul",
    "mat",
    "mat2",
    "mat3",
    "mat4",
    "max",
    "min",
    "normalize",
    "pow",
    "resource",
    "reflect",
    "struct",
    "texture_sample",
    "u32",
    "uniform",
    "varying",
    "vec",
    "vec2",
    "vec3",
    "vec4",
    "vertex",
]
