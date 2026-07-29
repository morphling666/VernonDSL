from .stage_registry import ENTRY_DECORATORS

FRONTEND_VERSION = 4
FUNCTION_DECORATORS = ENTRY_DECORATORS | {"func"}
INTRINSIC_METHODS = frozenset(
    {
        "norm",
        "normalize",
    }
)
GENERATED_BUILTINS = frozenset(
    {
        "fragment_coord",
        "front_facing",
        "instance_id",
        "resolution",
        "vertex_id",
    }
)
