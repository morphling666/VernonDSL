FRONTEND_VERSION = 3

ENTRY_DECORATORS = frozenset({"kernel", "vertex", "fragment"})
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
