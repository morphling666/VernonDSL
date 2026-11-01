from .stage_registry import ENTRY_DECORATORS

# Host functions run on the CPU host, never as a shader stage; externs declare C functions Host code calls.
HOST_FUNCTION_DECORATORS = frozenset({"host", "extern"})
FUNCTION_DECORATORS = ENTRY_DECORATORS | {"func"} | HOST_FUNCTION_DECORATORS


def library_symbol(library: tuple[str, int], function: str) -> str:
    """The C symbol `vd.library(name, version)` gives an extern declared without `symbol=`."""

    return f"{library[0]}_v{library[1]}_{function}"


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
