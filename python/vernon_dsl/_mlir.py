from __future__ import annotations


def split_top_level(text: str, delimiter: str = ",") -> tuple[str, ...]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, character in enumerate(text):
        if character in "<[({":
            depth += 1
        elif character in ">])}":
            depth -= 1
            if depth < 0:
                return ()
        elif character == delimiter and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    if depth != 0:
        return ()
    parts.append(text[start:].strip())
    return tuple(parts)


def generic_type_arguments(spelling: str, constructor: str) -> tuple[str, ...] | None:
    text = spelling.strip()
    prefix = f"{constructor}<"
    if not text.startswith(prefix) or not text.endswith(">"):
        return None
    arguments = split_top_level(text[len(prefix) : -1])
    return arguments or None


def first_generic_type_argument(spelling: str, constructor: str) -> str | None:
    arguments = generic_type_arguments(spelling, constructor)
    return arguments[0] if arguments else None


def ranked_tensor_parts(spelling: str) -> tuple[tuple[str, ...], str] | None:
    arguments = generic_type_arguments(spelling, "tensor")
    if arguments is None or len(arguments) != 1:
        return None
    parts = arguments[0].split("x")
    if len(parts) < 2 or not parts[-1]:
        return None
    return tuple(parts[:-1]), parts[-1]


__all__ = [
    "first_generic_type_argument",
    "generic_type_arguments",
    "ranked_tensor_parts",
    "split_top_level",
]
