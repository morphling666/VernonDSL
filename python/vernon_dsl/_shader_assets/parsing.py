from __future__ import annotations

import ast
import keyword
from pathlib import Path
from typing import Any

from ..bundle import PipelineCompileError, canonical_json
from ..decorators import ENTRY_DECORATOR_STAGES, GRAPHICS_STAGE_ORDER
from ..language.ast_utils import decorator_name, dotted_name
from ..module_graph import load_project
from .descriptors import ShaderModuleDescriptor, ShaderPipelineDescriptor, ShaderStageReference


def pipeline_asset_reference(value: str | Path) -> tuple[Path, str]:
    spelling = str(value)
    marker = spelling.rfind(".py:")
    if marker < 0:
        raise PipelineCompileError(
            "pipeline asset input must be a Python descriptor reference in source.py:descriptor_name form"
        )
    source = Path(spelling[: marker + 3]).resolve()
    descriptor_name = spelling[marker + 4 :]
    if not descriptor_name or not descriptor_name.isidentifier() or keyword.iskeyword(descriptor_name):
        raise PipelineCompileError("pipeline asset reference requires a valid Python descriptor name after source.py:")
    return source, descriptor_name


def _feature_bindings(tree: ast.Module) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        value = statement.value
        if not isinstance(value, ast.Call) or (dotted_name(value.func) or "").split(".")[-1] != "feature":
            continue
        if (
            len(value.args) != 1
            or value.keywords
            or not isinstance(value.args[0], ast.Constant)
            or not isinstance(value.args[0].value, str)
        ):
            raise PipelineCompileError("feature declarations used by pipeline assets require one string literal")
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if len(targets) != 1 or not isinstance(targets[0], ast.Name):
            raise PipelineCompileError("feature declarations used by pipeline assets require a simple name")
        bindings[targets[0].id] = value.args[0].value
    return bindings


def _literal_keyword(keywords: dict[str, ast.expr], name: str) -> Any:
    node = keywords.get(name)
    if node is None:
        raise PipelineCompileError(f"pipeline_asset declaration requires '{name}'")
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        raise PipelineCompileError(f"pipeline_asset '{name}' must be a literal value") from None


def parse_python_pipeline_asset(source: str | Path, descriptor_name: str) -> ShaderPipelineDescriptor:
    source_path = Path(source).resolve()
    if not source_path.is_file():
        raise PipelineCompileError(f"pipeline asset source does not exist: {source_path}")
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    except (OSError, SyntaxError) as error:
        raise PipelineCompileError(f"cannot parse pipeline asset source {source_path}: {error}") from None

    declarations = []
    for statement in tree.body:
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
            if isinstance(statement, ast.AnnAssign)
            else []
        )
        value = statement.value if isinstance(statement, (ast.Assign, ast.AnnAssign)) else None
        if (
            len(targets) == 1
            and isinstance(targets[0], ast.Name)
            and targets[0].id == descriptor_name
            and isinstance(value, ast.Call)
            and (dotted_name(value.func) or "").split(".")[-1] == "pipeline_asset"
        ):
            declarations.append(value)
    if not declarations:
        raise PipelineCompileError(f"pipeline asset declaration '{descriptor_name}' was not found in {source_path}")
    if len(declarations) != 1:
        raise PipelineCompileError(f"duplicate pipeline asset declaration: {descriptor_name}")
    declaration = declarations[0]
    if declaration.args:
        raise PipelineCompileError("pipeline_asset accepts keyword arguments only")
    keywords = {item.arg: item.value for item in declaration.keywords if item.arg is not None}
    if len(keywords) != len(declaration.keywords):
        raise PipelineCompileError("pipeline_asset does not accept expanded keyword arguments")
    unknown = set(keywords) - {"id", "program", "variants"}
    if unknown:
        raise PipelineCompileError("unknown pipeline_asset argument(s): " + ", ".join(sorted(unknown)))

    pipeline_id = _literal_keyword(keywords, "id")
    if not isinstance(pipeline_id, str) or not pipeline_id:
        raise PipelineCompileError("pipeline asset id must be a non-empty string")
    definitions = {
        statement.name: {decorator_name(decorator) for decorator in statement.decorator_list}
        for statement in tree.body
        if isinstance(statement, ast.FunctionDef)
    }

    def entry_stage(entry: ast.expr) -> tuple[str, str]:
        if not isinstance(entry, ast.Name):
            raise PipelineCompileError("pipeline_asset program entries must reference functions in the same module")
        decorators = definitions.get(entry.id)
        if decorators is None:
            raise PipelineCompileError(f"pipeline_asset program entry '{entry.id}' is not a function")
        stages = {ENTRY_DECORATOR_STAGES[name] for name in decorators if name in ENTRY_DECORATOR_STAGES}
        if len(stages) != 1:
            raise PipelineCompileError(
                f"pipeline_asset program entry '{entry.id}' must have exactly one entry-stage decorator"
            )
        return stages.pop(), entry.id

    program = keywords.get("program")
    if program is None:
        raise PipelineCompileError("pipeline_asset declaration requires 'program'")
    stage_functions: dict[str, str] = {}
    if isinstance(program, ast.Name):
        stage, entry = entry_stage(program)
        if stage != "compute":
            raise PipelineCompileError("single-entry pipeline_asset program must be a compute Kernel")
        stage_functions[stage] = entry
    elif isinstance(program, ast.Tuple):
        if not program.elts:
            raise PipelineCompileError("graphics pipeline_asset program must contain at least one stage")
        declared_stages: list[str] = []
        for item in program.elts:
            stage, entry = entry_stage(item)
            if stage == "compute":
                raise PipelineCompileError("graphics pipeline_asset program cannot contain a compute Kernel")
            if stage in stage_functions:
                raise PipelineCompileError(f"graphics pipeline_asset program contains duplicate '{stage}' stage")
            stage_functions[stage] = entry
            declared_stages.append(stage)
        order = {stage: index for index, stage in enumerate(GRAPHICS_STAGE_ORDER)}
        if declared_stages != sorted(declared_stages, key=lambda stage: order.get(stage, len(order))):
            raise PipelineCompileError("graphics pipeline_asset program stages are not in topology order")
    else:
        raise PipelineCompileError(
            "pipeline_asset program must be one compute Kernel or a tuple of graphics entry functions"
        )

    features = _feature_bindings(tree)
    variants_node = keywords.get("variants")
    variants: tuple[tuple[str, ...], ...]
    if variants_node is None:
        variants = ((),)
    elif not isinstance(variants_node, (ast.Tuple, ast.List)):
        raise PipelineCompileError("pipeline_asset variants must be a tuple or list")
    else:
        parsed: list[tuple[str, ...]] = []
        for row in variants_node.elts:
            if not isinstance(row, (ast.Tuple, ast.List)):
                raise PipelineCompileError("each pipeline asset variant must be a tuple or list")
            if any(not isinstance(item, ast.Name) or item.id not in features for item in row.elts):
                raise PipelineCompileError("pipeline asset variants must reference locally declared features")
            key = tuple(features[item.id] for item in row.elts if isinstance(item, ast.Name))
            if list(key) != sorted(key) or len(set(key)) != len(key):
                raise PipelineCompileError(f"pipeline asset variant is not canonical: {list(key)}")
            if key in parsed:
                raise PipelineCompileError(f"duplicate pipeline asset variant: {list(key)}")
            parsed.append(key)
        variants = tuple(parsed)
    if not variants:
        raise PipelineCompileError("pipeline must include at least one variant")
    if len(variants) > 16:
        raise PipelineCompileError(f"pipeline declares {len(variants)}, exceeding variant cap 16")

    requested = {name for variant in variants for name in variant}
    unknown_features = requested - set(load_project(source_path).features)
    if unknown_features:
        raise PipelineCompileError("variant requests undeclared feature(s): " + ", ".join(sorted(unknown_features)))

    module_id = f"python/{source_path.stem}"
    manifest = {
        "type": "python_pipeline_asset",
        "source": source_path.name,
        "name": descriptor_name,
        "id": pipeline_id,
        "stages": stage_functions,
        "variants": [list(key) for key in variants],
    }
    encoded = canonical_json(manifest)
    module = ShaderModuleDescriptor(module_id, source_path, source_path, encoded)
    stages = {stage: ShaderStageReference(module_id, entry) for stage, entry in stage_functions.items()}
    return ShaderPipelineDescriptor(
        pipeline_id,
        stages,
        variants,
        source_path,
        encoded,
        {module_id: module},
    )


__all__ = ["parse_python_pipeline_asset", "pipeline_asset_reference"]
