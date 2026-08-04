from __future__ import annotations

import ast
import hashlib
import keyword
import re
from pathlib import Path
from typing import Any

from ..bundle import PipelineCompileError, canonical_json
from ..language.ast_utils import decorator_name, dotted_name
from ..language.stage_registry import ENTRY_DECORATOR_STAGES, validate_graphics_topology
from ..module_graph import load_project
from .descriptors import ShaderModuleDescriptor, ShaderPipelineDescriptor, ShaderStageReference

_AD_PATH = re.compile(r"^[A-Za-z_]\w*(?:\.(?:[A-Za-z_]\w*|\d+))*$")


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


def _ad_rule_sets(tree: ast.Module) -> dict[str, tuple[str, tuple[str, ...]]]:
    declarations: dict[str, tuple[str, tuple[str, ...]]] = {}
    allowed = {"rasterization", "visibility", "depth", "blend", "texture"}
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        value = statement.value
        if (
            len(targets) != 1
            or not isinstance(targets[0], ast.Name)
            or not isinstance(value, ast.Call)
            or (dotted_name(value.func) or "").split(".")[-1] != "rule_set"
        ):
            continue
        if value.args or any(item.arg is None for item in value.keywords):
            raise PipelineCompileError("vd.ad.rule_set accepts keyword arguments only")
        keywords = {item.arg: item.value for item in value.keywords if item.arg is not None}
        unknown = set(keywords) - allowed - {"id"}
        if unknown:
            raise PipelineCompileError("unknown VJP rule(s): " + ", ".join(sorted(unknown)))
        rule_id = keywords.get("id")
        if not isinstance(rule_id, ast.Constant) or not isinstance(rule_id.value, str) or not rule_id.value:
            raise PipelineCompileError("VJP rule set id must be a non-empty string literal")
        for name, rule in keywords.items():
            if name != "id" and not isinstance(rule, ast.Name):
                raise PipelineCompileError(f"VJP rule '{name}' must reference a module-level function")
        declarations[targets[0].id] = (rule_id.value, tuple(sorted(set(keywords) - {"id"})))
    return declarations


def _transform_record(
    wrt: tuple[str, ...],
    rule_set_id: str | None,
    rule_set_identity: str | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "kind": "vjp",
        "wrt": list(wrt),
        "output_cotangents": [],
        "gradient_policy": "f16:f32,f32:f32,f64:f64",
        "accumulation_policy": "fresh",
        "tape_policy": "bounded",
        "derivative_rules_version": 1,
    }
    if rule_set_id is not None:
        record["rule_set"] = rule_set_id
        record["rule_set_identity"] = rule_set_identity
    record["identity"] = hashlib.sha256(canonical_json(record).encode("utf-8")).hexdigest()
    return record


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

    program_expression = keywords.get("program")
    if program_expression is None:
        raise PipelineCompileError("pipeline_asset declaration requires 'program'")
    transform: dict[str, Any] | None = None
    if (
        isinstance(program_expression, ast.Call)
        and (dotted_name(program_expression.func) or "").split(".")[-1] == "vjp"
    ):
        if len(program_expression.args) != 1 or any(item.arg is None for item in program_expression.keywords):
            raise PipelineCompileError("vd.ad.vjp requires one program operand and keyword arguments")
        transform_keywords = {item.arg: item.value for item in program_expression.keywords if item.arg is not None}
        unknown_transform = set(transform_keywords) - {"wrt", "rules"}
        if unknown_transform:
            raise PipelineCompileError("unknown vd.ad.vjp argument(s): " + ", ".join(sorted(unknown_transform)))
        wrt_node = transform_keywords.get("wrt")
        try:
            wrt_value = ast.literal_eval(wrt_node) if wrt_node is not None else None
        except (ValueError, TypeError, SyntaxError):
            wrt_value = None
        if (
            not isinstance(wrt_value, (tuple, list))
            or not wrt_value
            or any(not isinstance(path, str) or not _AD_PATH.fullmatch(path) for path in wrt_value)
        ):
            raise PipelineCompileError("vd.ad.vjp wrt must be a non-empty literal tuple or list of source paths")
        if len(set(wrt_value)) != len(wrt_value):
            raise PipelineCompileError("vd.ad.vjp wrt paths must be unique")
        rule_set_id = None
        rule_set_identity = None
        rules_node = transform_keywords.get("rules")
        if rules_node is not None:
            if not isinstance(rules_node, ast.Name):
                raise PipelineCompileError("vd.ad.vjp rules must reference a module-level vd.ad.rule_set")
            declared_rule_sets = _ad_rule_sets(tree)
            if rules_node.id not in declared_rule_sets:
                raise PipelineCompileError(f"unknown VJP rule set '{rules_node.id}'")
            rule_set_id, rule_names = declared_rule_sets[rules_node.id]
            rule_set_record = {"id": rule_set_id, "rules": list(rule_names)}
            rule_set_identity = hashlib.sha256(canonical_json(rule_set_record).encode("utf-8")).hexdigest()
        transform = _transform_record(
            tuple(sorted(wrt_value)),
            rule_set_id,
            rule_set_identity,
        )
        program = program_expression.args[0]
    else:
        program = program_expression
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
        try:
            validate_graphics_topology(declared_stages)
        except ValueError as error:
            message = str(error).replace("graphics pipeline", "graphics pipeline_asset program")
            raise PipelineCompileError(message) from None
    else:
        raise PipelineCompileError(
            "pipeline_asset program must be one compute Kernel or a tuple of graphics entry functions"
        )
    if transform is not None:
        graphics = set(stage_functions) != {"compute"}
        if graphics and "rule_set" not in transform:
            raise PipelineCompileError("graphics VJP requires a named custom rule set")
        if not graphics and "rule_set" in transform:
            raise PipelineCompileError("compute VJP does not accept graphics custom rules")

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
    if transform is not None:
        manifest["transform"] = transform
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
        transform,
    )


__all__ = ["parse_python_pipeline_asset", "pipeline_asset_reference"]
