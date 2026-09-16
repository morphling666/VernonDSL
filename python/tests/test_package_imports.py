from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_shader_contract_import_keeps_native_lazy_and_public_exports_resolve() -> None:
    root = Path(__file__).resolve().parents[2]
    source = """
import sys

import vernon_dsl.shader_contracts as shader_contracts

assert shader_contracts.BUILTIN_CONTRACTS
assert "vernon_dsl._native" not in sys.modules

import vernon_dsl as vd

assert "vernon_dsl._native" not in sys.modules
assert set(vd.__all__) <= set(dir(vd))

from vernon_dsl import Architecture, Module, ProgramAssetDeclaration, RenderPass, TensorStorage, zeros
from vernon_dsl.module import Module as ExpectedModule
from vernon_dsl.program_assets import ProgramAssetDeclaration as ExpectedProgramAssetDeclaration
from vernon_dsl.render import RenderPass as ExpectedRenderPass
from vernon_dsl.runtime import Architecture as ExpectedArchitecture
from vernon_dsl.runtime import TensorStorage as ExpectedTensorStorage
from vernon_dsl.storage import zeros as expected_zeros

assert Architecture is ExpectedArchitecture
assert Module is ExpectedModule
assert ProgramAssetDeclaration is ExpectedProgramAssetDeclaration
assert RenderPass is ExpectedRenderPass
assert TensorStorage is ExpectedTensorStorage
assert zeros is expected_zeros

star_exports = {}
exec("from vernon_dsl import *", star_exports)
assert set(vd.__all__) <= set(star_exports)
"""
    python_path = os.pathsep.join(filter(None, (str(root / "python"), os.environ.get("PYTHONPATH"))))
    subprocess.run(
        [sys.executable, "-c", source],
        cwd=root,
        env={**os.environ, "PYTHONPATH": python_path},
        check=True,
    )
