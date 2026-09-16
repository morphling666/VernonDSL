# ruff: noqa: F821

import os

import lit.formats
from lit.llvm import llvm_config

config.name = "VERNON_MLIR"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.vernon_mlir_test_exec_root
config.excludes = ["CMakeLists.txt"]

config.substitutions.append(("%vernon-opt", config.vernon_opt))
config.substitutions.append(("%FileCheck", config.filecheck))
config.substitutions.append(("%not", config.not_tool))
