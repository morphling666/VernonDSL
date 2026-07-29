#include "compiler_dispatch.h"

#include "compiler_artifacts.h"
#include "compiler_cpu.h"
#include "compiler_cuda.h"
#include "compiler_dxc.h"
#include "compiler_frontend.h"
#include "compiler_spirv.h"
#include "compiler_spirv_cross.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/IR/VernonValueAbi.h"

#include <cstring>
#include <string>

namespace vernon::compiler {
namespace {

std::string copyStringView(VernonStringView value) {
    return value.data && value.size ? std::string(value.data, value.size) : std::string();
}

bool validateTargetCapabilities(PreparedModule &prepared, VernonTarget target, std::string &diagnostics) {
    mlir::OwningOpRef<mlir::ModuleOp> module = prepared.clone();
    bool usesDeviceAtomics = false;
    module->walk([&](mlir::vernon::AtomicOp atomic) {
        auto view = mlir::dyn_cast<mlir::vernon::TensorViewType>(atomic.getStorage().getType());
        usesDeviceAtomics |= view && view.getAddressSpace() == "device";
    });
    if (usesDeviceAtomics && target != VERNON_TARGET_CPU && target != VERNON_TARGET_CUDA &&
        target != VERNON_TARGET_VULKAN) {
        diagnostics = "device-scope storage TensorView atomics are supported only by CPU, CUDA, and Vulkan targets";
        return false;
    }
    if (target != VERNON_TARGET_CPU && target != VERNON_TARGET_CUDA)
        return true;
    if (target == VERNON_TARGET_CPU) {
        bool invalid = false;
        module->walk([&](mlir::Operation *operation) {
            if (invalid ||
                !mlir::isa<mlir::vernon::WorkgroupAllocOp, mlir::vernon::AtomicOp, mlir::vernon::BarrierOp>(operation))
                return;
            if (auto atomic = mlir::dyn_cast<mlir::vernon::AtomicOp>(operation); atomic) {
                auto view = mlir::dyn_cast<mlir::vernon::TensorViewType>(atomic.getStorage().getType());
                if (!view) {
                    diagnostics = "atomic storage operand is not a TensorView";
                    invalid = true;
                    return;
                }
                if (view.getAddressSpace() == "device")
                    return;
            }
            mlir::func::FuncOp function = operation->getParentOfType<mlir::func::FuncOp>();
            auto size = function ? function->getAttrOfType<mlir::DenseI32ArrayAttr>("vernon.workgroup_size") : nullptr;
            if (!size || size.size() != 3 || size[0] != 1 || size[1] != 1 || size[2] != 1)
                invalid = true;
        });
        if (invalid)
            diagnostics =
                "CPU reference synchronization requires workgroup_size=(1, 1, 1); use a GPU target for cooperative "
                "workgroups";
        return !invalid;
    }
    for (mlir::func::FuncOp function : module->getOps<mlir::func::FuncOp>()) {
        auto entry = function->getAttrOfType<mlir::UnitAttr>("vernon.entry");
        if (!entry)
            continue;
        auto stage = function->getAttrOfType<mlir::StringAttr>("vernon.stage");
        if (!stage || stage.getValue() != "compute") {
            diagnostics = "CUDA target capability rejects non-compute entry '" + function.getSymName().str() + "'";
            return false;
        }
        for (unsigned index = 0; index < function.getNumArguments(); ++index) {
            if (function.getArgAttrDict(index).get("vernon.builtin"))
                continue;
            mlir::FailureOr<mlir::vernon::PhysicalValueAbiPlan> plan = mlir::vernon::getPhysicalValueAbiPlan(
                function.getArgumentTypes()[index], *module, mlir::vernon::PhysicalAbiProfile::CudaKernelParameter);
            if (mlir::failed(plan)) {
                diagnostics = "CUDA target capability cannot plan compute argument #" + std::to_string(index);
                return false;
            }
            const auto *unsupported = std::get_if<mlir::vernon::UnsupportedPhysicalValueAbi>(&*plan);
            if (!unsupported)
                continue;
            diagnostics = "CUDA target capability '" + unsupported->reason + "' rejects compute argument #" +
                          std::to_string(index);
            return false;
        }
    }
    bool deviceBarrier = false;
    module->walk([&](mlir::vernon::BarrierOp barrier) { deviceBarrier |= barrier.getScope() == "device"; });
    if (deviceBarrier) {
        diagnostics = "CUDA target capability rejects device-scope barriers; use workgroup_barrier";
        return false;
    }
    return true;
}

} // namespace

VernonTargetCapabilities targetCapabilities(VernonTarget target) {
#if !defined(VERNON_DXC_EXECUTABLE)
    if (target == VERNON_TARGET_DIRECTX)
        return VernonTargetCapabilities{0, 0, 0, 0};
#endif
    if (target == VERNON_TARGET_CPU || target == VERNON_TARGET_VULKAN)
        return VernonTargetCapabilities{1, 1, 1, 0};
    if (target == VERNON_TARGET_CUDA)
        return VernonTargetCapabilities{1, 0, 1, 0};
    if (target == VERNON_TARGET_OPENGL || target == VERNON_TARGET_OPENGL_ES || target == VERNON_TARGET_METAL ||
        target == VERNON_TARGET_DIRECTX)
        return VernonTargetCapabilities{1, 1, 1, 0};
    // Do not advertise an IR-only path as a usable target; availability means
    // the complete lowering and artifact pipeline is linked.
    return VernonTargetCapabilities{0, 0, 0, 0};
}

VernonStatus parseCompileOptions(const VernonCompileOptions *source, VernonTarget target, CompileOptions &options,
                                 std::string &diagnostics) {
    if (!source)
        return VERNON_STATUS_OK;
    constexpr size_t requiredSize = offsetof(VernonCompileOptions, glsl_version) + sizeof(uint32_t);
    if (source->struct_size < requiredSize) {
        diagnostics = "compile options structure is too small";
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    options.glslVersion = source->glsl_version;
    if (options.glslVersion != 0 && target != VERNON_TARGET_OPENGL && target != VERNON_TARGET_OPENGL_ES) {
        diagnostics = "GLSL version is valid only for OpenGL and OpenGL ES targets";
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    if (options.glslVersion != 0 && (options.glslVersion < 100 || options.glslVersion > 999)) {
        diagnostics = "GLSL version must be a three-digit version number";
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    const uint32_t requestedHlslShaderModel =
        source->struct_size >= offsetof(VernonCompileOptions, hlsl_shader_model) + sizeof(uint32_t)
            ? source->hlsl_shader_model
            : 0;
    if (requestedHlslShaderModel != 0)
        options.hlslShaderModel = requestedHlslShaderModel;
    const bool validHlslShaderModel = options.hlslShaderModel >= 60;
    if (requestedHlslShaderModel != 0 && target != VERNON_TARGET_DIRECTX) {
        diagnostics = "HLSL Shader Model is valid only for the DirectX target";
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    if (target == VERNON_TARGET_DIRECTX && !validHlslShaderModel) {
        diagnostics = "DirectX runtime artifacts require HLSL Shader Model 6.0 or newer";
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    auto readCpuOption = [&](size_t offset, std::string &destination) {
        if (source->struct_size < offset + sizeof(VernonStringView))
            return true;
        VernonStringView value{};
        std::memcpy(&value, reinterpret_cast<const char *>(source) + offset, sizeof(value));
        if (value.size != 0 && !value.data) {
            diagnostics = "CPU compile option has null data";
            return false;
        }
        destination = copyStringView(value);
        return true;
    };
    if (!readCpuOption(offsetof(VernonCompileOptions, cpu_target_triple), options.cpu.targetTriple) ||
        !readCpuOption(offsetof(VernonCompileOptions, cpu_name), options.cpu.cpu) ||
        !readCpuOption(offsetof(VernonCompileOptions, cpu_features), options.cpu.features))
        return VERNON_STATUS_INVALID_ARGUMENT;
    if (target != VERNON_TARGET_CPU &&
        (!options.cpu.targetTriple.empty() || !options.cpu.cpu.empty() || !options.cpu.features.empty())) {
        diagnostics = "CPU code generation options are valid only for the CPU target";
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    return VERNON_STATUS_OK;
}

VernonStatus compileTarget(PreparedModule &module, VernonTarget target, const CompileOptions &options,
                           std::vector<Artifact> &artifacts, std::string &reflection, std::string &diagnostics,
                           CpuExecutionStatePtr &cpuExecution) {
    diagnostics.clear();
    if (!validateTargetCapabilities(module, target, diagnostics)) {
        artifacts.clear();
        return VERNON_STATUS_UNSUPPORTED_TARGET;
    }
    if (target == VERNON_TARGET_CPU) {
        if (!compileCpu(module, options.cpu, artifacts, reflection, diagnostics, cpuExecution)) {
            artifacts.clear();
            return VERNON_STATUS_INTERNAL_ERROR;
        }
        addArtifactTable(reflection, artifacts, target, options.glslVersion, options.cpu.targetTriple, options.cpu.cpu,
                         options.cpu.features);
        return VERNON_STATUS_OK;
    }
    if (target == VERNON_TARGET_VULKAN || target == VERNON_TARGET_OPENGL || target == VERNON_TARGET_OPENGL_ES ||
        target == VERNON_TARGET_METAL || target == VERNON_TARGET_DIRECTX) {
        if (!compileSpirv(module, target, artifacts, diagnostics)) {
            artifacts.clear();
            return VERNON_STATUS_INTERNAL_ERROR;
        }
        if (target != VERNON_TARGET_VULKAN &&
            !crossCompileSpirv(artifacts, diagnostics, target, options.glslVersion, options.hlslShaderModel)) {
            artifacts.clear();
            return VERNON_STATUS_INTERNAL_ERROR;
        }
        if (target == VERNON_TARGET_DIRECTX) {
            std::vector<Artifact> dxilArtifacts;
            if (!compileHlslToDxil(artifacts, options.hlslShaderModel, dxilArtifacts, diagnostics)) {
                artifacts.clear();
                return VERNON_STATUS_INTERNAL_ERROR;
            }
            artifacts = std::move(dxilArtifacts);
        }
        addArtifactTable(reflection, artifacts, target, options.glslVersion, {}, {}, {}, options.hlslShaderModel);
        return VERNON_STATUS_OK;
    }
    if (target == VERNON_TARGET_CUDA) {
        if (!compileCuda(module, artifacts, diagnostics)) {
            artifacts.clear();
            return VERNON_STATUS_INTERNAL_ERROR;
        }
        addArtifactTable(reflection, artifacts, target, options.glslVersion);
        return VERNON_STATUS_OK;
    }
    artifacts.clear();
    diagnostics = "the requested target lowering pipeline is not available";
    return VERNON_STATUS_UNSUPPORTED_TARGET;
}

bool linkCpuHostObject(const void *object, size_t objectSize, Artifact &artifact, std::string &diagnostics) {
    return linkHostObject(object, objectSize, artifact, diagnostics);
}

VernonCpuEntryPoint findCompiledCpuEntry(const CpuExecutionState *execution, std::string_view entry) {
    return findCpuEntry(execution, entry);
}

} // namespace vernon::compiler
