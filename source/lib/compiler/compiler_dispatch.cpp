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
    module->walk([&](mlir::vernon::PhysicalAtomicOp atomic) {
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
                !mlir::isa<mlir::vernon::WorkgroupAllocOp, mlir::vernon::PhysicalAtomicOp, mlir::vernon::BarrierOp>(
                    operation))
                return;
            if (auto atomic = mlir::dyn_cast<mlir::vernon::PhysicalAtomicOp>(operation); atomic) {
                auto view = mlir::dyn_cast<mlir::vernon::TensorViewType>(atomic.getStorage().getType());
                if (!view) {
                    diagnostics = "physical atomic storage operand is not a TensorView";
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

CompileOptions defaultCompileOptions(VernonTarget target) {
    switch (target) {
    case VERNON_TARGET_CPU:
        return CpuCodegenOptions{};
    case VERNON_TARGET_OPENGL:
    case VERNON_TARGET_OPENGL_ES:
        return OpenGLCompileOptions{target, 0};
    case VERNON_TARGET_VULKAN:
        return VulkanCompileOptions{};
    case VERNON_TARGET_METAL:
        return MetalCompileOptions{};
    case VERNON_TARGET_DIRECTX:
        return DirectXCompileOptions{};
    case VERNON_TARGET_CUDA:
        return CudaCompileOptions{};
    }
    return VulkanCompileOptions{};
}

VernonTarget compileTargetKind(const CompileOptions &options) {
    if (std::holds_alternative<CpuCodegenOptions>(options))
        return VERNON_TARGET_CPU;
    if (const auto *opengl = std::get_if<OpenGLCompileOptions>(&options))
        return opengl->target;
    if (std::holds_alternative<VulkanCompileOptions>(options))
        return VERNON_TARGET_VULKAN;
    if (std::holds_alternative<MetalCompileOptions>(options))
        return VERNON_TARGET_METAL;
    if (std::holds_alternative<DirectXCompileOptions>(options))
        return VERNON_TARGET_DIRECTX;
    return VERNON_TARGET_CUDA;
}

VernonStatus parseCompileOptions(const VernonCompileOptions &source, CompileOptions &options,
                                 std::string &diagnostics) {
    if (source.struct_size < sizeof(VernonCompileOptions)) {
        diagnostics = "compile options structure is too small";
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    if (source.target < VERNON_TARGET_CPU || source.target > VERNON_TARGET_CUDA) {
        diagnostics = "unknown compilation target";
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    if (source.target == VERNON_TARGET_CPU) {
        CpuCodegenOptions cpu;
        auto readCpuOption = [&](VernonStringView value, std::string &destination) {
            if (value.size != 0 && !value.data) {
                diagnostics = "CPU compile option has null data";
                return false;
            }
            destination = copyStringView(value);
            return true;
        };
        if (!readCpuOption(source.as.cpu.triple, cpu.targetTriple) ||
            !readCpuOption(source.as.cpu.processor, cpu.cpu) || !readCpuOption(source.as.cpu.features, cpu.features))
            return VERNON_STATUS_INVALID_ARGUMENT;
        options = std::move(cpu);
    } else if (source.target == VERNON_TARGET_OPENGL || source.target == VERNON_TARGET_OPENGL_ES) {
        const uint32_t version = source.as.opengl.version;
        if (version != 0 && (version < 100 || version > 999)) {
            diagnostics = "OpenGL version must be a three-digit GLSL version number";
            return VERNON_STATUS_INVALID_ARGUMENT;
        }
        options = OpenGLCompileOptions{source.target, version};
    } else if (source.target == VERNON_TARGET_METAL) {
        if (source.as.metal.platform != VERNON_METAL_PLATFORM_MACOS &&
            source.as.metal.platform != VERNON_METAL_PLATFORM_IOS) {
            diagnostics = "Metal platform must be macOS or iOS";
            return VERNON_STATUS_INVALID_ARGUMENT;
        }
        options = MetalCompileOptions{source.as.metal.platform};
    } else if (source.target == VERNON_TARGET_DIRECTX) {
        const uint32_t shaderModel = source.as.directx.shader_model == 0 ? 60 : source.as.directx.shader_model;
        if (shaderModel < 60) {
            diagnostics = "DirectX runtime artifacts require HLSL Shader Model 6.0 or newer";
            return VERNON_STATUS_INVALID_ARGUMENT;
        }
        options = DirectXCompileOptions{shaderModel};
    } else {
        options = defaultCompileOptions(source.target);
    }
    return VERNON_STATUS_OK;
}

VernonStatus compileTarget(PreparedModule &module, const CompileOptions &options, std::vector<Artifact> &artifacts,
                           std::string &reflection, std::string &diagnostics, CpuExecutionStatePtr &cpuExecution) {
    const VernonTarget target = compileTargetKind(options);
    diagnostics.clear();
    if (!validateTargetCapabilities(module, target, diagnostics)) {
        artifacts.clear();
        return VERNON_STATUS_UNSUPPORTED_TARGET;
    }
    if (target == VERNON_TARGET_CPU) {
        const auto &cpu = std::get<CpuCodegenOptions>(options);
        if (!compileCpu(module, cpu, artifacts, reflection, diagnostics, cpuExecution)) {
            artifacts.clear();
            return VERNON_STATUS_INTERNAL_ERROR;
        }
        if (!addArtifactTable(reflection, diagnostics, artifacts, options)) {
            artifacts.clear();
            return VERNON_STATUS_INTERNAL_ERROR;
        }
        return VERNON_STATUS_OK;
    }
    if (target == VERNON_TARGET_VULKAN || target == VERNON_TARGET_OPENGL || target == VERNON_TARGET_OPENGL_ES ||
        target == VERNON_TARGET_METAL || target == VERNON_TARGET_DIRECTX) {
        const uint32_t glslVersion = target == VERNON_TARGET_OPENGL || target == VERNON_TARGET_OPENGL_ES
                                         ? std::get<OpenGLCompileOptions>(options).version
                                         : 0;
        const uint32_t hlslShaderModel =
            target == VERNON_TARGET_DIRECTX ? std::get<DirectXCompileOptions>(options).shaderModel : 60;
        const VernonMetalPlatform metalPlatform = target == VERNON_TARGET_METAL
                                                      ? std::get<MetalCompileOptions>(options).platform
                                                      : VERNON_METAL_PLATFORM_MACOS;
        std::vector<TargetResourceSlot> targetResourceSlots;
        if (!compileSpirv(module, target, artifacts, diagnostics)) {
            artifacts.clear();
            return VERNON_STATUS_INTERNAL_ERROR;
        }
        if (target != VERNON_TARGET_VULKAN &&
            !crossCompileSpirv(artifacts, diagnostics, target, glslVersion, hlslShaderModel, metalPlatform,
                               target == VERNON_TARGET_METAL ? &targetResourceSlots : nullptr)) {
            artifacts.clear();
            return VERNON_STATUS_INTERNAL_ERROR;
        }
        if (target == VERNON_TARGET_DIRECTX) {
            std::vector<Artifact> dxilArtifacts;
            if (!compileHlslToDxil(artifacts, hlslShaderModel, dxilArtifacts, diagnostics)) {
                artifacts.clear();
                return VERNON_STATUS_INTERNAL_ERROR;
            }
            artifacts = std::move(dxilArtifacts);
        }
        if (!addArtifactTable(reflection, diagnostics, artifacts, options, targetResourceSlots)) {
            artifacts.clear();
            return VERNON_STATUS_INTERNAL_ERROR;
        }
        return VERNON_STATUS_OK;
    }
    if (target == VERNON_TARGET_CUDA) {
        if (!compileCuda(module, artifacts, diagnostics)) {
            artifacts.clear();
            return VERNON_STATUS_INTERNAL_ERROR;
        }
        if (!addArtifactTable(reflection, diagnostics, artifacts, options)) {
            artifacts.clear();
            return VERNON_STATUS_INTERNAL_ERROR;
        }
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
