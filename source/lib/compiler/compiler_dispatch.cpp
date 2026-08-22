#include "compiler_dispatch.h"

#include "compiler_artifacts.h"
#include "compiler_cpu.h"
#include "compiler_cuda.h"
#include "compiler_dxc.h"
#include "compiler_frontend.h"
#include "compiler_reflection.h"
#include "compiler_spirv.h"
#include "compiler_spirv_cross.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/IR/VernonValueAbi.h"
#include "llvm/ADT/STLExtras.h"

#include <cstring>
#include <string>

namespace vernon::compiler {

namespace {

std::string copyStringView(VernonStringView value) {
    return value.data && value.size ? std::string(value.data, value.size) : std::string();
}

bool containsBackendLocalTapeHandle(mlir::Type type) {
    if (mlir::isa<mlir::vernon::AdTapeType, mlir::vernon::AdRegionHeaderType>(type))
        return true;
    if (auto tuple = mlir::dyn_cast<mlir::TupleType>(type))
        return llvm::any_of(tuple.getTypes(), containsBackendLocalTapeHandle);
    return false;
}

VernonTargetCapabilities queryTargetCapabilities(VernonTarget target) {
#if !defined(VERNON_DXC_EXECUTABLE)
    if (target == VERNON_TARGET_DIRECTX)
        return {};
#endif
    switch (target) {
    case VERNON_TARGET_CPU:
        return {1, 1, 1, 1, 1};
    case VERNON_TARGET_VULKAN:
        return {1, 1, 1, 1, 0};
    case VERNON_TARGET_CUDA:
        return {1, 0, 1, 1, 1};
    case VERNON_TARGET_OPENGL:
    case VERNON_TARGET_OPENGL_ES:
    case VERNON_TARGET_METAL:
    case VERNON_TARGET_DIRECTX:
        return {1, 1, 1, 1, 0};
    }
    return {};
}

bool validateTargetCapabilities(PreparedModule &prepared, const TargetProfile &profile, std::string &diagnostics) {
    mlir::ModuleOp module = prepared.logicalModule();
    const VernonTarget target = profile.target;
    const VernonTargetCapabilities capabilities = queryTargetCapabilities(target);
    bool usesDeviceAtomics = false;
    bool usesFloatDeviceAtomics = false;
    bool cpuSynchronizationUnsupported = false;
    bool cudaDeviceBarrier = false;
    module.walk([&](mlir::Operation *operation) {
        mlir::Value atomicStorage;
        if (auto atomic = mlir::dyn_cast<mlir::vernon::AtomicOp>(operation))
            atomicStorage = atomic.getStorage();
        else if (auto atomic = mlir::dyn_cast<mlir::vernon::PhysicalAtomicOp>(operation))
            atomicStorage = atomic.getStorage();
        if (atomicStorage) {
            auto view = mlir::dyn_cast<mlir::vernon::TensorViewType>(atomicStorage.getType());
            usesDeviceAtomics |= view && view.getAddressSpace() == "device";
            usesFloatDeviceAtomics |= view && view.getAddressSpace() == "device" && view.getElementType().isF32();
            if (!view)
                cpuSynchronizationUnsupported = true;
            if (view && view.getAddressSpace() == "device")
                return;
        }
        if (auto barrier = mlir::dyn_cast<mlir::vernon::BarrierOp>(operation);
            barrier && barrier.getScope() == "device") {
            cudaDeviceBarrier = true;
            cpuSynchronizationUnsupported = true;
        }
    });
    if (usesFloatDeviceAtomics &&
        profile.accumulation.device.f32 == mlir::vernon::AtomicAddImplementation::Unsupported) {
        diagnostics = "device-scope f32 atomic add has no legal implementation in the target profile";
        return false;
    }
    if (usesDeviceAtomics && !capabilities.supports_device_storage_atomics) {
        diagnostics = "device-scope storage TensorView atomics require a target storage-atomic capability";
        return false;
    }
    if (target == VERNON_TARGET_CPU && cpuSynchronizationUnsupported) {
        diagnostics = "CPU target does not support cross-workgroup device barriers; split the work into multiple "
                      "kernel launches";
        return false;
    }
    if (target == VERNON_TARGET_CUDA) {
        if (cudaDeviceBarrier) {
            diagnostics = "CUDA target capability rejects device-scope barriers; use workgroup_barrier";
            return false;
        }
        for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>()) {
            if (!function->hasAttr("vernon.entry"))
                continue;
            auto stage = function->getAttrOfType<mlir::StringAttr>("vernon.stage");
            if (!stage || stage.getValue() != "compute") {
                diagnostics = "CUDA target capability rejects non-compute entry '" + function.getSymName().str() + "'";
                return false;
            }
            for (unsigned index = 0; index < function.getNumArguments(); ++index) {
                if (function.getArgAttrDict(index).get("vernon.builtin") ||
                    containsBackendLocalTapeHandle(function.getArgumentTypes()[index]))
                    continue;
                mlir::DictionaryAttr attrs = function.getArgAttrDict(index);
                llvm::SmallVector<llvm::StringRef> logicalDtypes;
                if (auto dtypes = attrs.getAs<mlir::ArrayAttr>("vernon.abi_leaf_dtypes"))
                    for (mlir::Attribute dtype : dtypes)
                        if (auto value = mlir::dyn_cast<mlir::StringAttr>(dtype))
                            logicalDtypes.push_back(value.getValue());
                if (logicalDtypes.empty())
                    if (auto sugar = attrs.getAs<mlir::StringAttr>("vernon.dtype"); sugar && !sugar.getValue().empty())
                        logicalDtypes.push_back(sugar.getValue());
                mlir::FailureOr<mlir::vernon::BackendInterfaceAbiPlan> plan = mlir::vernon::getBackendInterfaceAbiPlan(
                    function.getArgumentTypes()[index], module, mlir::vernon::PhysicalAbiProfile::CudaKernelParameter,
                    logicalDtypes);
                if (mlir::failed(plan)) {
                    diagnostics = "CUDA target capability cannot plan compute argument #" + std::to_string(index);
                    return false;
                }
                if (const auto *unsupported = std::get_if<mlir::vernon::UnsupportedBackendInterfaceAbi>(&*plan)) {
                    diagnostics = "CUDA target capability '" + unsupported->reason + "' rejects compute argument #" +
                                  std::to_string(index);
                    return false;
                }
            }
        }
    }
    return true;
}

} // namespace

VernonTargetCapabilities targetCapabilities(VernonTarget target) { return queryTargetCapabilities(target); }

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

TargetProfile resolveTargetProfile(const CompileOptions &options) {
    using mlir::vernon::AggregateGradientStorage;
    using mlir::vernon::AtomicAddImplementation;
    TargetProfile profile;
    profile.target = compileTargetKind(options);
    profile.accumulation.aggregateGradientStorage = AggregateGradientStorage::InvocationPrivateStaging;
    if (std::holds_alternative<CpuCodegenOptions>(options)) {
        profile.accumulation.device = {AtomicAddImplementation::Native, AtomicAddImplementation::Native};
        profile.accumulation.workgroup = {AtomicAddImplementation::Native, AtomicAddImplementation::Native};
        return profile;
    }
    if (std::holds_alternative<CudaCompileOptions>(options)) {
        profile.accumulation.device = {AtomicAddImplementation::Native, AtomicAddImplementation::Unsupported};
        profile.accumulation.workgroup = {AtomicAddImplementation::Native, AtomicAddImplementation::Unsupported};
        profile.accumulation.supportsWorkgroupReduction = true;
        return profile;
    }
    profile.accumulation.device = {AtomicAddImplementation::IntegerCompareExchange,
                                   AtomicAddImplementation::Unsupported};
    profile.accumulation.workgroup = {AtomicAddImplementation::IntegerCompareExchange,
                                      AtomicAddImplementation::Unsupported};
    profile.accumulation.supportsWorkgroupReduction = true;
    return profile;
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
                           std::string &reflection, std::string &diagnostics,
                           const VernonCpuRuntimeHelpersV1 *cpuRuntimeHelpers, CpuExecutionStatePtr &cpuExecution) {
    const TargetProfile profile = resolveTargetProfile(options);
    const VernonTarget target = profile.target;
    diagnostics.clear();
    if (!validateTargetCapabilities(module, profile, diagnostics)) {
        artifacts.clear();
        return VERNON_STATUS_UNSUPPORTED_TARGET;
    }
    if (target == VERNON_TARGET_CPU) {
        const auto &cpu = std::get<CpuCodegenOptions>(options);
        const CpuCompileResult result =
            compileCpu(module, cpu, artifacts, reflection, diagnostics, cpuRuntimeHelpers, cpuExecution);
        if (result != CpuCompileResult::Success) {
            artifacts.clear();
            return result == CpuCompileResult::VerificationFailure ? VERNON_STATUS_VERIFICATION_ERROR
                                                                   : VERNON_STATUS_INTERNAL_ERROR;
        }
        if (!selectTargetPhysicalLayouts(reflection, target, diagnostics)) {
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
        if (!compileSpirv(module, profile, artifacts, reflection, diagnostics)) {
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
        if (!selectTargetPhysicalLayouts(reflection, target, diagnostics)) {
            artifacts.clear();
            return VERNON_STATUS_INTERNAL_ERROR;
        }
        if (!addArtifactTable(reflection, diagnostics, artifacts, options, targetResourceSlots)) {
            artifacts.clear();
            return VERNON_STATUS_INTERNAL_ERROR;
        }
        return VERNON_STATUS_OK;
    }
    if (target == VERNON_TARGET_CUDA) {
        if (!compileCuda(module, profile, artifacts, reflection, diagnostics)) {
            artifacts.clear();
            return VERNON_STATUS_INTERNAL_ERROR;
        }
        if (!selectTargetPhysicalLayouts(reflection, target, diagnostics)) {
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

VernonCpuEntryPoint findCompiledCpuEntry(const CpuExecutionState *execution, std::string_view entry) {
    return findCpuEntry(execution, entry);
}

} // namespace vernon::compiler
