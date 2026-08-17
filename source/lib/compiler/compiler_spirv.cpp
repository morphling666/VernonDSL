#include "compiler_spirv.h"

#include "compiler_dispatch.h"
#include "compiler_frontend.h"
#include "compiler_reflection.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Dialect/Vernon/Transforms/VernonConvertGPUToSPIRV.h"
#include "mlir/Dialect/Vernon/Transforms/VernonLowerAccumulation.h"
#include "mlir/Dialect/Vernon/Transforms/VernonLowerGPUTensors.h"
#include "mlir/Dialect/Vernon/Transforms/VernonLowerSynchronization.h"
#include "mlir/Dialect/Vernon/Transforms/VernonSpirvMarkers.h"
#include "mlir/Dialect/Vernon/Transforms/VernonToGPU.h"
#include "mlir/Dialect/Vernon/Transforms/VernonToSpirv.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/SPIRV/Serialization.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vernon::compiler_detail {
namespace {

constexpr uint32_t kSpirvMagic = 0x07230203u;
constexpr uint16_t kOpConstant = 43;
constexpr uint16_t kOpConstantComposite = 44;
constexpr uint16_t kOpImage = 100;
constexpr uint16_t kOpImageQuerySizeLod = 103;
constexpr uint16_t kOpIAdd = 128;

uint16_t wordCount(uint32_t instruction) { return static_cast<uint16_t>(instruction >> 16); }

uint16_t opcode(uint32_t instruction) { return static_cast<uint16_t>(instruction); }

bool allConstituentsEqual(llvm::ArrayRef<uint32_t> words, size_t offset,
                          const llvm::DenseMap<uint32_t, uint32_t> &constantValues, uint32_t expected) {
    if (wordCount(words[offset]) < 4)
        return false;
    for (size_t index = offset + 3; index < offset + wordCount(words[offset]); ++index) {
        const auto value = constantValues.find(words[index]);
        if (value == constantValues.end() || value->second != expected)
            return false;
    }
    return true;
}

} // namespace

bool materializeImageQuerySizeLod(llvm::SmallVectorImpl<uint32_t> &words, size_t expectedReplacementCount,
                                  std::string &diagnostics) {
    if (words.size() < 5 || words[0] != kSpirvMagic) {
        diagnostics = "serialized SPIR-V has an invalid header";
        return false;
    }
    const uint32_t idBound = words[3];
    const auto validId = [idBound](uint32_t id) { return id != 0 && id < idBound; };

    llvm::SmallVector<size_t> instructions;
    llvm::DenseMap<uint32_t, uint32_t> constantValues;
    for (size_t cursor = 5; cursor < words.size();) {
        const uint16_t count = wordCount(words[cursor]);
        if (count == 0 || cursor + count > words.size()) {
            diagnostics = "serialized SPIR-V instruction stream is malformed";
            return false;
        }
        instructions.push_back(cursor);
        if (opcode(words[cursor]) == kOpConstant && count == 4) {
            if (!validId(words[cursor + 2])) {
                diagnostics = "texture-size marker constant has an invalid result ID";
                return false;
            }
            constantValues[words[cursor + 2]] = words[cursor + 3];
        }
        cursor += count;
    }

    llvm::DenseSet<uint32_t> resultMarkersA;
    llvm::DenseSet<uint32_t> resultMarkersB;
    for (const size_t offset : instructions) {
        if (opcode(words[offset]) != kOpConstantComposite)
            continue;
        const uint32_t resultId = words[offset + 2];
        if (!validId(resultId)) {
            diagnostics = "texture-size composite marker has an invalid result ID";
            return false;
        }
        if (allConstituentsEqual(words, offset, constantValues, mlir::vernon::kImageQueryResultMarkerA))
            resultMarkersA.insert(resultId);
        if (allConstituentsEqual(words, offset, constantValues, mlir::vernon::kImageQueryResultMarkerB))
            resultMarkersB.insert(resultId);
    }

    llvm::DenseSet<size_t> scalarMarkerInstructions;
    llvm::SmallVector<size_t> resultMarkerInstructions;
    for (const size_t offset : instructions) {
        if (opcode(words[offset]) != kOpIAdd || wordCount(words[offset]) != 5)
            continue;
        const auto left = constantValues.find(words[offset + 3]);
        const auto right = constantValues.find(words[offset + 4]);
        if ((left != constantValues.end() && left->second == mlir::vernon::kImageQueryLodMarker) ||
            (right != constantValues.end() && right->second == mlir::vernon::kImageQueryLodMarker))
            scalarMarkerInstructions.insert(offset);

        const bool hasA = resultMarkersA.contains(words[offset + 3]) || resultMarkersA.contains(words[offset + 4]);
        const bool hasB = resultMarkersB.contains(words[offset + 3]) || resultMarkersB.contains(words[offset + 4]);
        if (hasA && hasB)
            resultMarkerInstructions.push_back(offset);
    }

    llvm::DenseSet<size_t> consumedScalarMarkers;
    size_t replacementCount = 0;
    for (const size_t resultOffset : resultMarkerInstructions) {
        const auto position = std::find(instructions.begin(), instructions.end(), resultOffset);
        if (position == instructions.end() || std::distance(instructions.begin(), position) < 2) {
            diagnostics = "texture-size result marker has no image/lod prefix";
            return false;
        }
        const size_t lodOffset = *(position - 1);
        const size_t imageOffset = *(position - 2);
        if (!scalarMarkerInstructions.contains(lodOffset) || opcode(words[imageOffset]) != kOpImage ||
            wordCount(words[imageOffset]) != 4) {
            diagnostics = "texture-size marker sequence is malformed";
            return false;
        }

        const uint32_t imageId = words[imageOffset + 2];
        const auto left = constantValues.find(words[lodOffset + 3]);
        const uint32_t lodId = left != constantValues.end() && left->second == mlir::vernon::kImageQueryLodMarker
                                   ? words[lodOffset + 4]
                                   : words[lodOffset + 3];
        const uint32_t lodResultId = words[lodOffset + 2];
        const uint32_t resultId = words[resultOffset + 2];
        if (!validId(words[imageOffset + 1]) || !validId(imageId) || !validId(words[imageOffset + 3]) ||
            !validId(words[lodOffset + 1]) || !validId(lodResultId) || !validId(lodId) ||
            !validId(words[resultOffset + 1]) || !validId(resultId) || imageId == lodResultId || imageId == resultId ||
            lodResultId == resultId) {
            diagnostics = "texture-size marker contains invalid image, lod, or "
                          "result IDs";
            return false;
        }

        words[resultOffset] = (static_cast<uint32_t>(5) << 16) | kOpImageQuerySizeLod;
        words[resultOffset + 3] = imageId;
        words[resultOffset + 4] = lodId;
        consumedScalarMarkers.insert(lodOffset);
        ++replacementCount;
    }

    if (replacementCount != expectedReplacementCount || replacementCount != resultMarkerInstructions.size() ||
        consumedScalarMarkers.size() != scalarMarkerInstructions.size()) {
        diagnostics = "texture-size marker replacement count mismatch";
        return false;
    }
    return true;
}

} // namespace vernon::compiler_detail

namespace vernon::compiler {
namespace {

mlir::spirv::TargetEnvAttr targetEnv(mlir::MLIRContext *context, bool requireF32AtomicAdd) {
    llvm::SmallVector<mlir::spirv::Capability> capabilities{mlir::spirv::Capability::Shader};
    llvm::SmallVector<mlir::spirv::Extension> extensions;
    if (requireF32AtomicAdd) {
        capabilities.push_back(mlir::spirv::Capability::AtomicFloat32AddEXT);
        extensions.push_back(mlir::spirv::Extension::SPV_EXT_shader_atomic_float_add);
    }
    auto triple = mlir::spirv::VerCapExtAttr::get(mlir::spirv::Version::V_1_3, capabilities, extensions, context);
    return mlir::spirv::TargetEnvAttr::get(
        triple, mlir::spirv::getDefaultResourceLimits(context), mlir::spirv::ClientAPI::Vulkan,
        mlir::spirv::Vendor::Unknown, mlir::spirv::DeviceType::Unknown, mlir::spirv::TargetEnvAttr::kUnknownDeviceID);
}

struct AttachGpuSpirvTargetPass
    : public mlir::PassWrapper<AttachGpuSpirvTargetPass, mlir::OperationPass<mlir::gpu::GPUModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AttachGpuSpirvTargetPass)

    explicit AttachGpuSpirvTargetPass(bool requireF32AtomicAdd) : requireF32AtomicAdd(requireF32AtomicAdd) {}
    AttachGpuSpirvTargetPass(const AttachGpuSpirvTargetPass &other)
        : PassWrapper(other), requireF32AtomicAdd(other.requireF32AtomicAdd) {}

    void runOnOperation() override {
        getOperation()->setAttr(mlir::spirv::getTargetEnvAttrName(), targetEnv(&getContext(), requireF32AtomicAdd));
    }

    bool requireF32AtomicAdd{};
};

struct AttachSpirvTargetPass
    : public mlir::PassWrapper<AttachSpirvTargetPass, mlir::OperationPass<mlir::spirv::ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AttachSpirvTargetPass)

    explicit AttachSpirvTargetPass(bool requireF32AtomicAdd) : requireF32AtomicAdd(requireF32AtomicAdd) {}
    AttachSpirvTargetPass(const AttachSpirvTargetPass &other)
        : PassWrapper(other), requireF32AtomicAdd(other.requireF32AtomicAdd) {}

    void runOnOperation() override {
        mlir::spirv::ModuleOp module = getOperation();
        module->setAttr(mlir::spirv::getTargetEnvAttrName(), targetEnv(module.getContext(), requireF32AtomicAdd));
    }

    bool requireF32AtomicAdd{};
};

} // namespace

bool compileSpirv(PreparedModule &prepared, const TargetProfile &profile, std::vector<Artifact> &artifacts,
                  std::string &reflection, std::string &diagnostics) {
    mlir::MLIRContext &context = prepared.context();
    const VernonTarget target = profile.target;
    mlir::ScopedDiagnosticHandler handler(
        &context, [&](mlir::Diagnostic &diagnostic) { appendDiagnostic(diagnostics, diagnostic); });
    mlir::FailureOr<TargetPreparationResult> preparedTarget =
        prepareTargetModule(prepared, preparePortableTargetModule);
    if (mlir::failed(preparedTarget))
        return false;
    mlir::OwningOpRef<mlir::ModuleOp> module = std::move(preparedTarget->module);
    {
        mlir::PassManager passManager(&context);
        passManager.addPass(mlir::vernon::createVernonLowerAccumulationPass(profile.accumulation));
        passManager.addPass(mlir::vernon::createVernonVerifyGeneratedAccumulationPass(profile.accumulation));
        if (mlir::failed(passManager.run(*module)))
            return false;
    }
    mlir::FailureOr<std::string> targetReflection =
        buildReflection(*module, prepared.logicalReflection(), preparedTarget->entries, preparedTarget->provenance);
    if (mlir::failed(targetReflection))
        return false;
    reflection = std::move(*targetReflection);
    bool requiresRuntimeContractViolation = false;
    module->walk([&](mlir::cf::AssertOp) { requiresRuntimeContractViolation = true; });
    if (requiresRuntimeContractViolation) {
        diagnostics = "SPIR-V targets do not support dynamic range steps because they cannot report the required "
                      "runtime contract violation for step=0";
        return false;
    }
    if (moduleUsesF16(module.get())) {
        diagnostics = "GPU targets do not currently support f16";
        return false;
    }

    {
        mlir::PassManager passManager(&context);
        passManager.addNestedPass<mlir::func::FuncOp>(mlir::createForToWhileLoopPass());
        passManager.addPass(mlir::vernon::createVernonToGPUPass(true, true));
        if (mlir::failed(passManager.run(*module)))
            return false;
    }
    bool requireF32AtomicAdd = false;
    module->walk([&](mlir::Operation *operation) {
        auto implementation = operation->getAttrOfType<mlir::StringAttr>(mlir::vernon::kAtomicImplementationAttrName);
        requireF32AtomicAdd |= implementation && implementation.getValue() == mlir::vernon::kNativeAtomicImplementation;
    });
    {
        mlir::PassManager passManager(&context);
        passManager.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::vernon::createVernonLowerGPUSynchronizationPass(true));
        passManager.addNestedPass<mlir::gpu::GPUModuleOp>(
            std::make_unique<AttachGpuSpirvTargetPass>(requireF32AtomicAdd));
        passManager.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::vernon::createVernonLowerGPUTensorsPass(true));
        passManager.addPass(mlir::vernon::createVernonConvertGPUToSPIRVPass());
        passManager.addPass(mlir::vernon::createVernonToSPIRVPass(target == VERNON_TARGET_VULKAN));
        passManager.addNestedPass<mlir::spirv::ModuleOp>(mlir::spirv::createSPIRVLowerABIAttributesPass());
        passManager.addNestedPass<mlir::spirv::ModuleOp>(std::make_unique<AttachSpirvTargetPass>(requireF32AtomicAdd));
        passManager.addNestedPass<mlir::spirv::ModuleOp>(mlir::spirv::createSPIRVUpdateVCEPass());
        if (mlir::failed(passManager.run(*module)))
            return false;
    }

    llvm::SmallVector<mlir::spirv::ModuleOp> spirvModules;
    module->walk([&](mlir::spirv::ModuleOp spirvModule) { spirvModules.push_back(spirvModule); });
    if (spirvModules.empty()) {
        diagnostics = "module has no graphics or compute entry points for Vulkan";
        return false;
    }

    artifacts.clear();
    for (auto [index, spirvModule] : llvm::enumerate(spirvModules)) {
        size_t expectedImageQueryCount = 0;
        if (auto count = spirvModule->getAttrOfType<mlir::IntegerAttr>(mlir::vernon::kImageQueryExpectedCountAttr)) {
            if (count.getInt() < 0) {
                diagnostics = "texture-size query count cannot be negative";
                return false;
            }
            expectedImageQueryCount = static_cast<size_t>(count.getInt());
            spirvModule->removeAttr(mlir::vernon::kImageQueryExpectedCountAttr);
        }
        llvm::SmallVector<uint32_t> words;
        if (mlir::failed(mlir::spirv::serialize(spirvModule, words)))
            return false;
        if (!vernon::compiler_detail::materializeImageQuerySizeLod(words, expectedImageQueryCount, diagnostics))
            return false;
        std::string binary(reinterpret_cast<const char *>(words.data()), words.size() * sizeof(uint32_t));
        std::string name = spirvModules.size() == 1 ? "module.spv" : "module_" + std::to_string(index) + ".spv";
        artifacts.push_back(Artifact{std::move(name), std::move(binary)});
    }
    return true;
}

} // namespace vernon::compiler
