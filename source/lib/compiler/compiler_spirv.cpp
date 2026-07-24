#include "compiler_spirv.h"

#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
#include "mlir/Conversion/MathToSPIRV/MathToSPIRVPass.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/Transforms/VernonInlineHelpers.h"
#include "mlir/Dialect/Vernon/Transforms/VernonLowerGPUTensors.h"
#include "mlir/Dialect/Vernon/Transforms/VernonSpirvMarkers.h"
#include "mlir/Dialect/Vernon/Transforms/VernonToGPU.h"
#include "mlir/Dialect/Vernon/Transforms/VernonToSpirv.h"
#include "mlir/Dialect/Vernon/Transforms/VernonValidation.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"
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

bool containsF16(mlir::Type type) {
    if (type.isF16())
        return true;
    if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(type))
        return containsF16(shaped.getElementType());
    if (auto view = mlir::dyn_cast<mlir::vernon::TensorViewType>(type))
        return containsF16(view.getElementType());
    if (auto function = mlir::dyn_cast<mlir::FunctionType>(type))
        return llvm::any_of(function.getInputs(), containsF16) || llvm::any_of(function.getResults(), containsF16);
    return false;
}

bool moduleUsesF16(mlir::ModuleOp module) {
    bool usesF16 = false;
    module.walk([&](mlir::Operation *operation) {
        usesF16 = usesF16 || llvm::any_of(operation->getOperandTypes(), containsF16) ||
                  llvm::any_of(operation->getResultTypes(), containsF16);
        for (mlir::Region &region : operation->getRegions())
            for (mlir::Block &block : region)
                usesF16 = usesF16 || llvm::any_of(block.getArgumentTypes(), containsF16);
        return usesF16 ? mlir::WalkResult::interrupt() : mlir::WalkResult::advance();
    });
    return usesF16;
}

struct AttachSpirvTargetPass
    : public mlir::PassWrapper<AttachSpirvTargetPass, mlir::OperationPass<mlir::spirv::ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AttachSpirvTargetPass)

    void runOnOperation() override {
        mlir::spirv::ModuleOp module = getOperation();
        auto triple = mlir::spirv::VerCapExtAttr::get(mlir::spirv::Version::V_1_3, {mlir::spirv::Capability::Shader},
                                                      llvm::ArrayRef<mlir::spirv::Extension>(), module.getContext());
        module->setAttr(mlir::spirv::getTargetEnvAttrName(),
                        mlir::spirv::TargetEnvAttr::get(
                            triple, mlir::spirv::getDefaultResourceLimits(module.getContext()),
                            mlir::spirv::ClientAPI::Vulkan, mlir::spirv::Vendor::Unknown,
                            mlir::spirv::DeviceType::Unknown, mlir::spirv::TargetEnvAttr::kUnknownDeviceID));
    }
};

} // namespace

bool compileSpirv(mlir::MLIRContext &context, const char *source, size_t sourceSize, VernonTarget target,
                  std::vector<Artifact> &artifacts, std::string &diagnostics) {
    mlir::ScopedDiagnosticHandler handler(
        &context, [&](mlir::Diagnostic &diagnostic) { appendDiagnostic(diagnostics, diagnostic); });
    llvm::StringRef text(source ? source : "", sourceSize);
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
    if (!module)
        return false;
    bool requiresRuntimeContractViolation = false;
    module->walk([&](mlir::cf::AssertOp) { requiresRuntimeContractViolation = true; });
    if (requiresRuntimeContractViolation) {
        diagnostics = "SPIR-V targets do not support dynamic range steps because they cannot report the required "
                      "runtime contract violation for step=0";
        return false;
    }
    if (target == VERNON_TARGET_VULKAN && moduleUsesF16(module.get())) {
        diagnostics = "Vulkan target does not support f16 until shaderFloat16 and 16-bit storage features are enabled";
        return false;
    }

    mlir::PassManager passManager(&context);
    passManager.addPass(mlir::vernon::createVernonValidatePass());
    // Backend lowerings intentionally only handle entry bodies. Inline shared
    // helpers while the module is still in common typed MLIR so every target
    // sees the same implementation.
    passManager.addPass(mlir::vernon::createVernonInlineHelpersPass());
    passManager.addPass(mlir::vernon::createVernonToGPUPass(true));
    passManager.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::vernon::createVernonLowerGPUTensorsPass(true));
    passManager.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createConvertMathToSPIRVPass());
    passManager.addPass(mlir::createConvertGPUToSPIRVPass());
    passManager.addPass(mlir::vernon::createVernonToSPIRVPass(target == VERNON_TARGET_VULKAN));
    passManager.addNestedPass<mlir::spirv::ModuleOp>(mlir::spirv::createSPIRVLowerABIAttributesPass());
    passManager.addNestedPass<mlir::spirv::ModuleOp>(std::make_unique<AttachSpirvTargetPass>());
    passManager.addNestedPass<mlir::spirv::ModuleOp>(mlir::spirv::createSPIRVUpdateVCEPass());
    if (mlir::failed(passManager.run(*module)))
        return false;

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
