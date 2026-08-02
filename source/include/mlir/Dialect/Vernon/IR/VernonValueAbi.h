#pragma once

#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>

namespace mlir::vernon {

struct ResolvedStructField {
    std::string name;
    Type type;
};

struct ResolvedStructFields {
    StructDeclOp declaration;
    SmallVector<ResolvedStructField> fields;
};

FailureOr<ResolvedStructFields> resolveNamedStructFields(StructType structure, ModuleOp module);

FailureOr<std::pair<StructDeclOp, SmallVector<Type>>> resolveStructFields(StructType structure, ModuleOp module);

struct ValueAbiPathComponent {
    std::optional<std::string> field;
    uint64_t index{};

    static ValueAbiPathComponent getField(StringRef name);
    static ValueAbiPathComponent getIndex(uint64_t index);
};

struct ValueAbiLeaf {
    SmallVector<ValueAbiPathComponent> path;
    SmallVector<uint64_t> shape;
    Type scalarType;
    std::string dtype;
    uint64_t byteOffset{};
    uint64_t scalarCount{};
};

struct ValueAbiLayout {
    uint64_t size{};
    uint64_t alignment{};
    SmallVector<uint64_t> fieldOffsets;
    std::optional<uint64_t> elementStride;
    SmallVector<ValueAbiLeaf> leaves;
    std::string layoutHash;
};

enum class PhysicalAbiProfile {
    HostValue,
    CudaKernelParameter,
    VulkanStd140UniformBuffer,
    VulkanStd430StorageBuffer,
    VulkanPushConstant,
    OpenGLNativeUniform,
    DirectXConstantBuffer,
    MetalConstantBuffer,
    Count,
};

struct PhysicalValueAbiLayout {
    uint64_t size{};
    uint64_t alignment{};
    SmallVector<uint64_t> byteStrides;
    SmallVector<uint64_t> elementLeafOffsets;
};

enum class PhysicalResourceAbiKind {
    HostPointer,
    TensorViewDescriptor,
    CudaStorageLeaves,
    GraphicsStorageLeaves,
    GraphicsTexture,
    GraphicsSampler,
};

struct PhysicalResourceAbiLayout {
    PhysicalResourceAbiKind kind;
    uint64_t handleSize{};
    uint64_t handleAlignment{};
    std::optional<ValueAbiLayout> elementLayout;
};

struct UnsupportedPhysicalValueAbi {
    std::string reason;
};

using PhysicalValueAbiPlan =
    std::variant<PhysicalValueAbiLayout, PhysicalResourceAbiLayout, UnsupportedPhysicalValueAbi>;

FailureOr<ValueAbiLayout> getValueAbiLayout(Type type, ModuleOp module, ArrayRef<StringRef> logicalLeafDtypes = {});

/// Validate a logical Value with the same canonical planner used by reflection
/// and every physical ABI lowering.  This is intentionally available to IR
/// verifiers so malformed textual IR cannot bypass frontend validation.
LogicalResult verifyValueAbiType(Type type, ModuleOp module);

FailureOr<PhysicalValueAbiPlan> getPhysicalValueAbiPlan(Type type, ModuleOp module, PhysicalAbiProfile profile);

FailureOr<PhysicalValueAbiLayout> getPhysicalValueAbiLayout(Type type, ModuleOp module, PhysicalAbiProfile profile);

constexpr uint64_t kPortableWorkgroupStorageLimit = 16 * 1024;

struct WorkgroupPhysicalLeaf {
    Type scalarType;
    uint64_t scalarCount{};
    uint64_t byteSize{};
    size_t layoutLeafIndex{};
};

struct WorkgroupPhysicalStoragePlan {
    Type elementType;
    ValueAbiLayout layout;
    uint64_t recordCount{};
    uint64_t totalPhysicalBytes{};
    SmallVector<WorkgroupPhysicalLeaf> leaves;
};

FailureOr<WorkgroupPhysicalStoragePlan> getWorkgroupPhysicalStoragePlan(TensorViewType view, ModuleOp module);

} // namespace mlir::vernon
