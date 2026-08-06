#pragma once

#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>

namespace mlir::vernon {

/// True when a boundary type contains logical autodiff handles that must be
/// materialized by a target lowering before a physical Value ABI is planned.
bool containsLogicalAutodiffHandle(Type type);

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

enum class CanonicalAbiNodeKind {
    Scalar,
    Product,
    Array,
};

/// A node in the canonical, recursive in-memory Value ABI. Offsets are
/// relative to the containing node; logical paths live only in the derived
/// leaf view because an Array child represents every logical element.
struct CanonicalAbiNode {
    CanonicalAbiNodeKind kind;
    Type type;
    std::string representation;
    SmallVector<uint64_t> shape;
    uint64_t size{};
    uint64_t alignment{};
    SmallVector<uint64_t> childOffsets;
    std::optional<uint64_t> elementStride;
    SmallVector<std::shared_ptr<const CanonicalAbiNode>> children;
};

struct CanonicalAbiTree {
    std::shared_ptr<const CanonicalAbiNode> root;
};

/// A derived flat view of CanonicalAbiTree for scalar-oriented lowering.
struct ValueAbiLayout {
    uint64_t size{};
    uint64_t alignment{};
    SmallVector<uint64_t> fieldOffsets;
    std::optional<uint64_t> elementStride;
    SmallVector<ValueAbiLeaf> leaves;
    std::string layoutHash;
    CanonicalAbiTree tree;
};

struct CpuCallLane {
    size_t leafIndex{};
    uint64_t scalarIndex{};
};

struct ByteTransportNode;

struct CpuCallPlan {
    ValueAbiLayout layout;
    SmallVector<CpuCallLane> lanes;
    std::shared_ptr<const ByteTransportNode> root;
};

FailureOr<CpuCallPlan> getCpuCallPlan(Type type, ModuleOp module, ArrayRef<StringRef> logicalLeafDtypes = {});

bool isCpuOpaqueAbiType(Type type);

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

enum class ByteTransportNodeKind {
    Scalar,
    Product,
    Array,
};

struct ByteTransportNode {
    ByteTransportNodeKind kind;
    std::string representation;
    uint64_t byteOffset{};
    uint64_t size{};
    uint64_t alignment{};
    SmallVector<uint64_t> shape;
    SmallVector<uint64_t> byteStrides;
    SmallVector<std::shared_ptr<const ByteTransportNode>> children;
};

struct ByteTransportPlan {
    PhysicalAbiProfile profile;
    std::string canonicalLayoutHash;
    std::shared_ptr<const ByteTransportNode> root;
};

enum class PhysicalResourceAbiKind {
    HostPointer,
    TensorViewDescriptor,
    CudaStorageLeaves,
    GraphicsStorageLeaves,
    GraphicsTexture,
    GraphicsSampler,
};

struct ResourceBindingPlan {
    PhysicalResourceAbiKind kind;
    uint64_t handleSize{};
    uint64_t handleAlignment{};
};

struct NativeUniformPlan {
    std::string canonicalLayoutHash;
    std::shared_ptr<const ByteTransportNode> root;
};

struct KernelParameterPlan {
    std::string canonicalLayoutHash;
    std::shared_ptr<const ByteTransportNode> root;
};

struct UnsupportedBackendInterfaceAbi {
    std::string reason;
};

using BackendInterfaceAbiPlan = std::variant<ByteTransportPlan, NativeUniformPlan, KernelParameterPlan,
                                             ResourceBindingPlan, UnsupportedBackendInterfaceAbi>;

FailureOr<ValueAbiLayout> getValueAbiLayout(Type type, ModuleOp module, ArrayRef<StringRef> logicalLeafDtypes = {});

/// Validate a logical Value with the same canonical planner used by reflection
/// and every physical ABI lowering.  This is intentionally available to IR
/// verifiers so malformed textual IR cannot bypass frontend validation.
LogicalResult verifyValueAbiType(Type type, ModuleOp module);

FailureOr<BackendInterfaceAbiPlan> getBackendInterfaceAbiPlan(Type type, ModuleOp module, PhysicalAbiProfile profile,
                                                              ArrayRef<StringRef> logicalLeafDtypes = {});

FailureOr<ByteTransportPlan> getByteTransportPlan(Type type, ModuleOp module, PhysicalAbiProfile profile);

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
