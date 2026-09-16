#pragma once

#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace mlir::vernon {

enum class MetadataFieldKind {
    Offset,
    Extent,
    Stride,
};

struct MetadataFieldIdentity {
    unsigned argumentIndex{};
    MetadataFieldKind kind{};
    std::optional<unsigned> dimension;

    bool operator==(const MetadataFieldIdentity &other) const {
        return argumentIndex == other.argumentIndex && kind == other.kind && dimension == other.dimension;
    }
};

struct SemanticMetadataField {
    uint32_t ordinal{};
    MetadataFieldIdentity identity;
};

struct SemanticMetadataView {
    unsigned argumentIndex{};
    uint32_t firstFieldOrdinal{};
};

/// Immutable, entry-scoped TensorView metadata semantics. All values are
/// signed logical-element quantities and projection order is dimension order.
class SemanticMetadataPlan final {
public:
    ArrayRef<SemanticMetadataField> getFields() const { return fields; }
    ArrayRef<SemanticMetadataView> getViews() const { return views; }
    bool empty() const { return fields.empty(); }
    uint32_t getFieldCount() const { return static_cast<uint32_t>(fields.size()); }

private:
    friend FailureOr<std::shared_ptr<const SemanticMetadataPlan>> getSemanticMetadataPlan(FunctionOpInterface function);

    SmallVector<SemanticMetadataField> fields;
    SmallVector<SemanticMetadataView> views;
};

enum class MetadataPhysicalProfile {
    PortableShaderMetadataI32,
    CudaKernelMetadata,
    HostMetadata,
};

enum class MetadataCarrierKind {
    ConstantRegion,
    KernelParameter,
    CpuCallFrame,
};

struct PhysicalMetadataMember {
    uint32_t semanticOrdinal{};
    uint64_t byteOffset{};
    uint64_t size{};
    uint64_t alignment{};
};

struct MetadataNativeLocation {
    std::optional<uint32_t> descriptorSet;
    std::optional<uint32_t> binding;
    std::optional<uint32_t> kernelParameterOrdinal;
};

/// Immutable physical projection of one SemanticMetadataPlan.
class PhysicalMetadataPlan final {
public:
    MetadataCarrierKind getCarrierKind() const { return carrierKind; }
    StringRef getScalarRepresentation() const { return scalarRepresentation; }
    ArrayRef<PhysicalMetadataMember> getMembers() const { return members; }
    uint64_t getEncodedSize() const { return encodedSize; }
    uint64_t getBlockSize() const { return blockSize; }
    uint64_t getAbiAlignment() const { return abiAlignment; }
    uint64_t getCompiledSizeCeiling() const { return compiledSizeCeiling; }
    const MetadataNativeLocation &getNativeLocation() const { return nativeLocation; }
    StringRef getCanonicalLayoutHash() const { return canonicalLayoutHash; }

private:
    friend FailureOr<std::shared_ptr<const PhysicalMetadataPlan>>
    getPhysicalMetadataPlan(const SemanticMetadataPlan &semantic, MetadataPhysicalProfile profile,
                            MetadataNativeLocation nativeLocation, uint32_t hostIndexBitWidth);

    MetadataCarrierKind carrierKind{};
    StringRef scalarRepresentation;
    SmallVector<PhysicalMetadataMember> members;
    uint64_t encodedSize{};
    uint64_t blockSize{};
    uint64_t abiAlignment{};
    uint64_t compiledSizeCeiling{};
    MetadataNativeLocation nativeLocation;
    std::string canonicalLayoutHash;
};

FailureOr<std::shared_ptr<const SemanticMetadataPlan>> getSemanticMetadataPlan(FunctionOpInterface function);
FailureOr<uint32_t> getCudaMetadataKernelParameterOrdinal(FunctionOpInterface function, unsigned metadataArgumentIndex);

FailureOr<std::shared_ptr<const PhysicalMetadataPlan>>
getPhysicalMetadataPlan(const SemanticMetadataPlan &semantic, MetadataPhysicalProfile profile,
                        MetadataNativeLocation nativeLocation = {}, uint32_t hostIndexBitWidth = 0);

StringRef stringifyMetadataFieldKind(MetadataFieldKind kind);
StringRef stringifyMetadataPhysicalProfile(MetadataPhysicalProfile profile);
StringRef stringifyMetadataCarrierKind(MetadataCarrierKind kind);

constexpr uint64_t kPortableShaderMetadataSizeCeiling = 16 * 1024;

} // namespace mlir::vernon
