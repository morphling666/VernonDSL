#include "mlir/Dialect/Vernon/IR/VernonMetadataAbi.h"

#include "mlir/Dialect/Vernon/IR/VernonAttrs.h"
#include "mlir/Dialect/Vernon/IR/VernonValueAbi.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>

namespace mlir::vernon {

StringRef stringifyMetadataFieldKind(MetadataFieldKind kind) {
    switch (kind) {
    case MetadataFieldKind::Offset:
        return "offset";
    case MetadataFieldKind::Extent:
        return "extent";
    case MetadataFieldKind::Stride:
        return "stride";
    }
    llvm_unreachable("unknown metadata field kind");
}

StringRef stringifyMetadataPhysicalProfile(MetadataPhysicalProfile profile) {
    switch (profile) {
    case MetadataPhysicalProfile::PortableShaderMetadataI32:
        return "portable_shader_metadata_i32";
    case MetadataPhysicalProfile::CudaKernelMetadata:
        return "cuda_kernel_metadata_i64";
    case MetadataPhysicalProfile::HostMetadata:
        return "host_metadata";
    }
    llvm_unreachable("unknown metadata physical profile");
}

StringRef stringifyMetadataCarrierKind(MetadataCarrierKind kind) {
    switch (kind) {
    case MetadataCarrierKind::ConstantRegion:
        return "constant_region";
    case MetadataCarrierKind::KernelParameter:
        return "kernel_parameter";
    case MetadataCarrierKind::CpuCallFrame:
        return "cpu_call_frame";
    }
    llvm_unreachable("unknown metadata carrier kind");
}

FailureOr<std::shared_ptr<const SemanticMetadataPlan>> getSemanticMetadataPlan(FunctionOpInterface function) {
    if (!function)
        return failure();
    auto module = function->getParentOfType<ModuleOp>();
    if (!module)
        return failure();

    auto plan = std::make_shared<SemanticMetadataPlan>();
    auto append = [&](unsigned argumentIndex, MetadataFieldKind kind,
                      std::optional<unsigned> dimension = std::nullopt) -> LogicalResult {
        if (plan->fields.size() >= std::numeric_limits<uint32_t>::max())
            return failure();
        const uint32_t ordinal = static_cast<uint32_t>(plan->fields.size());
        plan->fields.push_back({ordinal, {argumentIndex, kind, dimension}});
        return success();
    };

    for (unsigned argumentIndex = 0; argumentIndex < function.getNumArguments(); ++argumentIndex) {
        auto view = dyn_cast<TensorViewType>(function.getArgumentTypes()[argumentIndex]);
        if (!view || view.getAddressSpace() != "device")
            continue;
        FailureOr<ValueAbiLayout> storage = getValueStorageLayout(view.getElementType(), module);
        if (failed(storage) || storage->leaves.empty() || storage->leaves.size() > std::numeric_limits<uint32_t>::max())
            return failure();
        const uint32_t first = static_cast<uint32_t>(plan->fields.size());
        if (failed(append(argumentIndex, MetadataFieldKind::Offset)))
            return failure();
        for (unsigned dimension = 0; dimension < view.getShape().size(); ++dimension)
            if (failed(append(argumentIndex, MetadataFieldKind::Extent, dimension)))
                return failure();
        for (unsigned dimension = 0; dimension < view.getShape().size(); ++dimension)
            if (failed(append(argumentIndex, MetadataFieldKind::Stride, dimension)))
                return failure();
        plan->views.push_back({argumentIndex, first});
    }

    for (auto [ordinal, field] : llvm::enumerate(plan->fields))
        if (field.ordinal != ordinal)
            return failure();
    return std::shared_ptr<const SemanticMetadataPlan>(std::move(plan));
}

FailureOr<uint32_t> getCudaMetadataKernelParameterOrdinal(FunctionOpInterface function,
                                                          unsigned metadataArgumentIndex) {
    if (!function || metadataArgumentIndex > function.getNumArguments() ||
        (metadataArgumentIndex < function.getNumArguments() &&
         !function.getArgAttr(metadataArgumentIndex, kTensorMetadataCarrierAttrName)))
        return failure();

    uint64_t ordinal = 0;
    for (unsigned index = 0; index < metadataArgumentIndex; ++index) {
        if (function.getArgAttr(index, kTensorMetadataCarrierAttrName))
            return failure();
        Type type = function.getArgumentTypes()[index];
        const bool builtin = function.getArgAttr(index, kBuiltinAttrName) != nullptr;
        auto interfaceKind = function.getArgAttrOfType<StringAttr>(index, kInterfaceAttrName);
        uint64_t arity = 0;
        if (!builtin && isa<TensorType, RankedTensorType>(type)) {
            arity = 5;
        } else if (interfaceKind && interfaceKind.getValue() == "resource") {
            if (isa<TextureType>(type)) {
                arity = 1;
            } else if (auto view = dyn_cast<TensorViewType>(type)) {
                FailureOr<ValueAbiLayout> storage =
                    getValueStorageLayout(view.getElementType(), function->getParentOfType<ModuleOp>());
                if (failed(storage) || storage->leaves.empty())
                    return failure();
                arity = 5 * static_cast<uint64_t>(storage->leaves.size());
            } else {
                return failure();
            }
        } else if (!builtin && (type.isIntOrIndexOrFloat() || isa<VectorType>(type))) {
            arity = 1;
        }
        if (arity > std::numeric_limits<uint32_t>::max() - ordinal)
            return failure();
        ordinal += arity;
    }
    return static_cast<uint32_t>(ordinal);
}

FailureOr<std::shared_ptr<const PhysicalMetadataPlan>> getPhysicalMetadataPlan(const SemanticMetadataPlan &semantic,
                                                                               MetadataPhysicalProfile profile,
                                                                               MetadataNativeLocation nativeLocation,
                                                                               uint32_t hostIndexBitWidth) {
    auto plan = std::make_shared<PhysicalMetadataPlan>();
    plan->nativeLocation = nativeLocation;
    uint64_t scalarSize = 0;
    if (profile == MetadataPhysicalProfile::PortableShaderMetadataI32) {
        scalarSize = 4;
        plan->scalarRepresentation = "i32";
        plan->carrierKind = MetadataCarrierKind::ConstantRegion;
        plan->abiAlignment = 16;
        plan->compiledSizeCeiling = kPortableShaderMetadataSizeCeiling;
    } else if (profile == MetadataPhysicalProfile::CudaKernelMetadata) {
        scalarSize = 8;
        plan->scalarRepresentation = "i64";
        plan->carrierKind = MetadataCarrierKind::KernelParameter;
        plan->abiAlignment = scalarSize;
        plan->compiledSizeCeiling = 4096;
    } else {
        if (hostIndexBitWidth != 32 && hostIndexBitWidth != 64)
            return failure();
        scalarSize = hostIndexBitWidth / 8;
        plan->scalarRepresentation = hostIndexBitWidth == 32 ? StringRef("i32") : StringRef("i64");
        plan->carrierKind = MetadataCarrierKind::CpuCallFrame;
        plan->abiAlignment = scalarSize;
        plan->compiledSizeCeiling = std::numeric_limits<uint32_t>::max();
    }
    const bool hasShaderLocation = nativeLocation.descriptorSet && nativeLocation.binding;
    const bool hasPartialShaderLocation = nativeLocation.descriptorSet || nativeLocation.binding;
    if (hasPartialShaderLocation != hasShaderLocation)
        return failure();
    if (profile == MetadataPhysicalProfile::PortableShaderMetadataI32) {
        if (nativeLocation.kernelParameterOrdinal)
            return failure();
    } else if (hasPartialShaderLocation) {
        return failure();
    }

    if (semantic.getFieldCount() > std::numeric_limits<uint64_t>::max() / scalarSize)
        return failure();
    plan->encodedSize = semantic.getFieldCount() * scalarSize;
    plan->blockSize = profile == MetadataPhysicalProfile::PortableShaderMetadataI32
                          ? llvm::alignTo(plan->encodedSize, plan->abiAlignment)
                          : plan->encodedSize;
    if (plan->blockSize > plan->compiledSizeCeiling)
        return failure();

    uint64_t previousEnd = 0;
    for (const SemanticMetadataField &field : semantic.getFields()) {
        const uint64_t offset = field.ordinal * scalarSize;
        if (offset < previousEnd || offset % scalarSize != 0)
            return failure();
        plan->members.push_back({field.ordinal, offset, scalarSize, scalarSize});
        previousEnd = offset + scalarSize;
    }
    if (previousEnd != plan->encodedSize || plan->members.size() != semantic.getFields().size())
        return failure();

    std::string canonical;
    llvm::raw_string_ostream stream(canonical);
    stream << "tensor_view_metadata(profile=" << stringifyMetadataPhysicalProfile(profile)
           << ",representation=" << plan->scalarRepresentation << ",encoded_size=" << plan->encodedSize
           << ",block_size=" << plan->blockSize << ",alignment=" << plan->abiAlignment << ";fields=";
    for (const SemanticMetadataField &field : semantic.getFields()) {
        stream << field.ordinal << '@' << field.identity.argumentIndex << ':'
               << stringifyMetadataFieldKind(field.identity.kind);
        if (field.identity.dimension)
            stream << '[' << *field.identity.dimension << ']';
        const PhysicalMetadataMember &member = plan->members[field.ordinal];
        stream << "->" << member.byteOffset << ':' << member.size << ':' << member.alignment << ';';
    }
    stream << ')';
    stream.flush();
    llvm::SHA256 hash;
    hash.update(canonical);
    plan->canonicalLayoutHash = llvm::toHex(hash.final(), true);
    return std::shared_ptr<const PhysicalMetadataPlan>(std::move(plan));
}

} // namespace mlir::vernon
