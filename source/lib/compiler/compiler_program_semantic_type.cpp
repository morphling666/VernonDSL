#include "compiler_program_semantic_type.h"

#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/IR/VernonValueAbi.h"
#include "llvm/ADT/DenseSet.h"

namespace vernon::compiler {
namespace {

using SemanticType = vernon::program::SemanticType;
using SemanticTypeKind = vernon::program::SemanticTypeKind;
using vernon::program::builtinStorageContract;
using vernon::program::BuiltinStorageContractId;

class MlirProgramTypeReader {
public:
    MlirProgramTypeReader(mlir::ModuleOp module, llvm::ArrayRef<llvm::StringRef> logicalDtypes)
        : module_(module), logicalDtypes_(logicalDtypes), requiresDeclaredDtypes_(!logicalDtypes.empty()) {}

    mlir::FailureOr<SemanticType> read(mlir::Type type) {
        if (type.isIntOrFloat())
            return readScalar(type);
        if (auto tensor = mlir::dyn_cast<mlir::vernon::TensorType>(type))
            return readContainer(SemanticTypeKind::Tensor, tensor.getShape(), tensor.getElementType());
        if (auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type))
            return readContainer(SemanticTypeKind::Tensor, tensor.getShape(), tensor.getElementType());
        if (auto vector = mlir::dyn_cast<mlir::VectorType>(type)) {
            if (vector.isScalable())
                return mlir::failure();
            return readContainer(SemanticTypeKind::Tensor, vector.getShape(), vector.getElementType());
        }
        if (auto view = mlir::dyn_cast<mlir::vernon::TensorViewType>(type))
            return readContainer(SemanticTypeKind::TensorView, view.getShape(), view.getElementType());
        if (auto tuple = mlir::dyn_cast<mlir::TupleType>(type)) {
            SemanticType result;
            result.kind = SemanticTypeKind::Tuple;
            for (mlir::Type element : tuple.getTypes()) {
                mlir::FailureOr<SemanticType> semantic = read(element);
                if (mlir::failed(semantic))
                    return mlir::failure();
                result.elements.push_back(std::move(*semantic));
            }
            return result;
        }
        if (auto structure = mlir::dyn_cast<mlir::vernon::StructType>(type))
            return readStruct(structure);
        if (auto texture = mlir::dyn_cast<mlir::vernon::TextureType>(type)) {
            const llvm::StringRef format =
                texture.getFormat() == "d32_float" ? llvm::StringRef("depth32_float") : texture.getFormat();
            SemanticType result;
            result.kind = SemanticTypeKind::Image;
            result.parameter = format.str();
            return result;
        }
        if (mlir::isa<mlir::vernon::SamplerType>(type)) {
            SemanticType result;
            result.kind = SemanticTypeKind::Sampler;
            return result;
        }
        if (mlir::isa<mlir::vernon::AdTapeType>(type)) {
            SemanticType result;
            result.kind = SemanticTypeKind::Opaque;
            result.parameter = std::string(builtinStorageContract(BuiltinStorageContractId::AdTape)->contract);
            return result;
        }
        return mlir::failure();
    }

    bool consumedAllDtypes() const { return dtypeIndex_ == logicalDtypes_.size(); }

private:
    static const vernon::program::ScalarDescriptor *inferredScalar(mlir::Type type) {
        if (type.isInteger(1))
            return vernon::program::programScalar(vernon::program::ScalarSemanticId::Bool);
        if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type)) {
            switch (integer.getWidth()) {
            case 8:
                return vernon::program::programScalar(vernon::program::ScalarSemanticId::I8);
            case 16:
                return vernon::program::programScalar(vernon::program::ScalarSemanticId::I16);
            case 32:
                return vernon::program::programScalar(vernon::program::ScalarSemanticId::I32);
            case 64:
                return vernon::program::programScalar(vernon::program::ScalarSemanticId::I64);
            default:
                return nullptr;
            }
        }
        if (type.isF16())
            return vernon::program::programScalar(vernon::program::ScalarSemanticId::F16);
        if (type.isF32())
            return vernon::program::programScalar(vernon::program::ScalarSemanticId::F32);
        if (type.isF64())
            return vernon::program::programScalar(vernon::program::ScalarSemanticId::F64);
        return nullptr;
    }

    static bool storageMatches(mlir::Type type, const vernon::program::ScalarDescriptor &descriptor) {
        using vernon::program::ScalarCategory;
        if (descriptor.category == ScalarCategory::Boolean)
            return type.isInteger(1);
        if (descriptor.category == ScalarCategory::SignedInteger ||
            descriptor.category == ScalarCategory::UnsignedInteger) {
            auto integer = mlir::dyn_cast<mlir::IntegerType>(type);
            return integer && integer.getWidth() == descriptor.bitWidth && integer.getWidth() != 1;
        }
        switch (descriptor.id) {
        case vernon::program::ScalarSemanticId::F16:
            return type.isF16();
        case vernon::program::ScalarSemanticId::F32:
            return type.isF32();
        case vernon::program::ScalarSemanticId::F64:
            return type.isF64();
        default:
            return false;
        }
    }

    mlir::FailureOr<SemanticType> readScalar(mlir::Type type) {
        const vernon::program::ScalarDescriptor *descriptor = nullptr;
        if (dtypeIndex_ < logicalDtypes_.size()) {
            descriptor = vernon::program::programScalar(
                std::string_view(logicalDtypes_[dtypeIndex_].data(), logicalDtypes_[dtypeIndex_].size()));
            ++dtypeIndex_;
        } else if (!requiresDeclaredDtypes_) {
            descriptor = inferredScalar(type);
        }
        if (!descriptor || !storageMatches(type, *descriptor))
            return mlir::failure();
        SemanticType result;
        result.scalar = descriptor->id;
        return result;
    }

    mlir::FailureOr<SemanticType> readContainer(SemanticTypeKind kind, llvm::ArrayRef<int64_t> shape,
                                                mlir::Type elementType) {
        mlir::FailureOr<SemanticType> element = read(elementType);
        if (mlir::failed(element))
            return mlir::failure();
        SemanticType result;
        result.kind = kind;
        result.dimensions.assign(shape.begin(), shape.end());
        result.elements.push_back(std::move(*element));
        return result;
    }

    mlir::FailureOr<SemanticType> readStruct(mlir::vernon::StructType structure) {
        if (!activeStructs_.insert(structure.getName()).second)
            return mlir::failure();
        mlir::FailureOr<mlir::vernon::ResolvedStructFields> fields =
            mlir::vernon::resolveNamedStructFields(structure, module_);
        if (mlir::failed(fields)) {
            activeStructs_.erase(structure.getName());
            return mlir::failure();
        }
        SemanticType result;
        result.kind = SemanticTypeKind::Struct;
        for (const mlir::vernon::ResolvedStructField &field : fields->fields) {
            mlir::FailureOr<SemanticType> semantic = read(field.type);
            if (mlir::failed(semantic)) {
                activeStructs_.erase(structure.getName());
                return mlir::failure();
            }
            result.fields.emplace_back(field.name, std::move(*semantic));
        }
        activeStructs_.erase(structure.getName());
        return result;
    }

    mlir::ModuleOp module_;
    llvm::ArrayRef<llvm::StringRef> logicalDtypes_;
    size_t dtypeIndex_{};
    bool requiresDeclaredDtypes_{};
    llvm::SmallDenseSet<llvm::StringRef> activeStructs_;
};

} // namespace

mlir::FailureOr<vernon::program::SemanticType> programSemanticType(mlir::ModuleOp module, mlir::Type type,
                                                                   llvm::ArrayRef<llvm::StringRef> logicalDtypes) {
    MlirProgramTypeReader reader(module, logicalDtypes);
    mlir::FailureOr<vernon::program::SemanticType> semantic = reader.read(type);
    if (mlir::failed(semantic) || !reader.consumedAllDtypes())
        return mlir::failure();
    return semantic;
}

} // namespace vernon::compiler
