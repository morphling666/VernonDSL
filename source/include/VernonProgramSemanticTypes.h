#ifndef VERNON_PROGRAM_SEMANTIC_TYPES_H
#define VERNON_PROGRAM_SEMANTIC_TYPES_H

#include <array>
#include <cctype>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vernon::program {

enum class ScalarSemanticId {
    Bool,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F16,
    F32,
    F64,
};

enum class ScalarCategory { Boolean, SignedInteger, UnsignedInteger, FloatingPoint };

struct ScalarDescriptor {
    ScalarSemanticId id;
    std::string_view spelling;
    uint16_t bitWidth;
    ScalarCategory category;
};

inline constexpr std::array<ScalarDescriptor, 12> programScalarDescriptors{{
    {ScalarSemanticId::Bool, "bool", 1, ScalarCategory::Boolean},
    {ScalarSemanticId::I8, "i8", 8, ScalarCategory::SignedInteger},
    {ScalarSemanticId::U8, "u8", 8, ScalarCategory::UnsignedInteger},
    {ScalarSemanticId::I16, "i16", 16, ScalarCategory::SignedInteger},
    {ScalarSemanticId::U16, "u16", 16, ScalarCategory::UnsignedInteger},
    {ScalarSemanticId::I32, "i32", 32, ScalarCategory::SignedInteger},
    {ScalarSemanticId::U32, "u32", 32, ScalarCategory::UnsignedInteger},
    {ScalarSemanticId::I64, "i64", 64, ScalarCategory::SignedInteger},
    {ScalarSemanticId::U64, "u64", 64, ScalarCategory::UnsignedInteger},
    {ScalarSemanticId::F16, "f16", 16, ScalarCategory::FloatingPoint},
    {ScalarSemanticId::F32, "f32", 32, ScalarCategory::FloatingPoint},
    {ScalarSemanticId::F64, "f64", 64, ScalarCategory::FloatingPoint},
}};

inline const ScalarDescriptor *programScalar(std::string_view spelling) {
    for (const ScalarDescriptor &scalar : programScalarDescriptors)
        if (scalar.spelling == spelling)
            return &scalar;
    return nullptr;
}

inline const ScalarDescriptor *programScalar(ScalarSemanticId id) {
    for (const ScalarDescriptor &scalar : programScalarDescriptors)
        if (scalar.id == id)
            return &scalar;
    return nullptr;
}

enum class BuiltinStorageContractId { Sampler, AdTape };

struct BuiltinStorageContract {
    BuiltinStorageContractId id;
    std::string_view contract;
    std::string_view canonicalType;
};

inline constexpr std::array<BuiltinStorageContract, 2> builtinStorageContracts{{
    {BuiltinStorageContractId::Sampler, "vernon.sampler", "sampler"},
    {BuiltinStorageContractId::AdTape, "vernon.ad_tape", "opaque<vernon.ad_tape>"},
}};

inline constexpr const BuiltinStorageContract *builtinStorageContract(BuiltinStorageContractId id) {
    for (const BuiltinStorageContract &contract : builtinStorageContracts)
        if (contract.id == id)
            return &contract;
    return nullptr;
}

enum class SemanticTypeKind {
    Scalar,
    Tuple,
    Struct,
    Tensor,
    TensorView,
    Image,
    ImageView,
    Sampler,
    Opaque,
};

struct SemanticType {
    SemanticTypeKind kind{SemanticTypeKind::Scalar};
    ScalarSemanticId scalar{ScalarSemanticId::Bool};
    std::vector<int64_t> dimensions;
    std::vector<SemanticType> elements;
    std::vector<std::pair<std::string, SemanticType>> fields;
    std::string parameter;

    bool operator==(const SemanticType &other) const {
        return kind == other.kind && scalar == other.scalar && dimensions == other.dimensions &&
               elements == other.elements && fields == other.fields && parameter == other.parameter;
    }
    bool operator!=(const SemanticType &other) const { return !(*this == other); }

    bool isResource() const {
        return kind == SemanticTypeKind::Image || kind == SemanticTypeKind::ImageView ||
               kind == SemanticTypeKind::Sampler;
    }
    bool isStorage() const { return kind == SemanticTypeKind::TensorView || kind == SemanticTypeKind::Opaque; }
    bool isTensorView() const { return kind == SemanticTypeKind::TensorView; }
    bool isImage() const { return kind == SemanticTypeKind::Image || kind == SemanticTypeKind::ImageView; }
    bool isSampler() const { return kind == SemanticTypeKind::Sampler; }
    bool isAdTape() const {
        return kind == SemanticTypeKind::Opaque &&
               parameter == builtinStorageContract(BuiltinStorageContractId::AdTape)->contract;
    }
    bool isRankedValue() const { return kind == SemanticTypeKind::Tensor; }
};

namespace detail {

inline bool validIdentifier(std::string_view value) {
    if (value.empty() || !(std::isalpha(static_cast<unsigned char>(value.front())) || value.front() == '_'))
        return false;
    for (char character : value.substr(1))
        if (!(std::isalnum(static_cast<unsigned char>(character)) || character == '_'))
            return false;
    return true;
}

inline bool validOpaqueContract(std::string_view value) {
    size_t begin = 0;
    while (begin < value.size()) {
        const size_t end = value.find('.', begin);
        const std::string_view component =
            value.substr(begin, end == std::string_view::npos ? value.size() - begin : end - begin);
        if (!validIdentifier(component))
            return false;
        if (end == std::string_view::npos)
            return true;
        begin = end + 1;
    }
    return false;
}

inline bool validImageFormat(std::string_view value) {
    constexpr std::array<std::string_view, 20> formats{
        "r8_unorm",      "r8_snorm",
        "r16_float",     "r32_float",
        "r32_uint",      "r32_sint",
        "rg8_unorm",     "rg16_float",
        "rg32_float",    "rgba8_unorm",
        "rgba8_srgb",    "bgra8_unorm",
        "bgra8_srgb",    "rgba16_float",
        "rgba32_float",  "depth16_unorm",
        "depth24_plus",  "depth24_plus_stencil8",
        "depth32_float", "depth32_float_stencil8",
    };
    if (value == "unknown")
        return true;
    for (std::string_view format : formats)
        if (format == value)
            return true;
    return false;
}

class SemanticTypeParser {
public:
    explicit SemanticTypeParser(std::string_view text) : text_(text) {}

    std::optional<SemanticType> parse() {
        if (text_.empty())
            return std::nullopt;
        for (char character : text_)
            if (std::isspace(static_cast<unsigned char>(character)))
                return std::nullopt;
        std::optional<SemanticType> result = parseType();
        if (!result || position_ != text_.size())
            return std::nullopt;
        return result;
    }

private:
    bool consume(std::string_view token) {
        if (text_.substr(position_, token.size()) != token)
            return false;
        position_ += token.size();
        return true;
    }

    std::optional<uint64_t> positiveInteger() {
        const size_t begin = position_;
        if (begin >= text_.size() || text_[begin] < '1' || text_[begin] > '9')
            return std::nullopt;
        uint64_t value = 0;
        while (position_ < text_.size() && text_[position_] >= '0' && text_[position_] <= '9') {
            const uint64_t digit = static_cast<uint64_t>(text_[position_] - '0');
            if (value > (static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - digit) / 10)
                return std::nullopt;
            value = value * 10 + digit;
            ++position_;
        }
        return value;
    }

    std::optional<std::string> identifierUntil(char delimiter) {
        const size_t begin = position_;
        const size_t end = text_.find(delimiter, begin);
        if (end == std::string_view::npos)
            return std::nullopt;
        std::string_view value = text_.substr(begin, end - begin);
        if (!validIdentifier(value))
            return std::nullopt;
        position_ = end;
        return std::string(value);
    }

    std::optional<SemanticType> parseType() {
        for (const ScalarDescriptor &scalar : programScalarDescriptors)
            if (text_.substr(position_, scalar.spelling.size()) == scalar.spelling &&
                (position_ + scalar.spelling.size() == text_.size() ||
                 text_[position_ + scalar.spelling.size()] == ',' ||
                 text_[position_ + scalar.spelling.size()] == '>')) {
                position_ += scalar.spelling.size();
                SemanticType result;
                result.scalar = scalar.id;
                return result;
            }
        if (consume("sampler")) {
            SemanticType result;
            result.kind = SemanticTypeKind::Sampler;
            return result;
        }
        if (consume("tuple<"))
            return parseTuple();
        if (consume("struct<"))
            return parseStruct();
        for (auto [prefix, kind] : {std::pair<std::string_view, SemanticTypeKind>{"tensor<", SemanticTypeKind::Tensor},
                                    {"tensor_view<", SemanticTypeKind::TensorView}})
            if (consume(prefix))
                return parseRanked(kind);
        for (auto [prefix, kind] : {std::pair<std::string_view, SemanticTypeKind>{"image<", SemanticTypeKind::Image},
                                    {"image_view<", SemanticTypeKind::ImageView},
                                    {"opaque<", SemanticTypeKind::Opaque}})
            if (consume(prefix))
                return parseParameter(kind);
        return std::nullopt;
    }

    std::optional<SemanticType> parseRanked(SemanticTypeKind kind) {
        SemanticType result;
        result.kind = kind;
        if (kind == SemanticTypeKind::Tensor || kind == SemanticTypeKind::TensorView) {
            const size_t saved = position_;
            if (auto element = parseType(); element && consume(">")) {
                result.elements.push_back(std::move(*element));
                return result;
            }
            position_ = saved;
        }
        while (true) {
            if ((kind == SemanticTypeKind::Tensor || kind == SemanticTypeKind::TensorView) && consume("?")) {
                result.dimensions.push_back(-1);
            } else {
                std::optional<uint64_t> extent = positiveInteger();
                if (!extent)
                    break;
                result.dimensions.push_back(static_cast<int64_t>(*extent));
            }
            if (!consume("x"))
                return std::nullopt;
            const size_t saved = position_;
            if (auto element = parseType()) {
                if (!consume(">"))
                    return std::nullopt;
                result.elements.push_back(std::move(*element));
                return result;
            }
            position_ = saved;
        }
        return std::nullopt;
    }

    std::optional<SemanticType> parseTuple() {
        SemanticType result;
        result.kind = SemanticTypeKind::Tuple;
        if (consume(">"))
            return result;
        while (true) {
            auto element = parseType();
            if (!element)
                return std::nullopt;
            result.elements.push_back(std::move(*element));
            if (consume(">"))
                return result;
            if (!consume(","))
                return std::nullopt;
        }
    }

    std::optional<SemanticType> parseStruct() {
        SemanticType result;
        result.kind = SemanticTypeKind::Struct;
        if (consume(">"))
            return result;
        while (true) {
            auto name = identifierUntil(':');
            if (!name || !consume(":"))
                return std::nullopt;
            for (const auto &field : result.fields)
                if (field.first == *name)
                    return std::nullopt;
            auto fieldType = parseType();
            if (!fieldType)
                return std::nullopt;
            result.fields.emplace_back(std::move(*name), std::move(*fieldType));
            if (consume(">"))
                return result;
            if (!consume(","))
                return std::nullopt;
        }
    }

    std::optional<SemanticType> parseParameter(SemanticTypeKind kind) {
        const size_t end = text_.find('>', position_);
        if (end == std::string_view::npos || end == position_)
            return std::nullopt;
        const std::string_view value = text_.substr(position_, end - position_);
        if ((kind == SemanticTypeKind::Image || kind == SemanticTypeKind::ImageView) && !validImageFormat(value))
            return std::nullopt;
        if (kind == SemanticTypeKind::Opaque && !validOpaqueContract(value))
            return std::nullopt;
        position_ = end + 1;
        SemanticType result;
        result.kind = kind;
        result.parameter = std::string(value);
        return result;
    }

    std::string_view text_;
    size_t position_{};
};

} // namespace detail

inline std::optional<SemanticType> parseSemanticType(std::string_view text) {
    return detail::SemanticTypeParser(text).parse();
}

inline std::string serializeSemanticType(const SemanticType &type) {
    if (type.kind == SemanticTypeKind::Scalar) {
        const ScalarDescriptor *scalar = programScalar(type.scalar);
        return scalar ? std::string(scalar->spelling) : std::string();
    }
    if (type.kind == SemanticTypeKind::Sampler)
        return "sampler";
    if (type.kind == SemanticTypeKind::Image || type.kind == SemanticTypeKind::ImageView ||
        type.kind == SemanticTypeKind::Opaque) {
        const char *prefix = type.kind == SemanticTypeKind::Image       ? "image<"
                             : type.kind == SemanticTypeKind::ImageView ? "image_view<"
                                                                        : "opaque<";
        return std::string(prefix) + type.parameter + ">";
    }
    if (type.kind == SemanticTypeKind::Tuple) {
        std::string result = "tuple<";
        for (size_t index = 0; index < type.elements.size(); ++index) {
            if (index)
                result += ',';
            result += serializeSemanticType(type.elements[index]);
        }
        return result + ">";
    }
    if (type.kind == SemanticTypeKind::Struct) {
        std::string result = "struct<";
        for (size_t index = 0; index < type.fields.size(); ++index) {
            if (index)
                result += ',';
            result += type.fields[index].first + ":" + serializeSemanticType(type.fields[index].second);
        }
        return result + ">";
    }
    const char *prefix = type.kind == SemanticTypeKind::Tensor ? "tensor<" : "tensor_view<";
    std::string result(prefix);
    for (int64_t dimension : type.dimensions)
        result += (dimension < 0 ? "?" : std::to_string(dimension)) + std::string("x");
    if (type.elements.size() != 1)
        return {};
    return result + serializeSemanticType(type.elements.front()) + ">";
}

inline bool appendSemanticScalarLeaves(const SemanticType &type, std::vector<ScalarSemanticId> &leaves) {
    if (type.kind == SemanticTypeKind::Scalar) {
        leaves.push_back(type.scalar);
        return true;
    }
    if (type.kind == SemanticTypeKind::Struct) {
        for (const auto &field : type.fields)
            if (!appendSemanticScalarLeaves(field.second, leaves))
                return false;
        return true;
    }
    if (type.elements.size() != 1 && type.kind != SemanticTypeKind::Tuple)
        return type.elements.empty() && (type.isStorage() || type.isResource());
    size_t repetitions = 1;
    if (type.isRankedValue() && !type.elements.empty() && type.elements.front().kind != SemanticTypeKind::Scalar) {
        for (int64_t dimension : type.dimensions) {
            if (dimension <= 0 || repetitions > std::numeric_limits<size_t>::max() / static_cast<size_t>(dimension))
                return false;
            repetitions *= static_cast<size_t>(dimension);
        }
    }
    for (size_t repetition = 0; repetition < repetitions; ++repetition)
        for (const SemanticType &element : type.elements)
            if (!appendSemanticScalarLeaves(element, leaves))
                return false;
    return true;
}

} // namespace vernon::program

#endif
