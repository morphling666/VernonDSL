#ifndef VERNON_FORMATS_HPP
#define VERNON_FORMATS_HPP

#include "VernonTextureTypes.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace vernon::formats {

enum class ComponentClass : uint8_t { Float, Unorm, Uint, Sint, Snorm };

struct Format {
    std::string_view name;
    VernonTextureFormat texture;
    ComponentClass componentClass;
    uint32_t pixelBytes;
    VernonHalImageDataFormat dataFormat;
    VernonHalImageDataType dataType;
    uint32_t baseline;
    uint32_t glInternalFormat;
    uint32_t glFormat;
    uint32_t glType;

    constexpr uint32_t aspects() const {
        return dataFormat == VERNON_HAL_IMAGE_DATA_DEPTH ? VERNON_IMAGE_ASPECT_DEPTH
               : dataFormat == VERNON_HAL_IMAGE_DATA_DEPTH_STENCIL
                   ? VERNON_IMAGE_ASPECT_DEPTH | VERNON_IMAGE_ASPECT_STENCIL
                   : VERNON_IMAGE_ASPECT_COLOR;
    }
    constexpr uint32_t components() const {
        return dataFormat == VERNON_HAL_IMAGE_DATA_RG                                                 ? 2
               : dataFormat == VERNON_HAL_IMAGE_DATA_RGB || dataFormat == VERNON_HAL_IMAGE_DATA_BGR   ? 3
               : dataFormat == VERNON_HAL_IMAGE_DATA_RGBA || dataFormat == VERNON_HAL_IMAGE_DATA_BGRA ? 4
                                                                                                      : 1;
    }
    constexpr bool allows(uint32_t capabilities) const { return (baseline & capabilities) == capabilities; }
    constexpr bool integer() const {
        return componentClass == ComponentClass::Uint || componentClass == ComponentClass::Sint;
    }
};

namespace detail {

inline constexpr uint32_t invalidColumn = 1u << 31;

constexpr uint32_t capabilityBit(std::string_view word) {
    return word == "renderable"    ? VERNON_HAL_FORMAT_RENDERABLE
           : word == "blendable"   ? VERNON_HAL_FORMAT_BLENDABLE
           : word == "filterable"  ? VERNON_HAL_FORMAT_FILTERABLE
           : word == "storage"     ? VERNON_HAL_FORMAT_STORAGE
           : word == "resolvable"  ? VERNON_HAL_FORMAT_RESOLVABLE
           : word == "multisample" ? VERNON_HAL_FORMAT_MULTISAMPLE
           : word == "copyable"    ? VERNON_HAL_FORMAT_COPYABLE
                                   : invalidColumn;
}

constexpr uint32_t capabilities(std::string_view words) {
    uint32_t result = 0;
    while (!words.empty()) {
        const size_t end = words.find(' ');
        result |= capabilityBit(words.substr(0, end));
        words = end == std::string_view::npos ? std::string_view{} : words.substr(end + 1);
    }
    return result;
}

constexpr ComponentClass componentClass(std::string_view name) {
    return name == "float"   ? ComponentClass::Float
           : name == "unorm" ? ComponentClass::Unorm
           : name == "uint"  ? ComponentClass::Uint
           : name == "sint"  ? ComponentClass::Sint
           : name == "snorm" ? ComponentClass::Snorm
                             : static_cast<ComponentClass>(0xFFu);
}

} // namespace detail

inline constexpr std::array table{
#define VERNON_FORMAT(name, NAME, texture, cls, bytes, dataFormat, dataType, caps, glInternal, glFormat, glType, ...)  \
    Format{#name,                                                                                                      \
           VERNON_TEXTURE_##NAME,                                                                                      \
           detail::componentClass(cls),                                                                                \
           bytes,                                                                                                      \
           VERNON_HAL_IMAGE_DATA_##dataFormat,                                                                         \
           VERNON_HAL_IMAGE_DATA_##dataType,                                                                           \
           detail::capabilities(caps),                                                                                 \
           glInternal,                                                                                                 \
           glFormat,                                                                                                   \
           glType},
#include "VernonFormats.def"
};

constexpr const Format *find(VernonTextureFormat format) {
    for (const Format &entry : table)
        if (entry.texture == format)
            return &entry;
    return nullptr;
}

constexpr const Format *find(std::string_view name) {
    for (const Format &entry : table)
        if (entry.name == name)
            return &entry;
    return nullptr;
}

template <typename Key> constexpr uint32_t aspects(Key format) {
    const Format *entry = find(format);
    return entry ? entry->aspects() : 0;
}

template <typename Key> constexpr bool hasDepth(Key format) {
    return (aspects(format) & VERNON_IMAGE_ASPECT_DEPTH) != 0;
}

template <typename Key> constexpr bool hasStencil(Key format) {
    return (aspects(format) & VERNON_IMAGE_ASPECT_STENCIL) != 0;
}

template <typename Key> constexpr bool isInteger(Key format) {
    const Format *entry = find(format);
    return entry && entry->integer();
}

// Shaders read normalized and float views alike as floats; integer views keep their signedness.
constexpr VernonImageSampleResultClass sampleResultClass(const Format &entry) {
    return entry.componentClass == ComponentClass::Uint   ? VERNON_IMAGE_SAMPLE_UNSIGNED_INTEGER
           : entry.componentClass == ComponentClass::Sint ? VERNON_IMAGE_SAMPLE_SIGNED_INTEGER
                                                          : VERNON_IMAGE_SAMPLE_FLOAT;
}

template <typename Key> constexpr bool providesSampleResultClass(Key format, VernonImageSampleResultClass resultClass) {
    const Format *entry = find(format);
    return entry && sampleResultClass(*entry) == resultClass;
}

constexpr std::string_view sampleResultClassName(VernonImageSampleResultClass resultClass) {
    switch (resultClass) {
    case VERNON_IMAGE_SAMPLE_FLOAT:
        return "float";
    case VERNON_IMAGE_SAMPLE_SIGNED_INTEGER:
        return "sint";
    case VERNON_IMAGE_SAMPLE_UNSIGNED_INTEGER:
        return "uint";
    case VERNON_IMAGE_SAMPLE_NONE:
        break;
    }
    return "none";
}

constexpr std::optional<VernonImageSampleResultClass> parseSampleResultClass(std::string_view name) {
    for (VernonImageSampleResultClass resultClass :
         {VERNON_IMAGE_SAMPLE_FLOAT, VERNON_IMAGE_SAMPLE_SIGNED_INTEGER, VERNON_IMAGE_SAMPLE_UNSIGNED_INTEGER})
        if (sampleResultClassName(resultClass) == name)
            return resultClass;
    return std::nullopt;
}

// A view may reinterpret an image only between the unorm and sRGB encodings of the same texels.
template <typename Key> constexpr bool viewCompatible(Key image, Key view) {
    const Format *imageEntry = find(image);
    const Format *viewEntry = find(view);
    const auto encodings = [](const Format *entry) {
        return entry->texture == VERNON_TEXTURE_RGBA8_UNORM || entry->texture == VERNON_TEXTURE_RGBA8_SRGB   ? 1
               : entry->texture == VERNON_TEXTURE_BGRA8_UNORM || entry->texture == VERNON_TEXTURE_BGRA8_SRGB ? 2
                                                                                                             : 0;
    };
    return imageEntry && viewEntry &&
           (imageEntry == viewEntry || (encodings(imageEntry) != 0 && encodings(imageEntry) == encodings(viewEntry)));
}

namespace detail {

constexpr bool validTable() {
    for (size_t index = 0; index < table.size(); ++index) {
        const Format &entry = table[index];
        if (entry.pixelBytes == 0 ||
            static_cast<uint8_t>(entry.componentClass) > static_cast<uint8_t>(ComponentClass::Snorm) ||
            (entry.baseline & invalidColumn) != 0 || find(entry.texture) != &entry || find(entry.name) != &entry)
            return false;
    }
    return true;
}

static_assert(validTable(), "VernonFormats.def has an unknown column word or a duplicate name or value");

} // namespace detail

} // namespace vernon::formats

#endif
