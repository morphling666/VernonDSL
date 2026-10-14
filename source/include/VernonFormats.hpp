#ifndef VERNON_FORMATS_HPP
#define VERNON_FORMATS_HPP

#include "VernonRHI.h"
#include "VernonTextureTypes.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace vernon::formats {

enum class ComponentClass : uint8_t { Float, Unorm, Uint, Sint };

struct Format {
    std::string_view name;
    VernonRhiFormat rhi;
    VernonTextureFormat texture;
    ComponentClass componentClass;
    uint32_t pixelBytes;
    VernonRhiImageDataFormat dataFormat;
    VernonRhiImageDataType dataType;
    uint32_t baseline;
    uint32_t glInternalFormat;
    uint32_t glFormat;
    uint32_t glType;

    constexpr uint32_t aspects() const {
        return dataFormat == VERNON_RHI_IMAGE_DATA_DEPTH ? VERNON_RHI_IMAGE_ASPECT_DEPTH
               : dataFormat == VERNON_RHI_IMAGE_DATA_DEPTH_STENCIL
                   ? VERNON_RHI_IMAGE_ASPECT_DEPTH | VERNON_RHI_IMAGE_ASPECT_STENCIL
                   : VERNON_RHI_IMAGE_ASPECT_COLOR;
    }
    constexpr bool allows(uint32_t capabilities) const { return (baseline & capabilities) == capabilities; }
    constexpr bool integer() const {
        return componentClass == ComponentClass::Uint || componentClass == ComponentClass::Sint;
    }
};

namespace detail {

inline constexpr uint32_t invalidColumn = 1u << 31;

constexpr uint32_t capabilityBit(std::string_view word) {
    return word == "renderable"    ? VERNON_RHI_FORMAT_RENDERABLE
           : word == "blendable"   ? VERNON_RHI_FORMAT_BLENDABLE
           : word == "filterable"  ? VERNON_RHI_FORMAT_FILTERABLE
           : word == "storage"     ? VERNON_RHI_FORMAT_STORAGE
           : word == "resolvable"  ? VERNON_RHI_FORMAT_RESOLVABLE
           : word == "multisample" ? VERNON_RHI_FORMAT_MULTISAMPLE
           : word == "copyable"    ? VERNON_RHI_FORMAT_COPYABLE
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
                             : static_cast<ComponentClass>(0xFFu);
}

} // namespace detail

inline constexpr std::array table{
#define VERNON_FORMAT(name, NAME, rhi, texture, cls, bytes, dataFormat, dataType, caps, glInternal, glFormat, glType,  \
                      ...)                                                                                             \
    Format{#name,                                                                                                      \
           VERNON_RHI_FORMAT_##NAME,                                                                                   \
           VERNON_TEXTURE_##NAME,                                                                                      \
           detail::componentClass(cls),                                                                                \
           bytes,                                                                                                      \
           VERNON_RHI_IMAGE_DATA_##dataFormat,                                                                         \
           VERNON_RHI_IMAGE_DATA_##dataType,                                                                           \
           detail::capabilities(caps),                                                                                 \
           glInternal,                                                                                                 \
           glFormat,                                                                                                   \
           glType},
#include "VernonFormats.def"
};

constexpr const Format *find(VernonRhiFormat format) {
    for (const Format &entry : table)
        if (entry.rhi == format)
            return &entry;
    return nullptr;
}

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
    return (aspects(format) & VERNON_RHI_IMAGE_ASPECT_DEPTH) != 0;
}

template <typename Key> constexpr bool hasStencil(Key format) {
    return (aspects(format) & VERNON_RHI_IMAGE_ASPECT_STENCIL) != 0;
}

template <typename Key> constexpr bool isInteger(Key format) {
    const Format *entry = find(format);
    return entry && entry->integer();
}

// A view may reinterpret an image only between the unorm and sRGB encodings of the same texels.
template <typename Key> constexpr bool viewCompatible(Key image, Key view) {
    const Format *imageEntry = find(image);
    const Format *viewEntry = find(view);
    const auto rgba8 = [](const Format *entry) {
        return entry->rhi == VERNON_RHI_FORMAT_RGBA8_UNORM || entry->rhi == VERNON_RHI_FORMAT_RGBA8_SRGB;
    };
    return imageEntry && viewEntry && (imageEntry == viewEntry || (rgba8(imageEntry) && rgba8(viewEntry)));
}

namespace detail {

constexpr bool validTable() {
    for (size_t index = 0; index < table.size(); ++index) {
        const Format &entry = table[index];
        if (entry.rhi == VERNON_RHI_FORMAT_UNDEFINED || entry.pixelBytes == 0 ||
            static_cast<uint8_t>(entry.componentClass) > static_cast<uint8_t>(ComponentClass::Sint) ||
            (entry.baseline & invalidColumn) != 0 || find(entry.rhi) != &entry || find(entry.texture) != &entry ||
            find(entry.name) != &entry)
            return false;
    }
    return true;
}

static_assert(validTable(), "VernonFormats.def has an unknown column word or a duplicate name or value");
static_assert(static_cast<uint32_t>(VERNON_RHI_IMAGE_ASPECT_COLOR) == VERNON_IMAGE_ASPECT_COLOR &&
                  static_cast<uint32_t>(VERNON_RHI_IMAGE_ASPECT_DEPTH) == VERNON_IMAGE_ASPECT_DEPTH &&
                  static_cast<uint32_t>(VERNON_RHI_IMAGE_ASPECT_STENCIL) == VERNON_IMAGE_ASPECT_STENCIL,
              "Format::aspects() serves both the RHI and the Runtime aspect bits");

} // namespace detail

} // namespace vernon::formats

#endif
