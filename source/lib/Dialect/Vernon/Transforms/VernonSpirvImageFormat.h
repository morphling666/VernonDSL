#pragma once

#include "VernonFormats.hpp"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

#include <optional>

namespace mlir::vernon {

inline std::optional<spirv::ImageFormat> spirvStorageImageFormat(llvm::StringRef format) {
#define VERNON_FORMAT(name, NAME, texture, cls, bytes, dataFormat, dataType, caps, glInternal, glFormat, glType,       \
                      vulkan, dxgi, metal, spirvFormat)                                                                \
    if (format == #name && spirv::ImageFormat::spirvFormat != spirv::ImageFormat::Unknown)                             \
        return spirv::ImageFormat::spirvFormat;
#include "VernonFormats.def"
    return std::nullopt;
}

// A sampled texture reads through a combined image-sampler, so its image needs a sampler and leaves the format to the
// bound view; a storage texture names its exact format and cannot be a cube.
inline std::optional<spirv::ImageType> spirvTextureImageType(TextureType texture) {
    std::optional<spirv::Dim> dimension = llvm::StringSwitch<std::optional<spirv::Dim>>(texture.getDimension())
                                              .Cases("2d", "2d_array", spirv::Dim::Dim2D)
                                              .Case("3d", spirv::Dim::Dim3D)
                                              .Cases("cube", "cube_array", spirv::Dim::Cube)
                                              .Default(std::nullopt);
    if (!dimension)
        return std::nullopt;
    const spirv::ImageArrayedInfo arrayed =
        texture.isArrayed() ? spirv::ImageArrayedInfo::Arrayed : spirv::ImageArrayedInfo::NonArrayed;
    if (texture.getAccess() == "sampled")
        return spirv::ImageType::get(texture.getElementType(), *dimension, spirv::ImageDepthInfo::NoDepth, arrayed,
                                     spirv::ImageSamplingInfo::SingleSampled, spirv::ImageSamplerUseInfo::NeedSampler,
                                     spirv::ImageFormat::Unknown);
    std::optional<spirv::ImageFormat> format = spirvStorageImageFormat(texture.getFormat());
    if (!format || texture.isCube())
        return std::nullopt;
    // A signed format needs a signed sampled type: SPIRV-Cross rejects an Rgba8i image of OpTypeInt 32 0. The texel
    // of a read or write must match the sampled type exactly, so it is bitcast at each access.
    const std::string_view name(texture.getFormat().data(), texture.getFormat().size());
    const ::vernon::formats::Format *entry = ::vernon::formats::find(name);
    Type sampledType = entry && entry->componentClass == ::vernon::formats::ComponentClass::Sint
                           ? Type(IntegerType::get(texture.getContext(), 32, IntegerType::Signed))
                           : texture.getElementType();
    return spirv::ImageType::get(sampledType, *dimension, spirv::ImageDepthInfo::NoDepth, arrayed,
                                 spirv::ImageSamplingInfo::SingleSampled, spirv::ImageSamplerUseInfo::NoSampler,
                                 *format);
}

// The texel type an access to `image` uses: `texelType` with the image's sampled type as its element.
inline Type spirvImageTexelType(spirv::ImageType image, Type texelType) {
    auto vector = dyn_cast<VectorType>(texelType);
    if (!vector || vector.getElementType() == image.getElementType())
        return texelType;
    return VectorType::get(vector.getShape(), image.getElementType());
}

inline void buildSpirvTextureStore(OpBuilder &builder, Location location, Value image, Value coordinates, Value texel) {
    Type imageTexelType = spirvImageTexelType(cast<spirv::ImageType>(image.getType()), texel.getType());
    if (imageTexelType != texel.getType())
        texel = spirv::BitcastOp::create(builder, location, imageTexelType, texel);
    spirv::ImageWriteOp::create(builder, location, image, coordinates, texel, spirv::ImageOperandsAttr(), ValueRange{});
}

inline std::optional<Type> spirvTextureDescriptorType(TextureType texture) {
    std::optional<spirv::ImageType> image = spirvTextureImageType(texture);
    if (!image)
        return std::nullopt;
    return texture.getAccess() == "sampled" ? Type(spirv::SampledImageType::get(*image)) : Type(*image);
}

// A null lod samples with implicit derivatives, which only fragment stages have.
inline Value buildSpirvTextureSample(OpBuilder &builder, Location location, Type resultType, Value sampledImage,
                                     Value coordinates, Value lod) {
    if (!lod)
        return spirv::ImageSampleImplicitLodOp::create(builder, location, resultType, sampledImage, coordinates,
                                                       spirv::ImageOperandsAttr(), ValueRange{});
    auto imageOperands = spirv::ImageOperandsAttr::get(builder.getContext(), spirv::ImageOperands::Lod);
    return spirv::ImageSampleExplicitLodOp::create(builder, location, resultType, sampledImage, coordinates,
                                                   imageOperands, ValueRange{lod});
}

// Implicit derivatives with the level they select offset by `bias` levels, so only fragment stages have it.
inline Value buildSpirvTextureSampleBias(OpBuilder &builder, Location location, Type resultType, Value sampledImage,
                                         Value coordinates, Value bias) {
    auto imageOperands = spirv::ImageOperandsAttr::get(builder.getContext(), spirv::ImageOperands::Bias);
    return spirv::ImageSampleImplicitLodOp::create(builder, location, resultType, sampledImage, coordinates,
                                                   imageOperands, ValueRange{bias});
}

/* A storage image is read directly; a sampled one is fetched through the image of its combined image-sampler. The
 * fetch has no Lod operand, so it reads the base level: MLIR's verifier rejects a Lod on spirv.ImageFetch because the
 * op does not implement FetchOpInterface. */
inline Value buildSpirvTextureLoad(OpBuilder &builder, Location location, Type resultType, Value image,
                                   Value coordinates) {
    auto sampledImageType = dyn_cast<spirv::SampledImageType>(image.getType());
    if (!sampledImageType) {
        Type imageTexelType = spirvImageTexelType(cast<spirv::ImageType>(image.getType()), resultType);
        Value texel = spirv::ImageReadOp::create(builder, location, imageTexelType, image, coordinates,
                                                 spirv::ImageOperandsAttr(), ValueRange{});
        return imageTexelType == resultType ? texel : spirv::BitcastOp::create(builder, location, resultType, texel);
    }
    Value texels = spirv::ImageOp::create(builder, location, sampledImageType.getImageType(), image);
    return spirv::ImageFetchOp::create(builder, location, resultType, texels, coordinates, spirv::ImageOperandsAttr(),
                                       ValueRange{});
}

} // namespace mlir::vernon
