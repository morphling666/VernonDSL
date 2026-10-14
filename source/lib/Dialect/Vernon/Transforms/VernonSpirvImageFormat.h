#pragma once

#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace mlir::vernon {

inline std::optional<spirv::ImageFormat> spirvStorageImageFormat(llvm::StringRef format) {
#define VERNON_FORMAT(name, NAME, rhi, texture, cls, bytes, dataFormat, dataType, caps, glInternal, glFormat, glType,  \
                      vulkan, dxgi, metal, spirvFormat)                                                                \
    if (format == #name && spirv::ImageFormat::spirvFormat != spirv::ImageFormat::Unknown)                             \
        return spirv::ImageFormat::spirvFormat;
#include "VernonFormats.def"
    return std::nullopt;
}

} // namespace mlir::vernon
