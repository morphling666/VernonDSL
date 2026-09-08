#ifndef VERNON_CPU_ABI_WRAPPER_H
#define VERNON_CPU_ABI_WRAPPER_H

#include <cstdint>
#include <string>
#include <vector>

namespace llvm {
class Error;
class Module;
} // namespace llvm

namespace vernon {

enum class CpuAbiArgumentKind {
    CanonicalValue,
    OpaqueScalar,
    TensorView,
};

struct CpuCallLanePacking {
    uint64_t offset;
    uint64_t size;
};

struct CpuAbiArgumentPacking {
    uint64_t offset;
    uint64_t size;
    CpuAbiArgumentKind kind;
    uint32_t tensorRank;
    std::string builtin;
    std::vector<uint64_t> tensorLeafElementSizes;
    std::vector<CpuCallLanePacking> callLanes;
};

struct CpuAbiWrapperMetadata {
    std::string internalFunctionSymbol;
    std::vector<CpuAbiArgumentPacking> sourceArguments;
    uint64_t argumentsSize;
    uint64_t resultsSize;
    std::vector<CpuCallLanePacking> resultCallLanes;
    bool requiresTextureCallbacks;
    bool requiresPhases;
    std::string exportedWrapperSymbol;
};

llvm::Error defineCpuTextureSampleHelper(llvm::Module &module);
llvm::Error emitCpuAbiWrapper(llvm::Module &module, const CpuAbiWrapperMetadata &metadata);

} // namespace vernon

#endif
