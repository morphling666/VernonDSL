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
    MetadataCarrier,
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
inline constexpr const char *cpuStaticRegistrationSymbol = "vernonRuntimeRegisterStaticCpuEntries";
inline constexpr const char *cpuStaticUnregistrationSymbol = "vernonRuntimeUnregisterStaticCpuEntries";

// Emits a static constructor that registers every entry wrapper with the
// Runtime and a static destructor that withdraws them. A JIT that loads such
// an object must bind both symbols itself, because JIT entries are owned by
// the compile result and must never reach the process-wide registry.
llvm::Error emitCpuStaticRegistration(llvm::Module &module, const std::vector<CpuAbiWrapperMetadata> &entries);

} // namespace vernon

#endif
