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
    // Where each TensorView's innermost stride sits in the packed arguments; a range whose strides are all 1 runs a
    // copy of the lane loop that knows them, so its accesses are consecutive.
    std::vector<CpuCallLanePacking> innermostStrides;
};

llvm::Error defineCpuTextureSampleHelper(llvm::Module &module);
// Emits the entry's range wrapper and, when the kernel qualifies (VernonCpuChunk), its chunk entry.
llvm::Error emitCpuAbiWrapper(llvm::Module &module, const CpuAbiWrapperMetadata &metadata);
inline std::string cpuChunkEntrySymbol(const CpuAbiWrapperMetadata &metadata) {
    return metadata.exportedWrapperSymbol + "_chunk";
}
inline constexpr const char *cpuStaticRegistrationSymbol = "vernonRuntimeRegisterStaticCpuEntries";
inline constexpr const char *cpuStaticUnregistrationSymbol = "vernonRuntimeUnregisterStaticCpuEntries";

// Emits a static constructor that registers every entry wrapper with the
// Runtime and a static destructor that withdraws them. A JIT that loads such
// an object must bind both symbols itself, because JIT entries are owned by
// the compile result and must never reach the process-wide registry.
llvm::Error emitCpuStaticRegistration(llvm::Module &module, const std::vector<CpuAbiWrapperMetadata> &entries);

} // namespace vernon

#endif
