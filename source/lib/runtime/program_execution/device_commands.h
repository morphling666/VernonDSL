#ifndef VERNON_RUNTIME_PROGRAM_EXECUTION_DEVICE_COMMANDS_H
#define VERNON_RUNTIME_PROGRAM_EXECUTION_DEVICE_COMMANDS_H

#include "VernonRuntime.h"
#include "execution_graph/execution_command_model.h"
#include "runtime/resolved_stage_types.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

struct VernonRuntimeContext;

namespace vernon::execution::detail {
struct RhiCommandExecutionPlan;
} // namespace vernon::execution::detail

namespace vernon::runtime::program_execution {

struct DeviceBufferCopy {
    VernonRhiBuffer source;
    VernonRhiBuffer destination;
    size_t sourceOffset{};
    size_t destinationOffset{};
    size_t size{};
};

struct DeviceBufferUpload {
    VernonRhiBuffer destination;
    size_t destinationOffset{};
    const void *source{};
    size_t size{};
};

inline bool checkedDeviceBufferOffset(uint64_t base, size_t relative, size_t &result) {
    if (base > std::numeric_limits<size_t>::max())
        return false;
    const size_t converted = static_cast<size_t>(base);
    if (relative > std::numeric_limits<size_t>::max() - converted)
        return false;
    result = converted + relative;
    return true;
}

struct DeviceImageCopy {
    VernonRhiImage source;
    VernonRhiImage destination;
    std::vector<VernonRhiImageCopyRegion> regions;
};

bool encodeBufferCopies(VernonRuntimeContext &context, VernonRhiCommandEncoder encoder,
                        const std::vector<DeviceBufferCopy> &copies);
bool encodeImageCopies(VernonRuntimeContext &context, VernonRhiCommandEncoder encoder,
                       const std::vector<DeviceImageCopy> &copies);
VernonStatus buildDeviceTransferCommandPlan(VernonRuntimeContext &context,
                                            const std::vector<DeviceBufferCopy> &bufferCopies,
                                            const std::vector<DeviceImageCopy> &imageCopies,
                                            const std::vector<DeviceBufferUpload> &uploads,
                                            execution::detail::RhiCommandExecutionPlan &plan);
VernonStatus buildBufferTransferCommandPlan(VernonRuntimeContext &context, const std::vector<DeviceBufferCopy> &copies,
                                            const std::vector<DeviceBufferUpload> &uploads,
                                            execution::detail::RhiCommandExecutionPlan &plan);
VernonStatus encodePipelineCommand(VernonRuntimeContext &context, VernonRhiCommandEncoder encoder,
                                   VernonStageExecutable &pipeline, std::vector<VernonProgramArgument> &arguments,
                                   VernonLaunchSize grid);
VernonStatus buildPipelineCommandPlan(VernonRuntimeContext &context, const std::vector<DeviceBufferCopy> &copiesBefore,
                                      const std::vector<DeviceBufferUpload> &uploadsBefore,
                                      VernonStageExecutable &pipeline,
                                      const std::vector<VernonProgramArgument> &arguments, VernonLaunchSize grid,
                                      const std::vector<DeviceBufferCopy> &copiesAfter,
                                      execution::detail::CommandNodeKind kind,
                                      execution::detail::RhiCommandExecutionPlan &plan);

} // namespace vernon::runtime::program_execution

#endif
