#ifndef VERNON_TESTS_SUPPORT_RUNTIME_CORE_PACKETS_H
#define VERNON_TESTS_SUPPORT_RUNTIME_CORE_PACKETS_H

#include "VernonRuntimeCore.h"

#include <cstddef>
#include <cstdint>

namespace vernon::tests {

// Encodes what `packets` recorded as one batch with its recorded payload, then resets the recording.
inline VernonStatus encodeRecordedPackets(VernonRuntimeCoreCommands *commands, VernonRuntimeCorePackets *packets) {
    VernonRuntimeCorePacketBatch *batch = nullptr;
    VernonStatus status = vernonRuntimeCorePacketBatchCreate(packets, &batch);
    if (status == VERNON_STATUS_OK) {
        size_t wordCount = 0;
        const uint32_t *payload = vernonRuntimeCorePacketsGetPayload(packets, &wordCount);
        status = vernonRuntimeCoreEncodePackets(commands, batch, payload, wordCount);
    }
    vernonRuntimeCorePacketBatchDestroy(batch);
    vernonRuntimeCorePacketsReset(packets);
    return status;
}

} // namespace vernon::tests

#endif