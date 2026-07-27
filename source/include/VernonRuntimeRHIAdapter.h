#ifndef VERNON_RUNTIME_RHI_ADAPTER_H
#define VERNON_RUNTIME_RHI_ADAPTER_H

#include "VernonRHI.h"
#include "VernonRuntimeProvider.h"

#if defined(VERNON_RUNTIME_RHI_ADAPTER_STATIC)
#define VERNON_RUNTIME_RHI_ADAPTER_CAPI
#elif defined(_WIN32) && defined(VERNON_RUNTIME_RHI_ADAPTER_BUILD)
#define VERNON_RUNTIME_RHI_ADAPTER_CAPI __declspec(dllexport)
#elif defined(_WIN32)
#define VERNON_RUNTIME_RHI_ADAPTER_CAPI __declspec(dllimport)
#else
#define VERNON_RUNTIME_RHI_ADAPTER_CAPI
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VernonRuntimeRhiAdapter VernonRuntimeRhiAdapter;

VERNON_RUNTIME_RHI_ADAPTER_CAPI VernonRuntimeRhiAdapter *vernonRuntimeRhiAdapterCreateCuda(uint32_t device_index);
VERNON_RUNTIME_RHI_ADAPTER_CAPI VernonRuntimeRhiAdapter *
vernonRuntimeRhiAdapterCreateForDevice(VernonRhiDevice device, VernonRhiBackend backend);
VERNON_RUNTIME_RHI_ADAPTER_CAPI void vernonRuntimeRhiAdapterDestroy(VernonRuntimeRhiAdapter *adapter);
VERNON_RUNTIME_RHI_ADAPTER_CAPI const VernonRuntimeDeviceProvider *
vernonRuntimeRhiAdapterGetProvider(VernonRuntimeRhiAdapter *adapter);
VERNON_RUNTIME_RHI_ADAPTER_CAPI VernonStatus vernonRuntimeRhiAdapterSynchronize(VernonRuntimeRhiAdapter *adapter);
VERNON_RUNTIME_RHI_ADAPTER_CAPI void vernonRuntimeRhiAdapterInvalidateState(VernonRuntimeRhiAdapter *adapter);
VERNON_RUNTIME_RHI_ADAPTER_CAPI VernonStringView
vernonRuntimeRhiAdapterGetLastError(const VernonRuntimeRhiAdapter *adapter);
VERNON_RUNTIME_RHI_ADAPTER_CAPI VernonStatus
vernonRuntimeRhiAdapterReferenceBuffer(const VernonRuntimeRhiAdapter *adapter, VernonRhiBuffer buffer, uint64_t offset,
                                       uint64_t size, VernonRuntimeProviderResourceReference *output);
VERNON_RUNTIME_RHI_ADAPTER_CAPI VernonStatus vernonRuntimeRhiAdapterReferenceImage(
    const VernonRuntimeRhiAdapter *adapter, VernonRhiImage image, VernonRuntimeProviderResourceReference *output);
VERNON_RUNTIME_RHI_ADAPTER_CAPI VernonStatus vernonRuntimeRhiAdapterReferenceSampler(
    const VernonRuntimeRhiAdapter *adapter, VernonRhiSampler sampler, VernonRuntimeProviderResourceReference *output);

#ifdef __cplusplus
}
#endif

#endif
