#include "VernonExecutionGraph.h"
#include "VernonRuntimeRHIAdapter.h"

#include <gtest/gtest.h>

namespace {

struct BackendCase {
    VernonRhiBackend backend;
    const char *name;
    bool supportsImages;
};

class RhiResourceLifetime : public testing::TestWithParam<BackendCase> {};

TEST_P(RhiResourceLifetime, ExecutionGraphDestroysOwnedBuffersButNotImportedBuffers) {
    const BackendCase test = GetParam();
    VernonRhiOwnedDeviceDescriptor deviceDescriptor{};
    deviceDescriptor.struct_size = sizeof(deviceDescriptor);
    deviceDescriptor.backend = test.backend;
    const VernonRhiDevice device = vernonRhiCreateDevice(&deviceDescriptor);
    if (device.index == VERNON_RHI_INVALID_HANDLE_INDEX)
        GTEST_SKIP() << test.name << " is unavailable";

    VernonRhiBufferDescriptor bufferDescriptor{};
    bufferDescriptor.struct_size = sizeof(bufferDescriptor);
    bufferDescriptor.size = 64;
    bufferDescriptor.usage = VERNON_RHI_BUFFER_STORAGE;
    bufferDescriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;

    VernonRhiBuffer imported{};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &imported), VERNON_RHI_STATUS_OK);
    {
        vernon::execution::ExecutionGraph graph(device);
        const vernon::execution::GraphBuffer first = graph.importBuffer(imported);
        const vernon::execution::GraphBuffer second = graph.importBuffer(imported);
        EXPECT_EQ(first.id, second.id);
    }
    EXPECT_EQ(vernonRhiDeviceIsBufferValid(device, imported), 1u);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, imported), VERNON_RHI_STATUS_OK);

    VernonRhiBuffer owned{};
    {
        vernon::execution::ExecutionGraph graph(device);
        vernon::execution::GraphBuffer graphBuffer;
        ASSERT_EQ(graph.createBuffer(bufferDescriptor, graphBuffer), VERNON_RHI_STATUS_OK);
        owned = graphBuffer.handle;
        const vernon::execution::GraphBuffer duplicate = graph.importBuffer(owned);
        EXPECT_EQ(graphBuffer.id, duplicate.id);
        EXPECT_EQ(vernonRhiDeviceIsBufferValid(device, owned), 1u);
    }
    EXPECT_EQ(vernonRhiDeviceIsBufferValid(device, owned), 0u);

    VernonRhiBuffer recycled{};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &recycled), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(recycled.index, owned.index);
    EXPECT_NE(recycled.generation, owned.generation);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, recycled), VERNON_RHI_STATUS_OK);
    vernonRhiDestroyDevice(device);
}

TEST_P(RhiResourceLifetime, UnsupportedResourceKindsAreRejected) {
    const BackendCase test = GetParam();
    if (test.supportsImages)
        GTEST_SKIP() << test.name << " exposes images and samplers";

    VernonRhiOwnedDeviceDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.backend = test.backend;
    const VernonRhiDevice device = vernonRhiCreateDevice(&descriptor);
    if (device.index == VERNON_RHI_INVALID_HANDLE_INDEX)
        GTEST_SKIP() << test.name << " is unavailable";

    VernonRhiImageDescriptor imageDescriptor{};
    imageDescriptor.struct_size = sizeof(imageDescriptor);
    imageDescriptor.dimension = VERNON_RHI_IMAGE_2D;
    imageDescriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    imageDescriptor.width = 1;
    imageDescriptor.height = 1;
    imageDescriptor.depth = 1;
    imageDescriptor.mip_levels = 1;
    imageDescriptor.array_layers = 1;
    imageDescriptor.sample_count = 1;
    imageDescriptor.usage = VERNON_RHI_IMAGE_SAMPLED;
    VernonRhiImage image{};
    EXPECT_EQ(vernonRhiDeviceCreateImage(device, &imageDescriptor, &image), VERNON_RHI_STATUS_UNSUPPORTED);

    VernonRhiSamplerDescriptor samplerDescriptor{};
    samplerDescriptor.struct_size = sizeof(samplerDescriptor);
    VernonRhiSampler sampler{};
    EXPECT_EQ(vernonRhiDeviceCreateSampler(device, &samplerDescriptor, &sampler), VERNON_RHI_STATUS_UNSUPPORTED);
    vernonRhiDestroyDevice(device);
}

TEST_P(RhiResourceLifetime, RetainedBufferDelaysSlotReuse) {
    const BackendCase test = GetParam();
    VernonRhiOwnedDeviceDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.backend = test.backend;
    const VernonRhiDevice device = vernonRhiCreateDevice(&descriptor);
    if (device.index == VERNON_RHI_INVALID_HANDLE_INDEX)
        GTEST_SKIP() << test.name << " is unavailable";

    VernonRuntimeRhiAdapter *adapter = vernonRuntimeRhiAdapterCreateForDevice(device, test.backend);
    ASSERT_NE(adapter, nullptr);
    const VernonRuntimeDeviceProvider *provider = vernonRuntimeRhiAdapterGetProvider(adapter);
    ASSERT_NE(provider, nullptr);

    VernonRhiBufferDescriptor bufferDescriptor{};
    bufferDescriptor.struct_size = sizeof(bufferDescriptor);
    bufferDescriptor.size = 64;
    bufferDescriptor.usage = VERNON_RHI_BUFFER_STORAGE;
    bufferDescriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
    VernonRhiBuffer first{};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &first), VERNON_RHI_STATUS_OK);
    VernonRuntimeProviderResourceReference reference{};
    ASSERT_EQ(vernonRuntimeRhiAdapterReferenceBuffer(adapter, first, 0, 64, &reference), VERNON_STATUS_OK);
    ASSERT_EQ(provider->retain_resource(provider->user_data, reference), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyBuffer(device, first), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceIsBufferValid(device, first), 0u);

    VernonRhiBuffer replacement{};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &replacement), VERNON_RHI_STATUS_OK);
    EXPECT_NE(replacement.index, first.index);
    provider->release_resource(provider->user_data, reference);

    VernonRhiBuffer recycled{};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &recycled), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(recycled.index, first.index);
    EXPECT_NE(recycled.generation, first.generation);

    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, replacement), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, recycled), VERNON_RHI_STATUS_OK);
    vernonRuntimeRhiAdapterDestroy(adapter);
    vernonRhiDestroyDevice(device);
}

TEST_P(RhiResourceLifetime, RetainedImageAndSamplerDelaySlotReuse) {
    const BackendCase test = GetParam();
    if (!test.supportsImages)
        GTEST_SKIP() << test.name << " does not expose images or samplers";

    VernonRhiOwnedDeviceDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.backend = test.backend;
    const VernonRhiDevice device = vernonRhiCreateDevice(&descriptor);
    if (device.index == VERNON_RHI_INVALID_HANDLE_INDEX)
        GTEST_SKIP() << test.name << " is unavailable";

    VernonRuntimeRhiAdapter *adapter = vernonRuntimeRhiAdapterCreateForDevice(device, test.backend);
    ASSERT_NE(adapter, nullptr);
    const VernonRuntimeDeviceProvider *provider = vernonRuntimeRhiAdapterGetProvider(adapter);
    ASSERT_NE(provider, nullptr);

    VernonRhiImageDescriptor imageDescriptor{};
    imageDescriptor.struct_size = sizeof(imageDescriptor);
    imageDescriptor.dimension = VERNON_RHI_IMAGE_2D;
    imageDescriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    imageDescriptor.width = 1;
    imageDescriptor.height = 1;
    imageDescriptor.depth = 1;
    imageDescriptor.mip_levels = 1;
    imageDescriptor.array_layers = 1;
    imageDescriptor.sample_count = 1;
    imageDescriptor.usage = VERNON_RHI_IMAGE_SAMPLED;
    VernonRhiImage firstImage{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &imageDescriptor, &firstImage), VERNON_RHI_STATUS_OK);
    VernonRuntimeProviderResourceReference imageReference{};
    ASSERT_EQ(vernonRuntimeRhiAdapterReferenceImage(adapter, firstImage, &imageReference), VERNON_STATUS_OK);
    ASSERT_EQ(provider->retain_resource(provider->user_data, imageReference), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(device, firstImage), VERNON_RHI_STATUS_OK);
    VernonRhiImage replacementImage{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &imageDescriptor, &replacementImage), VERNON_RHI_STATUS_OK);
    EXPECT_NE(replacementImage.index, firstImage.index);
    provider->release_resource(provider->user_data, imageReference);
    VernonRhiImage recycledImage{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &imageDescriptor, &recycledImage), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(recycledImage.index, firstImage.index);
    EXPECT_NE(recycledImage.generation, firstImage.generation);

    VernonRhiSamplerDescriptor samplerDescriptor{};
    samplerDescriptor.struct_size = sizeof(samplerDescriptor);
    VernonRhiSampler firstSampler{};
    ASSERT_EQ(vernonRhiDeviceCreateSampler(device, &samplerDescriptor, &firstSampler), VERNON_RHI_STATUS_OK);
    VernonRuntimeProviderResourceReference samplerReference{};
    ASSERT_EQ(vernonRuntimeRhiAdapterReferenceSampler(adapter, firstSampler, &samplerReference), VERNON_STATUS_OK);
    ASSERT_EQ(provider->retain_resource(provider->user_data, samplerReference), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroySampler(device, firstSampler), VERNON_RHI_STATUS_OK);
    VernonRhiSampler replacementSampler{};
    ASSERT_EQ(vernonRhiDeviceCreateSampler(device, &samplerDescriptor, &replacementSampler), VERNON_RHI_STATUS_OK);
    EXPECT_NE(replacementSampler.index, firstSampler.index);
    provider->release_resource(provider->user_data, samplerReference);
    VernonRhiSampler recycledSampler{};
    ASSERT_EQ(vernonRhiDeviceCreateSampler(device, &samplerDescriptor, &recycledSampler), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(recycledSampler.index, firstSampler.index);
    EXPECT_NE(recycledSampler.generation, firstSampler.generation);

    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, replacementImage), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, recycledImage), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroySampler(device, replacementSampler), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroySampler(device, recycledSampler), VERNON_RHI_STATUS_OK);
    vernonRuntimeRhiAdapterDestroy(adapter);
    vernonRhiDestroyDevice(device);
}

TEST_P(RhiResourceLifetime, RetainedImageViewKeepsParentDescriptorAlive) {
    const BackendCase test = GetParam();
    if (!test.supportsImages)
        GTEST_SKIP() << test.name << " does not expose images";

    VernonRhiOwnedDeviceDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.backend = test.backend;
    const VernonRhiDevice device = vernonRhiCreateDevice(&descriptor);
    if (device.index == VERNON_RHI_INVALID_HANDLE_INDEX)
        GTEST_SKIP() << test.name << " is unavailable";
    VernonRuntimeRhiAdapter *adapter = vernonRuntimeRhiAdapterCreateForDevice(device, test.backend);
    ASSERT_NE(adapter, nullptr);
    const VernonRuntimeDeviceProvider *provider = vernonRuntimeRhiAdapterGetProvider(adapter);

    VernonRhiImageDescriptor imageDescriptor{};
    imageDescriptor.struct_size = sizeof(imageDescriptor);
    imageDescriptor.dimension = VERNON_RHI_IMAGE_2D;
    imageDescriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    imageDescriptor.width = imageDescriptor.height = 4;
    imageDescriptor.depth = 1;
    imageDescriptor.mip_levels = 2;
    imageDescriptor.array_layers = 1;
    imageDescriptor.sample_count = 1;
    imageDescriptor.usage = VERNON_RHI_IMAGE_SAMPLED;
    VernonRhiImage image{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &imageDescriptor, &image), VERNON_RHI_STATUS_OK);
    VernonRhiImageViewDescriptor viewDescriptor{};
    viewDescriptor.struct_size = sizeof(viewDescriptor);
    viewDescriptor.image = image;
    viewDescriptor.dimension = VERNON_RHI_IMAGE_2D;
    viewDescriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    viewDescriptor.base_mip_level = 1;
    viewDescriptor.mip_level_count = 1;
    viewDescriptor.array_layer_count = 1;
    viewDescriptor.aspects = VERNON_RHI_IMAGE_ASPECT_COLOR;
    VernonRhiImageView view{};
    ASSERT_EQ(vernonRhiDeviceCreateImageView(device, &viewDescriptor, &view), VERNON_RHI_STATUS_OK);
    VernonRuntimeProviderResourceReference reference{};
    ASSERT_EQ(vernonRuntimeRhiAdapterReferenceImageView(adapter, view, &reference), VERNON_STATUS_OK);
    ASSERT_EQ(provider->retain_resource(provider->user_data, reference), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImageView(device, view), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(device, image), VERNON_RHI_STATUS_OK);

    VernonRuntimeProviderImageDescription description{};
    description.struct_size = sizeof(description);
    ASSERT_EQ(provider->describe_image(provider->user_data, reference, &description), VERNON_STATUS_OK);
    EXPECT_NE(description.parent_identity, 0u);
    EXPECT_EQ(description.view.subresources.base_mip_level, 1u);
    EXPECT_EQ(description.view.subresources.mip_level_count, 1u);

    VernonRhiImage replacement{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &imageDescriptor, &replacement), VERNON_RHI_STATUS_OK);
    EXPECT_NE(replacement.index, image.index);
    provider->release_resource(provider->user_data, reference);
    VernonRhiImage recycled{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &imageDescriptor, &recycled), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(recycled.index, image.index);
    EXPECT_NE(recycled.generation, image.generation);

    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, replacement), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, recycled), VERNON_RHI_STATUS_OK);
    vernonRuntimeRhiAdapterDestroy(adapter);
    vernonRhiDestroyDevice(device);
}

TEST_P(RhiResourceLifetime, ExecutionGraphRetainsImportedImageViewAndParent) {
    const BackendCase test = GetParam();
    if (!test.supportsImages)
        GTEST_SKIP() << test.name << " does not expose images";

    VernonRhiOwnedDeviceDescriptor deviceDescriptor{};
    deviceDescriptor.struct_size = sizeof(deviceDescriptor);
    deviceDescriptor.backend = test.backend;
    const VernonRhiDevice device = vernonRhiCreateDevice(&deviceDescriptor);
    if (device.index == VERNON_RHI_INVALID_HANDLE_INDEX)
        GTEST_SKIP() << test.name << " is unavailable";

    VernonRhiImageDescriptor imageDescriptor{};
    imageDescriptor.struct_size = sizeof(imageDescriptor);
    imageDescriptor.dimension = VERNON_RHI_IMAGE_2D;
    imageDescriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    imageDescriptor.width = imageDescriptor.height = 4;
    imageDescriptor.depth = 1;
    imageDescriptor.mip_levels = 1;
    imageDescriptor.array_layers = 1;
    imageDescriptor.sample_count = 1;
    imageDescriptor.usage = VERNON_RHI_IMAGE_SAMPLED;
    VernonRhiImage image{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &imageDescriptor, &image), VERNON_RHI_STATUS_OK);

    VernonRhiImageViewDescriptor viewDescriptor{};
    viewDescriptor.struct_size = sizeof(viewDescriptor);
    viewDescriptor.image = image;
    viewDescriptor.dimension = VERNON_RHI_IMAGE_2D;
    viewDescriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    viewDescriptor.mip_level_count = 1;
    viewDescriptor.array_layer_count = 1;
    viewDescriptor.aspects = VERNON_RHI_IMAGE_ASPECT_COLOR;
    VernonRhiImageView view{};
    ASSERT_EQ(vernonRhiDeviceCreateImageView(device, &viewDescriptor, &view), VERNON_RHI_STATUS_OK);

    VernonRhiImage replacement{};
    {
        vernon::execution::ExecutionGraph graph(device);
        const vernon::execution::GraphImage imported = graph.importImage(image, view);
        ASSERT_NE(imported.id, UINT32_MAX);
        ASSERT_EQ(vernonRhiDeviceDestroyImageView(device, view), VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiDeviceDestroyImage(device, image), VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiDeviceCreateImage(device, &imageDescriptor, &replacement), VERNON_RHI_STATUS_OK);
        EXPECT_NE(replacement.index, image.index);
    }

    VernonRhiImage recycled{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &imageDescriptor, &recycled), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(recycled.index, image.index);
    EXPECT_NE(recycled.generation, image.generation);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, replacement), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, recycled), VERNON_RHI_STATUS_OK);
    vernonRhiDestroyDevice(device);
}

INSTANTIATE_TEST_SUITE_P(GpuBackends, RhiResourceLifetime,
                         testing::Values(BackendCase{VERNON_RHI_BACKEND_CUDA, "CUDA", false},
                                         BackendCase{VERNON_RHI_BACKEND_VULKAN, "Vulkan", true},
                                         BackendCase{VERNON_RHI_BACKEND_DIRECTX12, "DirectX12", true},
                                         BackendCase{VERNON_RHI_BACKEND_METAL, "Metal", true}),
                         [](const testing::TestParamInfo<BackendCase> &info) { return info.param.name; });

} // namespace
