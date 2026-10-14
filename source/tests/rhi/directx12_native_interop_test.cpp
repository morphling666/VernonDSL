#include "VernonRHI.h"
#include "rhi/directx12_backend.h"
#include "rhi/rhi_internal.h"
#include "runtime_rhi_test_utils.h"

#include <d3d12.h>
#include <dxgi1_6.h>
#include <gtest/gtest.h>
#include <wrl/client.h>

#include <array>
#include <vector>

namespace {

using Microsoft::WRL::ComPtr;

D3D12_HEAP_PROPERTIES defaultHeap() {
    D3D12_HEAP_PROPERTIES result{};
    result.Type = D3D12_HEAP_TYPE_DEFAULT;
    result.CreationNodeMask = 1;
    result.VisibleNodeMask = 1;
    return result;
}

TEST(DirectX12NativeInterop, BorrowsObjectsWithoutChangingTheirLifetime) {
    ComPtr<IDXGIFactory6> factory;
    ComPtr<IDXGIAdapter> adapter;
    ComPtr<ID3D12Device> nativeDevice;
    ASSERT_TRUE(SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))));
    ASSERT_TRUE(SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter))));
    ASSERT_TRUE(SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&nativeDevice))));

    D3D12_COMMAND_QUEUE_DESC queueDescription{};
    queueDescription.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> commands;
    ASSERT_TRUE(SUCCEEDED(nativeDevice->CreateCommandQueue(&queueDescription, IID_PPV_ARGS(&queue))));
    ASSERT_TRUE(
        SUCCEEDED(nativeDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))));
    ASSERT_TRUE(SUCCEEDED(nativeDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                                                          IID_PPV_ARGS(&commands))));

    D3D12_RESOURCE_DESC bufferDescription{};
    bufferDescription.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufferDescription.Width = 256;
    bufferDescription.Height = 1;
    bufferDescription.DepthOrArraySize = 1;
    bufferDescription.MipLevels = 1;
    bufferDescription.SampleDesc.Count = 1;
    bufferDescription.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    const D3D12_HEAP_PROPERTIES heap = defaultHeap();
    ComPtr<ID3D12Resource> buffer;
    ASSERT_TRUE(SUCCEEDED(nativeDevice->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_NONE, &bufferDescription, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&buffer))));

    D3D12_RESOURCE_DESC imageDescription{};
    imageDescription.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    imageDescription.Width = 4;
    imageDescription.Height = 4;
    imageDescription.DepthOrArraySize = 1;
    imageDescription.MipLevels = 1;
    imageDescription.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    imageDescription.SampleDesc.Count = 1;
    ComPtr<ID3D12Resource> image;
    ASSERT_TRUE(SUCCEEDED(nativeDevice->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_NONE, &imageDescription, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&image))));

    D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDescription{};
    descriptorHeapDescription.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    descriptorHeapDescription.NumDescriptors = 4;
    descriptorHeapDescription.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ComPtr<ID3D12DescriptorHeap> descriptorHeap;
    ASSERT_TRUE(
        SUCCEEDED(nativeDevice->CreateDescriptorHeap(&descriptorHeapDescription, IID_PPV_ARGS(&descriptorHeap))));

    const ULONG deviceReferences = nativeDevice->AddRef();
    nativeDevice->Release();
    const ULONG queueReferences = queue->AddRef();
    queue->Release();
    const ULONG commandReferences = commands->AddRef();
    commands->Release();
    const ULONG bufferReferences = buffer->AddRef();
    buffer->Release();
    const ULONG imageReferences = image->AddRef();
    image->Release();
    const ULONG heapReferences = descriptorHeap->AddRef();
    descriptorHeap->Release();

    VernonRhiDirectX12BorrowedDeviceDescriptor deviceDescriptor{};
    deviceDescriptor.struct_size = sizeof(deviceDescriptor);
    deviceDescriptor.device = nativeDevice.Get();
    deviceDescriptor.queue = queue.Get();
    deviceDescriptor.command_list = commands.Get();
    deviceDescriptor.queue_capabilities =
        VERNON_RHI_QUEUE_TRANSFER | VERNON_RHI_QUEUE_COMPUTE | VERNON_RHI_QUEUE_GRAPHICS;
    const VernonRhiDevice device = vernonRhiCreateBorrowedDirectX12Device(&deviceDescriptor);
    ASSERT_NE(device.index, VERNON_RHI_INVALID_HANDLE_INDEX);

    VernonRhiDirectX12BorrowedBufferDescriptor bufferDescriptor{};
    bufferDescriptor.struct_size = sizeof(bufferDescriptor);
    bufferDescriptor.resource = buffer.Get();
    bufferDescriptor.size = bufferDescription.Width;
    bufferDescriptor.usage = VERNON_RHI_BUFFER_STORAGE;
    bufferDescriptor.state = VERNON_RHI_STATE_COMMON;
    VernonRhiBuffer importedBuffer{};
    ASSERT_EQ(vernonRhiDirectX12DeviceImportBorrowedBuffer(device, &bufferDescriptor, &importedBuffer),
              VERNON_RHI_STATUS_OK);

    VernonRhiDirectX12BorrowedImageDescriptor imageImport{};
    imageImport.struct_size = sizeof(imageImport);
    imageImport.resource = image.Get();
    imageImport.image.struct_size = sizeof(imageImport.image);
    imageImport.image.dimension = VERNON_RHI_IMAGE_2D;
    imageImport.image.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    imageImport.image.width = 4;
    imageImport.image.height = 4;
    imageImport.image.depth = 1;
    imageImport.image.mip_levels = 1;
    imageImport.image.array_layers = 1;
    imageImport.image.sample_count = 1;
    imageImport.image.usage = VERNON_RHI_IMAGE_SAMPLED;
    imageImport.state = VERNON_RHI_STATE_COMMON;
    VernonRhiImage importedImage{};
    ASSERT_EQ(vernonRhiDirectX12DeviceImportBorrowedImage(device, &imageImport, &importedImage), VERNON_RHI_STATUS_OK);

    VernonRhiDirectX12BorrowedDescriptorRangeDescriptor rangeDescriptor{};
    rangeDescriptor.struct_size = sizeof(rangeDescriptor);
    rangeDescriptor.heap = descriptorHeap.Get();
    rangeDescriptor.cpu_handle = descriptorHeap->GetCPUDescriptorHandleForHeapStart().ptr;
    rangeDescriptor.gpu_handle = descriptorHeap->GetGPUDescriptorHandleForHeapStart().ptr;
    rangeDescriptor.descriptor_count = descriptorHeapDescription.NumDescriptors;
    rangeDescriptor.heap_type = VERNON_RHI_NATIVE_DESCRIPTOR_RESOURCE;
    VernonRhiNativeDescriptorRange importedRange{};
    ASSERT_EQ(vernonRhiDirectX12DeviceImportBorrowedDescriptorRange(device, &rangeDescriptor, &importedRange),
              VERNON_RHI_STATUS_OK);

    void *native = nullptr;
    EXPECT_EQ(vernonRhiDirectX12DeviceGetBorrowedQueue(device, &native), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(native, queue.Get());
    EXPECT_EQ(vernonRhiDirectX12DeviceGetBorrowedCommandList(device, &native), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(native, commands.Get());
    EXPECT_EQ(vernonRhiDeviceGetBufferNativeHandle(device, importedBuffer, &native), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(native, buffer.Get());
    uint64_t nativeImage = 0;
    EXPECT_EQ(vernonRhiDeviceGetImageNativeHandle(device, importedImage, &nativeImage), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(reinterpret_cast<void *>(nativeImage), image.Get());
    EXPECT_EQ(vernon::tests::hostReadStatuses(device),
              (std::array<VernonRhiStatus, 2>{VERNON_RHI_STATUS_UNSUPPORTED, VERNON_RHI_STATUS_UNSUPPORTED}));

    VernonRhiCommandEncoderDescriptor encoderDescriptor{};
    encoderDescriptor.struct_size = sizeof(encoderDescriptor);
    VernonRhiCommandEncoder encoder{};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &encoderDescriptor, &encoder), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderFinish(device, encoder), VERNON_RHI_STATUS_OK);
    VernonRhiCompletion completion{};
    ASSERT_EQ(vernonRhiDeviceSubmit(device, encoder, &completion), VERNON_RHI_STATUS_OK);
    VernonRhiCompletionState completionState{};
    ASSERT_EQ(vernonRhiCompletionGetState(device, completion, &completionState), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(completionState, VERNON_RHI_COMPLETION_PENDING);
    ASSERT_EQ(vernonRhiCompletionSignal(device, completion, VERNON_RHI_STATUS_OK), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCompletionWait(device, completion), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyCompletion(device, completion), VERNON_RHI_STATUS_OK);

    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, importedBuffer), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, importedImage), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyNativeDescriptorRange(device, importedRange), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceGetBufferNativeHandle(device, importedBuffer, &native),
              VERNON_RHI_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(vernonRhiDeviceIsImageValid(device, importedImage), 0u);
    vernonRhiDestroyDevice(device);
    EXPECT_TRUE(SUCCEEDED(commands->Close()));

    EXPECT_EQ(nativeDevice->AddRef(), deviceReferences);
    nativeDevice->Release();
    EXPECT_EQ(queue->AddRef(), queueReferences);
    queue->Release();
    EXPECT_EQ(commands->AddRef(), commandReferences);
    commands->Release();
    EXPECT_EQ(buffer->AddRef(), bufferReferences);
    buffer->Release();
    EXPECT_EQ(image->AddRef(), imageReferences);
    image->Release();
    EXPECT_EQ(descriptorHeap->AddRef(), heapReferences);
    descriptorHeap->Release();
}

TEST(DirectX12NativeInterop, BorrowedStorageRetiresOnlyThroughTheCompletedSubmission) {
    std::string error;
    vernon::rhi::directx12::DeviceState owner;
    if (!owner.initialize(0, true, error))
        GTEST_SKIP() << error;
    ASSERT_TRUE(owner.beginCommands(error)) << error;
    {
        vernon::rhi::directx12::DeviceState borrowed;
        ASSERT_TRUE(borrowed.initializeBorrowed(owner.device, owner.queue, owner.commandList(), error)) << error;
        // Each request fills a whole chunk, so a request takes any free chunk before it creates one.
        const auto stage = [&] {
            ID3D12Resource *resource{};
            size_t offset{};
            uint8_t *mapped{};
            EXPECT_TRUE(borrowed.acquireStaging(
                true, vernon::rhi::StagingPool<vernon::rhi::directx12::DeviceState::StagingChunk>::minimumCapacity, 16,
                resource, offset, mapped, error))
                << error;
            return resource;
        };
        uint64_t first{};
        uint64_t second{};
        stage();
        ASSERT_TRUE(borrowed.submitCommands(first, error)) << error;
        ID3D12Resource *const unexecuted = stage();
        ASSERT_TRUE(borrowed.submitCommands(second, error)) << error;
        ASSERT_LT(first, second);

        borrowed.abandonCommands();
        borrowed.retireBorrowedCommands(first);
        EXPECT_NE(stage(), unexecuted);
        EXPECT_NE(stage(), unexecuted);
        borrowed.retireBorrowedCommands(second);
    }
    uint64_t value{};
    ASSERT_TRUE(owner.submitCommands(value, error)) << error;
    ASSERT_TRUE(owner.waitForValue(value, error)) << error;
}

TEST(DirectX12NativeInterop, KeepsOwnedResourceAddressesStableAsSlotsGrow) {
    VernonRhiOwnedDeviceDescriptor deviceDescriptor{};
    deviceDescriptor.struct_size = sizeof(deviceDescriptor);
    deviceDescriptor.backend = VERNON_RHI_BACKEND_DIRECTX12;
    const VernonRhiDevice device = vernonRhiCreateDevice(&deviceDescriptor);
    if (device.index == VERNON_RHI_INVALID_HANDLE_INDEX)
        GTEST_SKIP() << "DirectX 12 device is unavailable";

    VernonRhiBufferDescriptor bufferDescriptor{};
    bufferDescriptor.struct_size = sizeof(bufferDescriptor);
    bufferDescriptor.size = 64;
    bufferDescriptor.alignment = 16;
    bufferDescriptor.usage = VERNON_RHI_BUFFER_VERTEX;
    bufferDescriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;

    std::vector<VernonRhiBuffer> buffers(64);
    for (VernonRhiBuffer &buffer : buffers)
        ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &buffer), VERNON_RHI_STATUS_OK);

    auto firstResourceResult = vernon::rhi::bufferResource(device, buffers.front());
    ASSERT_TRUE(firstResourceResult.isOk());
    const uint64_t firstResource = firstResourceResult.value();
    for (size_t index = 0; index < 64; ++index) {
        VernonRhiBuffer extra{};
        ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &extra), VERNON_RHI_STATUS_OK);
        buffers.push_back(extra);
    }
    auto stableResource = vernon::rhi::bufferResource(device, buffers.front());
    ASSERT_TRUE(stableResource.isOk());
    EXPECT_EQ(stableResource.value(), firstResource);

    for (VernonRhiBuffer buffer : buffers)
        EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, buffer), VERNON_RHI_STATUS_OK);
    vernonRhiDestroyDevice(device);
}

TEST(DirectX12NativeInterop, ImageViewsTrackTheStatesOfTheirImage) {
    VernonRhiOwnedDeviceDescriptor deviceDescriptor{};
    deviceDescriptor.struct_size = sizeof(deviceDescriptor);
    deviceDescriptor.backend = VERNON_RHI_BACKEND_DIRECTX12;
    const VernonRhiDevice device = vernonRhiCreateDevice(&deviceDescriptor);
    if (device.index == VERNON_RHI_INVALID_HANDLE_INDEX)
        GTEST_SKIP() << "DirectX 12 device is unavailable";

    VernonRhiImageDescriptor imageDescriptor{};
    imageDescriptor.struct_size = sizeof(imageDescriptor);
    imageDescriptor.dimension = VERNON_RHI_IMAGE_2D;
    imageDescriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    imageDescriptor.width = 4;
    imageDescriptor.height = 4;
    imageDescriptor.depth = 1;
    imageDescriptor.mip_levels = 2;
    imageDescriptor.array_layers = 1;
    imageDescriptor.sample_count = 1;
    imageDescriptor.usage = VERNON_RHI_IMAGE_COLOR_ATTACHMENT | VERNON_RHI_IMAGE_SAMPLED;
    VernonRhiImage image{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(device, &imageDescriptor, &image), VERNON_RHI_STATUS_OK);
    const auto createView = [&](uint32_t baseMip) {
        VernonRhiImageViewDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.image = image;
        descriptor.dimension = VERNON_RHI_IMAGE_2D;
        descriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
        descriptor.base_mip_level = baseMip;
        descriptor.mip_level_count = 1;
        descriptor.array_layer_count = 1;
        descriptor.aspects = VERNON_RHI_IMAGE_ASPECT_COLOR;
        VernonRhiImageView view{};
        EXPECT_EQ(vernonRhiDeviceCreateImageView(device, &descriptor, &view), VERNON_RHI_STATUS_OK);
        return view;
    };
    const VernonRhiImageView target = createView(0);
    const VernonRhiImageView sampled = createView(1);

    VernonRhiCommandEncoderDescriptor encoderDescriptor{};
    encoderDescriptor.struct_size = sizeof(encoderDescriptor);
    encoderDescriptor.required_capabilities = VERNON_RHI_QUEUE_GRAPHICS;
    VernonRhiCommandEncoder encoder{};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &encoderDescriptor, &encoder), VERNON_RHI_STATUS_OK);
    const uint64_t key = vernon::rhi::commandEncoderKey(device, encoder).value();
    const uint64_t targetKey = vernon::rhi::imageViewResource(device, target).value();
    vernon::rhi::RenderingScope scope;
    scope.colorCount = 1;
    scope.colors[0].view = targetKey;
    scope.width = 4;
    scope.height = 4;
    scope.layers = 1;
    const vernon::rhi::RenderingRead read{vernon::rhi::ResourceKind::ImageView,
                                          vernon::rhi::imageViewResource(device, sampled).value()};
    ASSERT_TRUE(vernon::rhi::beginEncoderRendering(device, key, scope, &read, 1).isOk());
    ASSERT_TRUE(vernon::rhi::openCommandRendering(device, key).isOk());
    const auto *view = reinterpret_cast<const vernon::rhi::directx12::Image *>(static_cast<uintptr_t>(
        vernon::rhi::resolveCommandResource(device, key, vernon::rhi::ResourceKind::ImageView, targetKey).value()));
    const vernon::rhi::directx12::Image &owner = *view->stateOwner;
    ASSERT_EQ(owner.subresourceStates.size(), 2u);
    EXPECT_EQ(owner.subresourceStates[0], D3D12_RESOURCE_STATE_RENDER_TARGET);
    EXPECT_EQ(owner.subresourceStates[1], D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
    ASSERT_TRUE(vernon::rhi::endEncoderRendering(device, key).isOk());
    EXPECT_EQ(vernonRhiDeviceDestroyCommandEncoder(device, encoder), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(owner.subresourceStates[0], D3D12_RESOURCE_STATE_COMMON);
    EXPECT_EQ(owner.subresourceStates[1], D3D12_RESOURCE_STATE_COMMON);
    for (VernonRhiImageView created : {target, sampled})
        EXPECT_EQ(vernonRhiDeviceDestroyImageView(device, created), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device, image), VERNON_RHI_STATUS_OK);
    vernonRhiDestroyDevice(device);
}

} // namespace
