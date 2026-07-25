#include "VernonRHI.h"

#include <d3d12.h>
#include <dxgi1_6.h>
#include <gtest/gtest.h>
#include <wrl/client.h>

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

} // namespace
