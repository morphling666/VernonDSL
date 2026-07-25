#include "runtime/runtime_test_hooks.h"
#include "vernon-c/Runtime.h"

#include <gtest/gtest.h>

#include <array>

namespace {

class DirectX12RuntimeTest : public testing::Test {
protected:
    void SetUp() override { vernon::runtime::setDirectX12WarpForTests(true); }
    void TearDown() override { vernon::runtime::setDirectX12WarpForTests(false); }
};

TEST_F(DirectX12RuntimeTest, CreatesWarpContextAndCopiesOwnedResources) {
    VernonRuntimeContext *context = vernonRuntimeCreate(VERNON_RUNTIME_DIRECTX12, 0);
    ASSERT_NE(context, nullptr);
    const VernonRuntimeCapabilities capabilities = vernonRuntimeGetContextCapabilities(context);
    EXPECT_TRUE(capabilities.available);
    EXPECT_TRUE(capabilities.supports_compute);
    EXPECT_TRUE(capabilities.supports_graphics);
    EXPECT_EQ(capabilities.api_version_major, 12);

    VernonDeviceBuffer *buffer = vernonRuntimeBufferAllocate(context, 16, 4);
    ASSERT_NE(buffer, nullptr);
    const std::array<uint32_t, 4> source{1, 2, 3, 4};
    std::array<uint32_t, 4> result{};
    EXPECT_EQ(vernonRuntimeCopyFromHost(buffer, 0, source.data(), sizeof(source)), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeCopyToHost(buffer, 0, result.data(), sizeof(result)), VERNON_STATUS_OK);
    EXPECT_EQ(result, source);
    EXPECT_EQ(vernonRuntimeBufferFree(buffer), VERNON_STATUS_OK);

    VernonDeviceTexture *texture = vernonRuntimeTextureCreate2D(context, 2, 2, VERNON_TEXTURE_RGBA8_UNORM);
    ASSERT_NE(texture, nullptr);
    const std::array<uint8_t, 16> pixels{255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255};
    std::array<uint8_t, 16> readback{};
    EXPECT_EQ(vernonRuntimeTextureCopyFromHost(texture, pixels.data(), pixels.size()), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeTextureCopyToHost(texture, readback.data(), readback.size()), VERNON_STATUS_OK);
    EXPECT_EQ(readback, pixels);
    EXPECT_EQ(vernonRuntimeTextureFree(texture), VERNON_STATUS_OK);

    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

} // namespace
