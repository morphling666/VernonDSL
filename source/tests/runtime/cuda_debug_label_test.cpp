#include "hal/cuda/cuda_device.h"
#include "hal/cuda/cuda_state.h"
#include "nvtx_call_recorder.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr const char kNoopKernel[] = R"(.version 6.0
.target sm_50
.address_size 64

.visible .entry vernon_noop()
{
    ret;
}
)";

class CudaDebugLabels : public testing::Test {
protected:
    void SetUp() override {
        state = std::make_shared<vernon::hal::cuda::DeviceState>();
        if (state->initialize(0) != vernon::hal::cuda::kSuccess)
            GTEST_SKIP() << "CUDA device is unavailable";
    }

    // Runs three dispatches of a kernel, labelled "a", "b" and none, and returns the NVTX calls they made.
    std::vector<std::string> run(bool labels) {
        auto device = vernon::hal::createCudaDevice(state, labels);
        EXPECT_TRUE(device);
        if (!device)
            return {};
        vernon::hal::ExecutableSource source;
        source.artifact = kNoopKernel;
        source.entry = "vernon_noop";
        auto executable = device->createExecutable(source);
        EXPECT_FALSE(executable.isErr());
        if (executable.isErr())
            return {};
        const std::unique_ptr<vernon::hal::Executable> kernel = std::move(executable).value();
        const char *names[] = {"a", "b", ""};
        std::vector<vernon::hal::DispatchCommand> commands(3);
        for (uint32_t index = 0; index < commands.size(); ++index) {
            commands[index].executable = kernel.get();
            commands[index].gridOffset = index * 3 * sizeof(uint32_t);
            commands[index].node = index;
            commands[index].label = names[index];
        }
        auto recorded = device->record(commands);
        EXPECT_FALSE(recorded.isErr());
        if (recorded.isErr())
            return {};
        const std::unique_ptr<vernon::hal::CommandBuffer> buffer = std::move(recorded).value();
        const uint32_t grids[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
        (void)vernon::tests::takeNvtxCalls();
        auto submitted = device->submit(*buffer, reinterpret_cast<const uint8_t *>(grids), sizeof(grids));
        EXPECT_FALSE(submitted.isErr());
        if (submitted.isErr())
            return {};
        EXPECT_FALSE(device->wait(submitted.value()).isErr());
        return vernon::tests::takeNvtxCalls();
    }

    std::shared_ptr<vernon::hal::cuda::DeviceState> state;
};

TEST_F(CudaDebugLabels, LaunchesEachLabelledDispatchInsideARangeOfItsLabel) {
    EXPECT_EQ(run(true), (std::vector<std::string>{"start a", "end a", "start b", "end b"}));
}

TEST_F(CudaDebugLabels, MakesNoNvtxCallsWithoutLabels) { EXPECT_TRUE(run(false).empty()); }

} // namespace
