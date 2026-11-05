#include "hal/cpu/cpu_device.h"
#include "runtime/cpu_workgroup_dispatch.h"
#include "runtime/schedule/executor.h"
#include "runtime/schedule/fragment.h"
#include "runtime/schedule/linker.h"

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

namespace {

using vernon::schedule::Dispatch;
using vernon::schedule::Fragment;

// The record of `fill(out: TensorView[f32, (dyn,), write], value: f32)`: the buffer address, the view's offset,
// extent and stride as i64, the value, and the grid.
constexpr uint64_t kOutSlot = 0;
constexpr uint64_t kOffsetField = 8;
constexpr uint64_t kExtentField = 16;
constexpr uint64_t kStrideField = 24;
constexpr uint64_t kValueSlot = 32;
constexpr uint64_t kGrid = 36;
constexpr uint64_t kRecordSize = 48;
constexpr uint32_t kWorkgroup = 4;

template <typename T> T load(const uint8_t *record, uint64_t offset) {
    T value;
    std::memcpy(&value, record + offset, sizeof(T));
    return value;
}

VernonStatus fillEntry(const VernonCpuInvocation *invocation) {
    const auto &range = *static_cast<const VernonCpuRange *>(invocation->arguments);
    const auto *record = static_cast<const uint8_t *>(range.arguments);
    auto *out = load<float *>(record, kOutSlot);
    const auto offset = load<int64_t>(record, kOffsetField);
    const auto extent = load<int64_t>(record, kExtentField);
    const auto stride = load<int64_t>(record, kStrideField);
    const auto value = load<float>(record, kValueSlot);
    for (size_t lane = range.lane_begin; lane < range.lane_end; ++lane) {
        const int64_t index = int64_t{range.group[0]} * range.workgroup[0] + static_cast<int64_t>(lane);
        if (index < extent)
            out[offset + index * stride] = value + static_cast<float>(index);
    }
    return VERNON_STATUS_OK;
}

Fragment fillFragment() {
    Fragment fragment;
    fragment.executables.push_back({&fillEntry, {kWorkgroup, 1, 1}});
    fragment.slots.push_back({VERNON_SCHEDULE_SLOT_BUFFER, VERNON_DATA_F32, 1, VERNON_SCHEDULE_ACCESS_WRITE,
                              sizeof(float), kOutSlot, sizeof(void *)});
    fragment.slots.push_back(
        {VERNON_SCHEDULE_SLOT_SCALAR, VERNON_DATA_F32, 0, 0, sizeof(float), kValueSlot, sizeof(float)});
    fragment.shapes = {{-1}, {}};
    fragment.extents = {{{0, 0, 0}}, {}};
    fragment.fields.push_back({0, VERNON_SCHEDULE_FIELD_OFFSET, 0, 8, kOffsetField});
    fragment.fields.push_back({0, VERNON_SCHEDULE_FIELD_EXTENT, 0, 8, kExtentField});
    fragment.fields.push_back({0, VERNON_SCHEDULE_FIELD_STRIDE, 0, 8, kStrideField});
    Dispatch dispatch;
    dispatch.executable = 0;
    dispatch.argumentsOffset = 0;
    dispatch.argumentsSize = kGrid;
    dispatch.gridOffset = kGrid;
    dispatch.accesses = {{false, 0, VERNON_SCHEDULE_ACCESS_WRITE}};
    fragment.operations.push_back(dispatch);
    fragment.recordSize = kRecordSize;
    fragment.recordTemplate.assign(kRecordSize, 0);
    fragment.callerGrid = true;
    return fragment;
}

std::string validationError(const Fragment &fragment) {
    auto valid = fragment.validate();
    return valid.isOk() ? std::string() : valid.error();
}

TEST(ScheduleFragment, AcceptsAWellFormedKernel) { EXPECT_EQ(validationError(fillFragment()), ""); }

TEST(ScheduleFragment, RejectsOverlappingRecordRanges) {
    Fragment fragment = fillFragment();
    fragment.fields[1].offset = kOffsetField + 4;
    EXPECT_NE(validationError(fragment), "");
}

TEST(ScheduleFragment, RejectsIncompleteTensorViewMetadata) {
    Fragment fragment = fillFragment();
    fragment.fields.pop_back();
    EXPECT_NE(validationError(fragment).find("metadata"), std::string::npos);
}

TEST(ScheduleFragment, RejectsAGridOutsideTheRecord) {
    Fragment fragment = fillFragment();
    fragment.operations[0].gridOffset = kRecordSize - 4;
    EXPECT_NE(validationError(fragment), "");
}

TEST(ScheduleFragment, AcceptsOneAddressFieldPerBufferAndRejectsTwo) {
    Fragment fragment = fillFragment();
    fragment.recordSize = kRecordSize + 16;
    fragment.recordTemplate.assign(fragment.recordSize, 0);
    fragment.fields.push_back({0, VERNON_SCHEDULE_FIELD_ADDRESS, 0, 8, kRecordSize});
    EXPECT_EQ(validationError(fragment), "");
    fragment.fields.push_back({0, VERNON_SCHEDULE_FIELD_ADDRESS, 0, 8, kRecordSize + 8});
    EXPECT_NE(validationError(fragment).find("metadata"), std::string::npos);
}

TEST(ScheduleFragment, RejectsATemplateThatDoesNotCoverTheRecord) {
    Fragment fragment = fillFragment();
    fragment.recordTemplate.pop_back();
    EXPECT_NE(validationError(fragment).find("template"), std::string::npos);
}

TEST(ScheduleFragment, RejectsAWriteToAScalarSlot) {
    Fragment fragment = fillFragment();
    fragment.operations[0].accesses = {{false, 1, VERNON_SCHEDULE_ACCESS_WRITE}};
    EXPECT_NE(validationError(fragment), "");
}

TEST(ScheduleFragment, RejectsAnExtentOfAScalarSlot) {
    Fragment fragment = fillFragment();
    fragment.extents[0][0] = {0, 1, 0};
    EXPECT_NE(validationError(fragment).find("extent"), std::string::npos);
}

TEST(ScheduleArena, SharesBytesBetweenTransientsWhoseIntervalsAreDisjoint) {
    std::vector<vernon::schedule::Transient> transients(3);
    transients[0].first = 0;
    transients[0].last = 1;
    transients[1].first = 1;
    transients[1].last = 2;
    transients[2].first = 2;
    transients[2].last = 3;
    const vernon::schedule::ArenaLayout layout = vernon::schedule::packArena(transients, {256, 512, 256}, 256);
    // The largest is placed first; the other two overlap it but not each other.
    EXPECT_EQ(layout.offsets[1], 0u);
    EXPECT_EQ(layout.offsets[0], 512u);
    EXPECT_EQ(layout.offsets[2], 512u);
    EXPECT_EQ(layout.size, 768u);
}

TEST(ScheduleLinker, LinksOneNodeAndRejectsCallerGridsInGraphs) {
    const Fragment fragment = fillFragment();
    auto linked = vernon::schedule::link({&fragment});
    ASSERT_TRUE(linked.isOk()) << linked.error();
    EXPECT_EQ(linked.value().recordSize, kRecordSize);
    EXPECT_EQ(linked.value().operations.size(), 1u);
    EXPECT_TRUE(vernon::schedule::link({&fragment, &fragment}).isErr());
}

class ScheduleExecution : public ::testing::Test {
protected:
    void SetUp() override {
        std::string error;
        scheduler_ = vernon::runtime::CpuWorkgroupScheduler::create(
            vernon::runtime::CpuWorkgroupScheduler::defaultConfig(), error);
        ASSERT_TRUE(scheduler_) << error;
        device_ = vernon::hal::createCpuDevice(*scheduler_);
        ASSERT_TRUE(device_);
    }

    std::unique_ptr<vernon::runtime::CpuWorkgroupScheduler> scheduler_;
    std::unique_ptr<vernon::hal::Device> device_;
};

TEST_F(ScheduleExecution, RecordsOnceAndCallsWithEachRecord) {
    const Fragment fragment = fillFragment();
    auto linked = vernon::schedule::link({&fragment});
    ASSERT_TRUE(linked.isOk()) << linked.error();
    auto recording = vernon::schedule::record(*device_, linked.value());
    ASSERT_TRUE(recording.isOk()) << recording.error();

    std::vector<float> storage(64, -1.0f);
    const auto call = [&](int64_t offset, int64_t extent, int64_t stride, float value, uint32_t groups) {
        uint8_t record[kRecordSize]{};
        float *data = storage.data();
        std::memcpy(record + kOutSlot, &data, sizeof(data));
        std::memcpy(record + kOffsetField, &offset, sizeof(offset));
        std::memcpy(record + kExtentField, &extent, sizeof(extent));
        std::memcpy(record + kStrideField, &stride, sizeof(stride));
        std::memcpy(record + kValueSlot, &value, sizeof(value));
        const uint32_t grid[3] = {groups, 1, 1};
        std::memcpy(record + kGrid, grid, sizeof(grid));
        return vernon::schedule::call(*device_, linked.value(), recording.value(), record, sizeof(record));
    };

    auto first = call(0, 10, 1, 100.0f, 3);
    ASSERT_TRUE(first.isOk()) << first.error();
    for (int index = 0; index < 10; ++index)
        EXPECT_EQ(storage[index], 100.0f + static_cast<float>(index));
    EXPECT_EQ(storage[10], -1.0f);

    // The same recording serves a view of another offset, extent and stride.
    auto second = call(20, 5, 3, 7.0f, 2);
    ASSERT_TRUE(second.isOk()) << second.error();
    for (int index = 0; index < 5; ++index)
        EXPECT_EQ(storage[20 + 3 * index], 7.0f + static_cast<float>(index));
    EXPECT_EQ(storage[21], -1.0f);

    // Each call signals the next timeline point, which the device can wait for.
    EXPECT_GT(second.value(), first.value());
    EXPECT_TRUE(device_->wait(second.value()).isOk());
    EXPECT_TRUE(device_->wait(second.value() + 1).isErr());
}

VernonStatus failingEntry(const VernonCpuInvocation *) { return VERNON_STATUS_INTERNAL_ERROR; }

// Adds one to every element of the groups it runs, so a group run twice or not at all shows in the output.
VernonStatus incrementChunk(const VernonCpuChunk *chunk) {
    if (!chunk || chunk->struct_size != sizeof(VernonCpuChunk) || chunk->group_begin >= chunk->group_end)
        return VERNON_STATUS_INVALID_ARGUMENT;
    const auto *record = static_cast<const uint8_t *>(chunk->arguments);
    auto *out = load<float *>(record, kOutSlot);
    const auto extent = load<int64_t>(record, kExtentField);
    for (uint64_t group = chunk->group_begin; group < chunk->group_end; ++group)
        for (uint32_t lane = 0; lane < chunk->workgroup[0]; ++lane) {
            const int64_t index = static_cast<int64_t>(group * chunk->workgroup[0] + lane);
            if (index < extent)
                out[index] += 1.0f;
        }
    return VERNON_STATUS_OK;
}

TEST_F(ScheduleExecution, RunsSpansOfWholeWorkgroupsThroughTheChunkEntry) {
    Fragment fragment = fillFragment();
    fragment.executables[0].cpuEntry = &failingEntry;
    fragment.executables[0].cpuChunkEntry = &incrementChunk;
    auto linked = vernon::schedule::link({&fragment});
    ASSERT_TRUE(linked.isOk()) << linked.error();
    auto recording = vernon::schedule::record(*device_, linked.value());
    ASSERT_TRUE(recording.isOk()) << recording.error();

    constexpr uint32_t groups = 4096;
    std::vector<float> storage(size_t{groups} * kWorkgroup, 0.0f);
    uint8_t record[kRecordSize]{};
    float *data = storage.data();
    const int64_t extent = static_cast<int64_t>(storage.size());
    std::memcpy(record + kOutSlot, &data, sizeof(data));
    std::memcpy(record + kExtentField, &extent, sizeof(extent));
    const uint32_t grid[3] = {groups, 1, 1};
    std::memcpy(record + kGrid, grid, sizeof(grid));
    auto called = vernon::schedule::call(*device_, linked.value(), recording.value(), record, sizeof(record));
    ASSERT_TRUE(called.isOk()) << called.error();
    for (size_t index = 0; index < storage.size(); ++index)
        ASSERT_EQ(storage[index], 1.0f) << index;
}

// Adds `value + index` to every element of the view, as `fillEntry` lays out its frame.
VernonStatus accumulateEntry(const VernonCpuInvocation *invocation) {
    const auto &range = *static_cast<const VernonCpuRange *>(invocation->arguments);
    const auto *record = static_cast<const uint8_t *>(range.arguments);
    auto *out = load<float *>(record, kOutSlot);
    const auto offset = load<int64_t>(record, kOffsetField);
    const auto extent = load<int64_t>(record, kExtentField);
    const auto stride = load<int64_t>(record, kStrideField);
    const auto value = load<float>(record, kValueSlot);
    for (size_t lane = range.lane_begin; lane < range.lane_end; ++lane) {
        const int64_t index = int64_t{range.group[0]} * range.workgroup[0] + static_cast<int64_t>(lane);
        if (index < extent)
            out[offset + index * stride] += value + static_cast<float>(index);
    }
    return VERNON_STATUS_OK;
}

// `doubled(source: read, out: write)`: each view as an address, an offset, an extent and a stride.
VernonStatus doubleEntry(const VernonCpuInvocation *invocation) {
    const auto &range = *static_cast<const VernonCpuRange *>(invocation->arguments);
    const auto *record = static_cast<const uint8_t *>(range.arguments);
    const auto *source = load<const float *>(record, 0);
    const auto sourceOffset = load<int64_t>(record, 8);
    const auto sourceStride = load<int64_t>(record, 24);
    auto *out = load<float *>(record, 32);
    const auto outOffset = load<int64_t>(record, 40);
    const auto extent = load<int64_t>(record, 48);
    const auto outStride = load<int64_t>(record, 56);
    for (size_t lane = range.lane_begin; lane < range.lane_end; ++lane) {
        const int64_t index = int64_t{range.group[0]} * range.workgroup[0] + static_cast<int64_t>(lane);
        if (index < extent)
            out[outOffset + index * outStride] = 2.0f * source[sourceOffset + index * sourceStride];
    }
    return VERNON_STATUS_OK;
}

// A Program of two nodes over a zeroed transient as long as its output: accumulate(tmp, value), then
// doubled(tmp, out). The fill's record comes first, then each node's frame.
Fragment chainFragment() {
    using vernon::schedule::BufferAccess;
    using vernon::schedule::OperationKind;
    constexpr uint32_t groups = 256;
    Fragment fragment;
    fragment.executables.push_back({&accumulateEntry, {kWorkgroup, 1, 1}});
    fragment.executables.push_back({&doubleEntry, {kWorkgroup, 1, 1}});
    fragment.slots.push_back(
        {VERNON_SCHEDULE_SLOT_BUFFER, VERNON_DATA_F32, 1, VERNON_SCHEDULE_ACCESS_WRITE, sizeof(float), 96, 8});
    fragment.slots.push_back({VERNON_SCHEDULE_SLOT_SCALAR, VERNON_DATA_F32, 0, 0, sizeof(float), 48, sizeof(float)});
    fragment.shapes = {{-1}, {}};
    fragment.extents = {{{0, 0, 0}}, {}};
    fragment.fields.push_back({0, VERNON_SCHEDULE_FIELD_OFFSET, 0, 8, 104});
    fragment.fields.push_back({0, VERNON_SCHEDULE_FIELD_EXTENT, 0, 8, 112});
    fragment.fields.push_back({0, VERNON_SCHEDULE_FIELD_STRIDE, 0, 8, 120});
    vernon::schedule::Transient transient;
    transient.dtype = VERNON_DATA_F32;
    transient.elementSize = sizeof(float);
    transient.extents = {{0, 0, 0}};
    transient.zeroed = true;
    transient.first = 0;
    transient.last = 2;
    fragment.transients.push_back(transient);
    fragment.transientFields = {
        {0, VERNON_SCHEDULE_FIELD_ADDRESS, 0, 8, 0},  {0, vernon::schedule::kTransientFieldBytes, 0, 8, 8},
        {0, VERNON_SCHEDULE_FIELD_ADDRESS, 0, 8, 16}, {0, VERNON_SCHEDULE_FIELD_OFFSET, 0, 8, 24},
        {0, VERNON_SCHEDULE_FIELD_EXTENT, 0, 8, 32},  {0, VERNON_SCHEDULE_FIELD_STRIDE, 0, 8, 40},
        {0, VERNON_SCHEDULE_FIELD_ADDRESS, 0, 8, 64}, {0, VERNON_SCHEDULE_FIELD_OFFSET, 0, 8, 72},
        {0, VERNON_SCHEDULE_FIELD_EXTENT, 0, 8, 80},  {0, VERNON_SCHEDULE_FIELD_STRIDE, 0, 8, 88},
    };
    Dispatch fill;
    fill.kind = OperationKind::Fill;
    fill.argumentsOffset = 0;
    fill.argumentsSize = 16;
    fill.accesses = {BufferAccess{true, 0, VERNON_SCHEDULE_ACCESS_WRITE}};
    Dispatch accumulate;
    accumulate.executable = 0;
    accumulate.argumentsOffset = 16;
    accumulate.argumentsSize = 36;
    accumulate.gridOffset = 52;
    accumulate.accesses = {BufferAccess{true, 0, VERNON_SCHEDULE_ACCESS_READ_WRITE}};
    Dispatch doubled;
    doubled.executable = 1;
    doubled.argumentsOffset = 64;
    doubled.argumentsSize = 64;
    doubled.gridOffset = 128;
    doubled.accesses = {BufferAccess{true, 0, VERNON_SCHEDULE_ACCESS_READ},
                        BufferAccess{false, 0, VERNON_SCHEDULE_ACCESS_WRITE}};
    fragment.operations = {fill, accumulate, doubled};
    fragment.recordSize = 144;
    fragment.recordTemplate.assign(fragment.recordSize, 0);
    const uint32_t grid[3] = {groups, 1, 1};
    std::memcpy(fragment.recordTemplate.data() + 52, grid, sizeof(grid));
    std::memcpy(fragment.recordTemplate.data() + 128, grid, sizeof(grid));
    return fragment;
}

TEST_F(ScheduleExecution, PlacesAZeroedTransientInTheArenaAndGrowsItWithItsCapacityClass) {
    const Fragment fragment = chainFragment();
    ASSERT_EQ(validationError(fragment), "");
    auto linked = vernon::schedule::link({&fragment});
    ASSERT_TRUE(linked.isOk()) << linked.error();
    auto recording = vernon::schedule::record(*device_, linked.value());
    ASSERT_TRUE(recording.isOk()) << recording.error();

    std::vector<float> out(1024, -1.0f);
    const auto call = [&](int64_t extent, float value) {
        std::vector<uint8_t> record = linked.value().recordTemplate;
        float *data = out.data();
        const int64_t offset = 0;
        const int64_t stride = 1;
        std::memcpy(record.data() + 96, &data, sizeof(data));
        std::memcpy(record.data() + 104, &offset, sizeof(offset));
        std::memcpy(record.data() + 112, &extent, sizeof(extent));
        std::memcpy(record.data() + 120, &stride, sizeof(stride));
        std::memcpy(record.data() + 48, &value, sizeof(value));
        return vernon::schedule::call(*device_, linked.value(), recording.value(), record.data(), record.size());
    };

    ASSERT_TRUE(call(10, 1.0f).isOk());
    for (int index = 0; index < 10; ++index)
        EXPECT_EQ(out[index], 2.0f * (1.0f + static_cast<float>(index)));
    const void *arena = recording.value().arena->address();
    // The fill runs on every call, so the transient does not keep the previous call's sums.
    ASSERT_TRUE(call(10, 3.0f).isOk());
    for (int index = 0; index < 10; ++index)
        EXPECT_EQ(out[index], 2.0f * (3.0f + static_cast<float>(index)));
    EXPECT_EQ(recording.value().arena->address(), arena);
    // 1000 floats need a larger capacity class, so the arena grows.
    ASSERT_TRUE(call(1000, 0.5f).isOk());
    for (int index = 0; index < 1000; ++index)
        ASSERT_EQ(out[index], 2.0f * (0.5f + static_cast<float>(index))) << index;
    EXPECT_GE(recording.value().arenaSize, 1000 * sizeof(float));
}

// A Program of one node, doubled(source, out), whose output is as long as its source: each view an address, an
// offset, an extent and a stride, then its fixed grid of 256 workgroups.
Fragment doubledFragment() {
    using vernon::schedule::BufferAccess;
    Fragment fragment;
    fragment.executables.push_back({&doubleEntry, {kWorkgroup, 1, 1}});
    fragment.slots.push_back(
        {VERNON_SCHEDULE_SLOT_BUFFER, VERNON_DATA_F32, 1, VERNON_SCHEDULE_ACCESS_READ, sizeof(float), 0, 8});
    fragment.slots.push_back(
        {VERNON_SCHEDULE_SLOT_BUFFER, VERNON_DATA_F32, 1, VERNON_SCHEDULE_ACCESS_WRITE, sizeof(float), 32, 8});
    fragment.shapes = {{-1}, {-1}};
    fragment.extents = {{{0, 0, 0}}, {{0, 0, 0}}};
    for (uint32_t slot = 0; slot < 2; ++slot) {
        fragment.fields.push_back({slot, VERNON_SCHEDULE_FIELD_OFFSET, 0, 8, 32 * slot + 8});
        fragment.fields.push_back({slot, VERNON_SCHEDULE_FIELD_EXTENT, 0, 8, 32 * slot + 16});
        fragment.fields.push_back({slot, VERNON_SCHEDULE_FIELD_STRIDE, 0, 8, 32 * slot + 24});
    }
    Dispatch dispatch;
    dispatch.argumentsOffset = 0;
    dispatch.argumentsSize = 64;
    dispatch.gridOffset = 64;
    dispatch.accesses = {BufferAccess{false, 0, VERNON_SCHEDULE_ACCESS_READ},
                         BufferAccess{false, 1, VERNON_SCHEDULE_ACCESS_WRITE}};
    fragment.operations.push_back(dispatch);
    fragment.recordSize = 80;
    fragment.recordTemplate.assign(fragment.recordSize, 0);
    const uint32_t grid[3] = {256, 1, 1};
    std::memcpy(fragment.recordTemplate.data() + 64, grid, sizeof(grid));
    return fragment;
}

// Writes a contiguous view of `data` into every site of linked slot `slot`, as a stub does from the slot's fields.
void bindView(std::vector<uint8_t> &record, const vernon::schedule::LinkedSchedule &linked, uint32_t slot, float *data,
              int64_t extent) {
    std::memcpy(record.data() + linked.slots[slot].offset, &data, sizeof(data));
    for (const VernonScheduleField &field : linked.fields) {
        if (field.slot != slot)
            continue;
        int64_t value = 0;
        if (field.kind == VERNON_SCHEDULE_FIELD_ADDRESS)
            value = static_cast<int64_t>(reinterpret_cast<uintptr_t>(data));
        else if (field.kind == VERNON_SCHEDULE_FIELD_EXTENT)
            value = extent;
        else if (field.kind == VERNON_SCHEDULE_FIELD_STRIDE)
            value = 1;
        std::memcpy(record.data() + field.offset, &value, sizeof(value));
    }
}

TEST_F(ScheduleExecution, LinksAnInternalConnectionIntoTheArena) {
    const Fragment first = doubledFragment();
    const Fragment second = doubledFragment();
    ASSERT_EQ(validationError(first), "");
    auto linked = vernon::schedule::link({&first, &second}, {{{{0, 1}, {1, 0}}, true}});
    ASSERT_TRUE(linked.isOk()) << linked.error();
    const vernon::schedule::LinkedSchedule &schedule = linked.value();
    ASSERT_EQ(schedule.slots.size(), 2u);
    EXPECT_EQ(schedule.nodeSlots,
              (std::vector<std::vector<uint32_t>>{{0, vernon::schedule::kNoSlot}, {vernon::schedule::kNoSlot, 1}}));
    ASSERT_EQ(schedule.transients.size(), 1u);
    EXPECT_EQ(schedule.transients[0].first, 0u);
    EXPECT_EQ(schedule.transients[0].last, 1u);
    // The second output is as long as the first node's source, through the connection.
    EXPECT_EQ(schedule.extents[1][0].slot, 0u);
    EXPECT_EQ(schedule.operations[1].after, std::vector<uint32_t>{0});
    auto recording = vernon::schedule::record(*device_, schedule);
    ASSERT_TRUE(recording.isOk()) << recording.error();

    for (int64_t extent : {10, 700}) {
        std::vector<float> source(static_cast<size_t>(extent));
        for (size_t index = 0; index < source.size(); ++index)
            source[index] = static_cast<float>(index);
        std::vector<float> out(source.size(), -1.0f);
        std::vector<uint8_t> record = schedule.recordTemplate;
        bindView(record, schedule, 0, source.data(), extent);
        bindView(record, schedule, 1, out.data(), extent);
        auto called = vernon::schedule::call(*device_, schedule, recording.value(), record.data(), record.size());
        ASSERT_TRUE(called.isOk()) << called.error();
        for (size_t index = 0; index < out.size(); ++index)
            ASSERT_EQ(out[index], 4.0f * source[index]) << index;
    }
}

TEST_F(ScheduleExecution, LinksAnExportedConnectionAsOneSlotOfBothNodes) {
    const Fragment first = doubledFragment();
    const Fragment second = doubledFragment();
    auto linked = vernon::schedule::link({&first, &second}, {{{{0, 1}, {1, 0}}, false}});
    ASSERT_TRUE(linked.isOk()) << linked.error();
    const vernon::schedule::LinkedSchedule &schedule = linked.value();
    ASSERT_EQ(schedule.slots.size(), 3u);
    EXPECT_EQ(schedule.nodeSlots, (std::vector<std::vector<uint32_t>>{{0, 1}, {1, 2}}));
    EXPECT_EQ(schedule.slots[1].access, static_cast<uint32_t>(VERNON_SCHEDULE_ACCESS_READ_WRITE));
    EXPECT_TRUE(schedule.transients.empty());
    auto recording = vernon::schedule::record(*device_, schedule);
    ASSERT_TRUE(recording.isOk()) << recording.error();

    std::vector<float> source{1.0f, 2.0f, 3.0f};
    std::vector<float> middle(3, -1.0f);
    std::vector<float> out(3, -1.0f);
    std::vector<uint8_t> record = schedule.recordTemplate;
    bindView(record, schedule, 0, source.data(), 3);
    bindView(record, schedule, 1, middle.data(), 3);
    bindView(record, schedule, 2, out.data(), 3);
    auto called = vernon::schedule::call(*device_, schedule, recording.value(), record.data(), record.size());
    ASSERT_TRUE(called.isOk()) << called.error();
    EXPECT_EQ(middle, (std::vector<float>{2.0f, 4.0f, 6.0f}));
    EXPECT_EQ(out, (std::vector<float>{4.0f, 8.0f, 12.0f}));
    // The connection's extent must equal the first source's, which its producer declares.
    bindView(record, schedule, 1, middle.data(), 2);
    EXPECT_TRUE(vernon::schedule::call(*device_, schedule, recording.value(), record.data(), record.size()).isErr());
}

TEST(ScheduleLinker, OrdersEachWriteAfterTheReadsAndWritesBeforeIt) {
    // Two nodes read X, then a third writes it: the third follows both reads, and the second follows nothing.
    const Fragment node = doubledFragment();
    auto linked = vernon::schedule::link({&node, &node, &node}, {{{{0, 0}, {1, 0}, {2, 1}}, false}});
    ASSERT_TRUE(linked.isOk()) << linked.error();
    const auto &operations = linked.value().operations;
    EXPECT_TRUE(operations[0].after.empty());
    EXPECT_TRUE(operations[1].after.empty());
    EXPECT_EQ(operations[2].after, (std::vector<uint32_t>{0, 1}));
}

TEST(ScheduleLinker, RejectsConnectionsItCannotLink) {
    const Fragment node = doubledFragment();
    // An internal connection's extents must come from a node slot, and a node must write it before any reads it.
    EXPECT_TRUE(vernon::schedule::link({&node}, {{{{0, 0}}, true}}).isErr());
    // A node slot is in one connection, and connected slots have one element type.
    EXPECT_TRUE(vernon::schedule::link({&node, &node}, {{{{0, 1}, {1, 0}}, false}, {{{0, 1}}, false}}).isErr());
    Fragment integers = doubledFragment();
    integers.slots[0].dtype = VERNON_DATA_I32;
    EXPECT_TRUE(vernon::schedule::link({&node, &integers}, {{{{0, 1}, {1, 0}}, false}}).isErr());
    EXPECT_TRUE(vernon::schedule::link({&node}, {{{{3, 0}}, false}}).isErr());
}

TEST_F(ScheduleExecution, RejectsAnArgumentWhoseExtentDisagreesWithTheOneItMustEqual) {
    Fragment fragment = fillFragment();
    fragment.extents[0][0] = {7, vernon::schedule::kNoSlot, 0};
    auto linked = vernon::schedule::link({&fragment});
    ASSERT_TRUE(linked.isOk()) << linked.error();
    auto recording = vernon::schedule::record(*device_, linked.value());
    ASSERT_TRUE(recording.isOk());
    std::vector<float> storage(8, 0.0f);
    uint8_t record[kRecordSize]{};
    float *data = storage.data();
    const int64_t extent = 6;
    std::memcpy(record + kOutSlot, &data, sizeof(data));
    std::memcpy(record + kExtentField, &extent, sizeof(extent));
    const uint32_t grid[3] = {2, 1, 1};
    std::memcpy(record + kGrid, grid, sizeof(grid));
    auto called = vernon::schedule::call(*device_, linked.value(), recording.value(), record, sizeof(record));
    ASSERT_TRUE(called.isErr());
    EXPECT_NE(called.error().find("extent"), std::string::npos);
}

TEST_F(ScheduleExecution, RejectsAZeroGridAndAWrongRecordSize) {
    const Fragment fragment = fillFragment();
    auto linked = vernon::schedule::link({&fragment});
    ASSERT_TRUE(linked.isOk());
    auto recording = vernon::schedule::record(*device_, linked.value());
    ASSERT_TRUE(recording.isOk());
    uint8_t record[kRecordSize]{};
    EXPECT_TRUE(vernon::schedule::call(*device_, linked.value(), recording.value(), record, sizeof(record)).isErr());
    EXPECT_TRUE(
        vernon::schedule::call(*device_, linked.value(), recording.value(), record, sizeof(record) - 1).isErr());
}

} // namespace
