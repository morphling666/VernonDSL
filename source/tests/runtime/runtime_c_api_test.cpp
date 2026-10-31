#include "runtime/backend_cpu.h"
#include "vernon-c/Runtime.h"
#include "vernon_host/containers.hpp"
#include "vernon_host/reflect.h"

#include <cstddef>
#include <cstdlib>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

static VernonStatus registered_stub(const VernonCpuInvocation *) { return VERNON_STATUS_OK; }
static VernonStatus other_image_stub(const VernonCpuInvocation *) { return VERNON_STATUS_INTERNAL_ERROR; }

TEST(RuntimeCApi, PullbackOpaqueHandleRejectsInvalidCalls) {
    EXPECT_EQ(vernonProgramPullbackApply(nullptr, nullptr, 0, nullptr), VERNON_STATUS_INVALID_ARGUMENT);
    vernonProgramPullbackDestroy(nullptr);
}

TEST(RuntimeCApi, StaticCpuRegistrationValidatesIdentity) {
    static const char static_symbol[] = "__vernon_cpu_test_fill";
    const VernonStaticCpuEntry entry{{static_symbol, sizeof(static_symbol) - 1}, registered_stub};
    ASSERT_TRUE(vernonRuntimeRegisterStaticCpuEntries(&entry, 1) == VERNON_STATUS_OK);
    ASSERT_TRUE(vernonRuntimeRegisterStaticCpuEntries(&entry, 1) == VERNON_STATUS_OK);
    ASSERT_TRUE(vernonRuntimeRegisterStaticCpuEntries(nullptr, 0) == VERNON_STATUS_OK);
    ASSERT_TRUE(vernonRuntimeRegisterStaticCpuEntries(nullptr, 1) == VERNON_STATUS_INVALID_ARGUMENT);
    const VernonStaticCpuEntry unnamed[]{entry, {{nullptr, 0}, registered_stub}};
    ASSERT_TRUE(vernonRuntimeRegisterStaticCpuEntries(unnamed, 2) == VERNON_STATUS_INVALID_ARGUMENT);
    const VernonStaticCpuEntry missingEntry{{static_symbol, sizeof(static_symbol) - 1}, nullptr};
    ASSERT_TRUE(vernonRuntimeRegisterStaticCpuEntries(&missingEntry, 1) == VERNON_STATUS_INVALID_ARGUMENT);
    ASSERT_TRUE(vernonRuntimeUnregisterStaticCpuEntries(&missingEntry, 1) == VERNON_STATUS_INVALID_ARGUMENT);
    ASSERT_TRUE(vernonRuntimeUnregisterStaticCpuEntries(&entry, 1) == VERNON_STATUS_OK);
    ASSERT_TRUE(vernonRuntimeUnregisterStaticCpuEntries(&entry, 1) == VERNON_STATUS_OK);
}

TEST(RuntimeCApi, StaticCpuRegistrationTracksEveryLoadedImage) {
    static const char symbol[] = "__vernon_cpu_test_images";
    const VernonStaticCpuEntry first{{symbol, sizeof(symbol) - 1}, registered_stub};
    const VernonStaticCpuEntry second{{symbol, sizeof(symbol) - 1}, other_image_stub};
    ASSERT_EQ(vernon::runtime::findStaticCpuEntry(symbol), nullptr);

    ASSERT_EQ(vernonRuntimeRegisterStaticCpuEntries(&first, 1), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeRegisterStaticCpuEntries(&first, 1), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeRegisterStaticCpuEntries(&second, 1), VERNON_STATUS_OK);
    EXPECT_EQ(vernon::runtime::findStaticCpuEntry(symbol), &registered_stub);

    ASSERT_EQ(vernonRuntimeUnregisterStaticCpuEntries(&first, 1), VERNON_STATUS_OK);
    EXPECT_EQ(vernon::runtime::findStaticCpuEntry(symbol), &registered_stub);
    ASSERT_EQ(vernonRuntimeUnregisterStaticCpuEntries(&first, 1), VERNON_STATUS_OK);
    EXPECT_EQ(vernon::runtime::findStaticCpuEntry(symbol), &other_image_stub);
    ASSERT_EQ(vernonRuntimeUnregisterStaticCpuEntries(&second, 1), VERNON_STATUS_OK);
    EXPECT_EQ(vernon::runtime::findStaticCpuEntry(symbol), nullptr);
}

TEST(RuntimeCApi, HostModuleRegistrationFindsTheNewestLiveTable) {
    // The registry never reads the tables, so any two distinct addresses stand in for them.
    static const int linked_storage = 0;
    static const int development_storage = 0;
    const auto *linked = reinterpret_cast<const VhModule *>(&linked_storage);
    const auto *development = reinterpret_cast<const VhModule *>(&development_storage);
    static const char id[] = "test/host-module";
    const VernonStringView name{id, sizeof(id) - 1};
    const VernonStaticHostModule first{name, linked};
    const VernonStaticHostModule second{name, development};

    EXPECT_EQ(vernonRuntimeFindHostModule(name), nullptr);
    EXPECT_EQ(vernonRuntimeFindHostModule({nullptr, 0}), nullptr);
    EXPECT_EQ(vernonRuntimeRegisterStaticHostModule(nullptr), VERNON_STATUS_INVALID_ARGUMENT);
    const VernonStaticHostModule unnamed{{nullptr, 0}, linked};
    EXPECT_EQ(vernonRuntimeRegisterStaticHostModule(&unnamed), VERNON_STATUS_INVALID_ARGUMENT);
    const VernonStaticHostModule empty{name, nullptr};
    EXPECT_EQ(vernonRuntimeRegisterStaticHostModule(&empty), VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(vernonRuntimeUnregisterStaticHostModule(&first), VERNON_STATUS_INVALID_ARGUMENT);

    ASSERT_EQ(vernonRuntimeRegisterStaticHostModule(&first), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeRegisterStaticHostModule(&first), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeFindHostModule(name), linked);
    ASSERT_EQ(vernonRuntimeRegisterStaticHostModule(&second), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeFindHostModule(name), development);

    ASSERT_EQ(vernonRuntimeUnregisterStaticHostModule(&second), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeFindHostModule(name), linked);
    EXPECT_EQ(vernonRuntimeUnregisterStaticHostModule(&second), VERNON_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(vernonRuntimeUnregisterStaticHostModule(&first), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeFindHostModule(name), linked);
    ASSERT_EQ(vernonRuntimeUnregisterStaticHostModule(&first), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeFindHostModule(name), nullptr);
}

// The runtime mirrors these reflect.h records instead of including the header.
static_assert(sizeof(VhList) == 24 && offsetof(VhList, size) == 8 && offsetof(VhList, capacity) == 16);
static_assert(offsetof(VhAllocator, user) == sizeof(void *) && offsetof(VhAllocator, release) == 3 * sizeof(void *));

TEST(RuntimeCApi, HostListsGrowThroughTheModuleAllocator) {
    struct Counting {
        int live = 0;
        uint64_t limit = UINT64_MAX;
    } counting;
    VhAllocator allocator{};
    allocator.struct_size = sizeof(VhAllocator);
    allocator.user = &counting;
    allocator.allocate = [](void *user, uint64_t size, uint64_t alignment) -> void * {
        auto *state = static_cast<Counting *>(user);
        EXPECT_EQ(alignment, alignof(int32_t));
        if (size > state->limit)
            return nullptr;
        ++state->live;
        return std::malloc(static_cast<size_t>(size));
    };
    allocator.release = [](void *user, void *pointer, uint64_t, uint64_t) {
        --static_cast<Counting *>(user)->live;
        std::free(pointer);
    };

    VhList list{};
    ASSERT_EQ(vernonRuntimeHostListReserve(&list, 3, sizeof(int32_t), alignof(int32_t), &allocator), 0);
    EXPECT_EQ(list.capacity, 3u);
    EXPECT_EQ(counting.live, 1);
    for (int32_t value = 0; value < 3; ++value)
        static_cast<int32_t *>(vh_list_data(&list))[list.size++] = value * 10;
    ASSERT_EQ(vernonRuntimeHostListReserve(&list, 4, sizeof(int32_t), alignof(int32_t), &allocator), 0);
    EXPECT_EQ(list.capacity, 6u) << "growth at least doubles";
    EXPECT_EQ(counting.live, 1);
    EXPECT_EQ(static_cast<int32_t *>(vh_list_data(&list))[2], 20) << "growth keeps the elements";
    EXPECT_EQ(vernonRuntimeHostListReserve(&list, 5, sizeof(int32_t), alignof(int32_t), &allocator), 0);
    EXPECT_EQ(list.capacity, 6u);

    counting.limit = 0;
    const VhList before = list;
    EXPECT_EQ(vernonRuntimeHostListReserve(&list, 100, sizeof(int32_t), alignof(int32_t), &allocator), -4);
    EXPECT_EQ(list.data, before.data) << "a failed reserve leaves the list unchanged";
    EXPECT_EQ(list.capacity, before.capacity);

    vernonRuntimeHostListRelease(&list, sizeof(int32_t), alignof(int32_t), &allocator);
    EXPECT_EQ(counting.live, 0);
    EXPECT_EQ(list.capacity, 0u);
    EXPECT_EQ(list.size, 0u);

    VhList defaulted{};
    ASSERT_EQ(vernonRuntimeHostListReserve(&defaulted, 8, 1, 1, nullptr), 0);
    vernonRuntimeHostListRelease(&defaulted, 1, 1, nullptr);
    vernonRuntimeHostListRelease(&defaulted, 1, 1, nullptr);
    EXPECT_EQ(defaulted.data, 0u) << "releasing an empty list is a no-op";
}

static_assert(sizeof(VhDict) == 48 && offsetof(VhDict, slots) == 32);

namespace {

struct CountingAllocator {
    int live = 0;
    VhAllocator allocator{};

    CountingAllocator() {
        allocator.struct_size = sizeof(VhAllocator);
        allocator.user = this;
        allocator.allocate = [](void *user, uint64_t size, uint64_t) -> void * {
            ++static_cast<CountingAllocator *>(user)->live;
            return std::malloc(static_cast<size_t>(size));
        };
        allocator.release = [](void *user, void *pointer, uint64_t, uint64_t) {
            --static_cast<CountingAllocator *>(user)->live;
            std::free(pointer);
        };
    }
};

struct IntEntry {
    uint64_t hash;
    int32_t key;
    int64_t value;
};

constexpr VernonHostDictLayout kIntLayout{sizeof(IntEntry), alignof(IntEntry),         offsetof(IntEntry, key),
                                          sizeof(int32_t),  offsetof(IntEntry, value), VERNON_HOST_DICT_KEY_BYTES};

IntEntry *intEntry(const VhDict &dict, uint64_t index) {
    return reinterpret_cast<IntEntry *>(static_cast<uintptr_t>(dict.entries)) + index;
}

} // namespace

TEST(RuntimeCApi, HostStoragesAllocateZeroedElementsThroughTheAllocator) {
    CountingAllocator counting;
    struct {
        uint64_t data;
        uint64_t shape[2];
    } storage{};
    const int64_t shape[2] = {3, 4};
    ASSERT_EQ(vernonRuntimeHostStorageAllocate(&storage, 2, shape, sizeof(float), alignof(float), &counting.allocator),
              0);
    EXPECT_EQ(counting.live, 1);
    EXPECT_EQ(storage.shape[0], 3u);
    EXPECT_EQ(storage.shape[1], 4u);
    auto *elements = reinterpret_cast<float *>(static_cast<uintptr_t>(storage.data));
    for (int index = 0; index < 12; ++index)
        EXPECT_EQ(elements[index], 0.0f);
    const float half = 0.5f;
    vernonRuntimeHostStorageFill(&storage, 2, &half, sizeof(float));
    EXPECT_EQ(elements[11], 0.5f);
    vernonRuntimeHostStorageRelease(&storage, 2, sizeof(float), alignof(float), &counting.allocator);
    EXPECT_EQ(counting.live, 0);
    EXPECT_EQ(storage.data, 0u);
    EXPECT_EQ(storage.shape[0], 0u) << "a released storage reads as empty";

    const int64_t empty[2] = {0, 5};
    ASSERT_EQ(vernonRuntimeHostStorageAllocate(&storage, 2, empty, sizeof(float), alignof(float), &counting.allocator),
              0);
    EXPECT_EQ(counting.live, 0) << "a storage of no elements holds no memory";
    EXPECT_EQ(storage.shape[1], 5u);
    vernonRuntimeHostStorageFill(&storage, 2, &half, sizeof(float));
    vernonRuntimeHostStorageRelease(&storage, 2, sizeof(float), alignof(float), &counting.allocator);

    const int64_t negative[2] = {2, -1};
    EXPECT_EQ(
        vernonRuntimeHostStorageAllocate(&storage, 2, negative, sizeof(float), alignof(float), &counting.allocator),
        -7);
    const int64_t huge[2] = {std::numeric_limits<int64_t>::max(), 4};
    EXPECT_EQ(vernonRuntimeHostStorageAllocate(&storage, 2, huge, sizeof(float), alignof(float), &counting.allocator),
              -4);
    EXPECT_EQ(storage.data, 0u);
    EXPECT_EQ(storage.shape[0], 0u) << "a failed allocation leaves the record unchanged";
    EXPECT_EQ(counting.live, 0);
}

TEST(RuntimeCApi, HostDictsKeepInsertionOrderAcrossRemovalAndGrowth) {
    CountingAllocator counting;
    VhDict dict{};
    for (int32_t key = 0; key < 100; ++key) {
        uint64_t entry = 0;
        ASSERT_EQ(vernonRuntimeHostDictInsert(&dict, &key, &kIntLayout, &counting.allocator, &entry), 0);
        EXPECT_EQ(intEntry(dict, entry)->value, 0) << "a new entry's value starts zeroed";
        intEntry(dict, entry)->value = key * 10;
    }
    EXPECT_EQ(dict.size, 100u);
    for (int32_t key = 0; key < 100; key += 2)
        EXPECT_EQ(vernonRuntimeHostDictRemove(&dict, &key, &kIntLayout, &counting.allocator), 1);
    const int32_t absent = 1000;
    EXPECT_EQ(vernonRuntimeHostDictRemove(&dict, &absent, &kIntLayout, &counting.allocator), 0);
    EXPECT_EQ(vernonRuntimeHostDictFind(&dict, &absent, &kIntLayout), -1);
    const int32_t seven = 7;
    const int64_t found = vernonRuntimeHostDictFind(&dict, &seven, &kIntLayout);
    ASSERT_GE(found, 0);
    EXPECT_EQ(intEntry(dict, static_cast<uint64_t>(found))->value, 70);

    uint64_t entry = 0;
    ASSERT_EQ(vernonRuntimeHostDictInsert(&dict, &seven, &kIntLayout, &counting.allocator, &entry), 0);
    EXPECT_EQ(entry, static_cast<uint64_t>(found)) << "inserting a present key finds its entry";
    // Filling the removed entries' room compacts: the live keys stay in insertion order.
    for (int32_t key = 200; key < 260; ++key)
        ASSERT_EQ(vernonRuntimeHostDictInsert(&dict, &key, &kIntLayout, &counting.allocator, &entry), 0);
    EXPECT_EQ(dict.size, 110u);
    std::vector<int32_t> order;
    for (uint64_t index = 0; index < dict.used; ++index)
        if (intEntry(dict, index)->hash != VH_DICT_REMOVED)
            order.push_back(intEntry(dict, index)->key);
    ASSERT_EQ(order.size(), 110u);
    EXPECT_EQ(order[0], 1);
    EXPECT_EQ(order[49], 99);
    EXPECT_EQ(order[50], 200);
    EXPECT_EQ(order[109], 259);

    vernonRuntimeHostDictClear(&dict, &kIntLayout, &counting.allocator);
    EXPECT_EQ(dict.size, 0u);
    EXPECT_EQ(vernonRuntimeHostDictFind(&dict, &seven, &kIntLayout), -1);
    vernonRuntimeHostDictRelease(&dict, &kIntLayout, &counting.allocator);
    EXPECT_EQ(counting.live, 0);
    EXPECT_EQ(dict.capacity, 0u);
}

TEST(RuntimeCApi, HostDictsOwnTheirStrKeys) {
    struct StrEntry {
        uint64_t hash;
        VhList key;
        float value;
    };
    constexpr VernonHostDictLayout layout{sizeof(StrEntry), alignof(StrEntry),         offsetof(StrEntry, key),
                                          sizeof(VhList),   offsetof(StrEntry, value), VERNON_HOST_DICT_KEY_STR};
    CountingAllocator counting;
    VhDict dict{};
    std::string text = "speed";
    const uint64_t view[2] = {static_cast<uint64_t>(reinterpret_cast<uintptr_t>(text.data())), text.size()};
    uint64_t entry = 0;
    ASSERT_EQ(vernonRuntimeHostDictInsert(&dict, view, &layout, &counting.allocator, &entry), 0);
    text = "XXXXX";
    const char *kept = "speed";
    const uint64_t probe[2] = {static_cast<uint64_t>(reinterpret_cast<uintptr_t>(kept)), 5};
    EXPECT_EQ(vernonRuntimeHostDictFind(&dict, probe, &layout), static_cast<int64_t>(entry))
        << "the dict copies the key's bytes";
    const uint64_t empty[2] = {0, 0};
    ASSERT_EQ(vernonRuntimeHostDictInsert(&dict, empty, &layout, &counting.allocator, &entry), 0);
    EXPECT_EQ(vernonRuntimeHostDictFind(&dict, empty, &layout), static_cast<int64_t>(entry));
    EXPECT_EQ(counting.live, 3) << "entries, slots, and the one nonempty key";
    EXPECT_EQ(vernonRuntimeHostDictRemove(&dict, probe, &layout, &counting.allocator), 1);
    EXPECT_EQ(counting.live, 2);
    vernonRuntimeHostDictRelease(&dict, &layout, &counting.allocator);
    EXPECT_EQ(counting.live, 0);
}

namespace {

VernonHostFormatSpec spec(char align, uint32_t width, char type = 0, int32_t precision = -1, uint32_t fill = 0,
                          char sign = 0) {
    VernonHostFormatSpec result{};
    result.fill = fill;
    result.align = static_cast<uint8_t>(align);
    result.sign = static_cast<uint8_t>(sign);
    result.type = static_cast<uint8_t>(type);
    result.width = width;
    result.precision = precision;
    return result;
}

template <typename Append> std::string formatted(Append append) {
    VhList text{};
    EXPECT_EQ(append(&text), 0);
    std::string result(static_cast<const char *>(vh_list_data(&text)), static_cast<size_t>(text.size));
    vernonRuntimeHostListRelease(&text, 1, 1, nullptr);
    return result;
}

std::string real(double value, const VernonHostFormatSpec *format = nullptr) {
    return formatted([&](VhList *text) { return vernonRuntimeHostStrAppendFloat(text, value, format, nullptr); });
}

std::string integer(int64_t value, const VernonHostFormatSpec *format = nullptr, bool is_signed = true) {
    return formatted([&](VhList *text) {
        return vernonRuntimeHostStrAppendInteger(text, static_cast<uint64_t>(value), is_signed, format, nullptr);
    });
}

} // namespace

// The expected texts are Python's repr and format() of the same values.
TEST(RuntimeCApi, HostStrFormattingMatchesPython) {
    EXPECT_EQ(real(0.1), "0.1");
    EXPECT_EQ(real(static_cast<double>(0.1f)), "0.10000000149011612");
    EXPECT_EQ(real(1e16), "1e+16");
    EXPECT_EQ(real(1e15), "1000000000000000.0");
    EXPECT_EQ(real(1e-5), "1e-05");
    EXPECT_EQ(real(0.0001), "0.0001");
    EXPECT_EQ(real(-0.0), "-0.0");
    EXPECT_EQ(real(2.5e-8), "2.5e-08");
    EXPECT_EQ(real(123456789.0), "123456789.0");
    EXPECT_EQ(real(1.5e300), "1.5e+300");
    EXPECT_EQ(real(5e-324), "5e-324");
    EXPECT_EQ(real(1e22), "1e+22");
    EXPECT_EQ(real(0.5), "0.5");
    EXPECT_EQ(real(std::numeric_limits<double>::infinity()), "inf");

    const VernonHostFormatSpec fixed3 = spec(0, 0, 'f', 3);
    EXPECT_EQ(real(0.1, &fixed3), "0.100");
    const VernonHostFormatSpec exponent2 = spec(0, 0, 'e', 2);
    EXPECT_EQ(real(0.1, &exponent2), "1.00e-01");
    const VernonHostFormatSpec right10 = spec('>', 10);
    EXPECT_EQ(real(1.0, &right10), "       1.0");
    const VernonHostFormatSpec plus = spec(0, 0, 0, -1, 0, '+');
    EXPECT_EQ(real(std::numeric_limits<double>::quiet_NaN(), &plus), "+nan");
    const VernonHostFormatSpec upper = spec(0, 0, 'E');
    EXPECT_EQ(real(1.5, &upper), "1.500000E+00");
    EXPECT_EQ(real(std::numeric_limits<double>::infinity(), &upper), "INF");
    const VernonHostFormatSpec zero_padded = spec('=', 8, 'f', 2, '0');
    EXPECT_EQ(real(-1.5, &zero_padded), "-0001.50");
    const VernonHostFormatSpec space = spec(0, 0, 0, -1, 0, ' ');
    EXPECT_EQ(real(-0.0, &space), "-0.0");
    EXPECT_EQ(real(3.0, &space), " 3.0");

    const VernonHostFormatSpec padded = spec('=', 4, 'd', -1, '0');
    EXPECT_EQ(integer(-3, &padded), "-003");
    const VernonHostFormatSpec hex = spec(0, 0, 'x');
    EXPECT_EQ(integer(255, &hex), "ff");
    const VernonHostFormatSpec centered = spec('^', 8, 'X', -1, '#');
    EXPECT_EQ(integer(255, &centered), "###FF###");
    const VernonHostFormatSpec width5 = spec(0, 5, 'd');
    EXPECT_EQ(integer(-9, &width5), "   -9");
    const VernonHostFormatSpec signed_plus = spec(0, 0, 'd', -1, 0, '+');
    EXPECT_EQ(integer(5, &signed_plus), "+5");
    EXPECT_EQ(integer(std::numeric_limits<int64_t>::min()), "-9223372036854775808");
    EXPECT_EQ(integer(-1, nullptr, false), "18446744073709551615");
    const VernonHostFormatSpec binary = spec(0, 0, 'b');
    EXPECT_EQ(integer(10, &binary), "1010");
    const VernonHostFormatSpec octal = spec(0, 0, 'o');
    EXPECT_EQ(integer(8, &octal), "10");

    auto text = [](std::string_view value, const VernonHostFormatSpec *format) {
        return formatted(
            [&](VhList *out) { return vernonRuntimeHostStrAppend(out, value.data(), value.size(), format, nullptr); });
    };
    const VernonHostFormatSpec star6 = spec('^', 6, 0, -1, '*');
    EXPECT_EQ(text("ab", &star6), "**ab**");
    const VernonHostFormatSpec star5 = spec('^', 5, 0, -1, '*');
    EXPECT_EQ(text("ab", &star5), "*ab**");
    const VernonHostFormatSpec right3 = spec('>', 3);
    EXPECT_EQ(text("\xC3\xA9", &right3), "  \xC3\xA9") << "width counts code points";
    const VernonHostFormatSpec dotted = spec(0, 3, 0, -1, 0xB7);
    EXPECT_EQ(text("x", &dotted), "x\xC2\xB7\xC2\xB7") << "text aligns left, with a multibyte fill";

    VhList grown{};
    ASSERT_EQ(vernonRuntimeHostStrAppend(&grown, "abc", 3, nullptr, nullptr), 0);
    for (int round = 0; round < 4; ++round)
        ASSERT_EQ(vernonRuntimeHostStrAppend(&grown, static_cast<const char *>(vh_list_data(&grown)), grown.size,
                                             nullptr, nullptr),
                  0);
    EXPECT_EQ(grown.size, 48u) << "appending a str to itself survives the growth";
    EXPECT_EQ(std::string(static_cast<const char *>(vh_list_data(&grown)) + 45, 3), "abc");
    vernonRuntimeHostListRelease(&grown, 1, 1, nullptr);
}

TEST(RuntimeCApi, HostContainerClassesChangeTheRecordsThroughTheRuntime) {
    CountingAllocator counting;
    const VhAllocator *slot = &counting.allocator;
    VhModule module{};
    module.struct_size = sizeof(VhModule);
    module.allocator = &slot;
    const vh::Allocator allocator(module);
    EXPECT_EQ(allocator.get(), &counting.allocator) << "a module's allocator is its slot's";

    vh::Str name;
    ASSERT_TRUE(name.assign(allocator, "speed"));
    EXPECT_TRUE(name == "speed");
    ASSERT_TRUE(name.assign(allocator, name.view().substr(2))) << "a str may take its own bytes";
    EXPECT_EQ(name.view(), "eed");

    vh::List<int32_t> values;
    for (int32_t value = 0; value < 10; ++value)
        ASSERT_TRUE(values.push_back(allocator, value));
    ASSERT_TRUE(values.push_back(allocator, values[9])) << "an element of the list survives the growth it causes";
    int32_t sum = 0;
    for (const int32_t value : values)
        sum += value;
    EXPECT_EQ(sum, 54);
    EXPECT_EQ(values.get(11), nullptr);
    values.pop_back();
    EXPECT_EQ(values.size(), 10u);

    vh::Dict<vh::Str, int32_t> levels;
    {
        std::string key = "a";
        ASSERT_TRUE(levels.assign(allocator, key, 1));
        key = "z";
    }
    ASSERT_TRUE(levels.assign(allocator, "b", 2));
    ASSERT_TRUE(levels.assign(allocator, "c", 3));
    EXPECT_TRUE(levels.erase(allocator, "b"));
    EXPECT_FALSE(levels.erase(allocator, "b"));
    int32_t *added = levels.emplace(allocator, "d");
    ASSERT_NE(added, nullptr);
    EXPECT_EQ(*added, 0) << "a new value starts zeroed";
    *added = 4;
    std::string order;
    for (auto [key, level] : levels)
        order += std::string(key) + std::to_string(level);
    EXPECT_EQ(order, "a1c3d4") << "insertion order, without removed entries";
    ASSERT_NE(levels.find("a"), nullptr) << "the dict owns a copy of each key";
    EXPECT_FALSE(levels.contains("z"));

    struct Handle {
        uint64_t bits;
    };
    vh::Dict<Handle, float[2]> places;
    const float place[2] = {1.0f, 2.0f};
    ASSERT_TRUE(places.assign(allocator, Handle{5}, place));
    const float (*found)[2] = places.find(Handle{5});
    ASSERT_NE(found, nullptr);
    EXPECT_EQ((*found)[1], 2.0f);

    const VernonHostDictLayout layout = vh::Dict<int32_t, int64_t>::layout();
    EXPECT_EQ(layout.entry_size, kIntLayout.entry_size);
    EXPECT_EQ(layout.key_offset, kIntLayout.key_offset);
    EXPECT_EQ(layout.value_offset, kIntLayout.value_offset);

    VhAllocator exhausted = counting.allocator;
    exhausted.allocate = [](void *, uint64_t, uint64_t) -> void * { return nullptr; };
    vh::List<int32_t> empty;
    EXPECT_FALSE(empty.push_back(&exhausted, 1));
    EXPECT_EQ(empty.size(), 0u) << "a failed growth leaves the list unchanged";
    EXPECT_EQ(levels.emplace(&exhausted, "e"), nullptr);
    EXPECT_EQ(levels.size(), 3u);

    name.release(allocator);
    values.release(allocator);
    levels.release(allocator);
    places.release(allocator);
    EXPECT_EQ(counting.live, 0);
}
