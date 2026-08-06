#include "VernonRuntime.hpp"
#include "runtime/autodiff/host_effect_transaction.h"
#include "runtime/autodiff/host_tape_allocator.h"
#include "runtime/autodiff/runtime_autodiff_internal.h"
#include "runtime/content_hash.h"
#include "runtime/runtime_state.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <new>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

thread_local VernonAdTapeReadLeafCallback instrumentedReadLeafTarget = nullptr;
thread_local size_t instrumentedReadLeafCalls = 0;

VernonAdTapeAllocatorStatus instrumentedReadLeaf(VernonAdTapeAllocator *allocator, VernonAdRegionHandle region,
                                                 size_t recordIndex, size_t leafOffset, void *data, size_t byteSize) {
    ++instrumentedReadLeafCalls;
    return instrumentedReadLeafTarget(allocator, region, recordIndex, leafOffset, data, byteSize);
}

std::string lastError(const VernonRuntimeContext *context) {
    const VernonStringView error = vernonRuntimeGetLastError(context);
    return error.data ? std::string(error.data, error.size) : std::string();
}

VernonStatus squareForward(const VernonCpuInvocation *invocation) {
    if (!invocation || invocation->arguments_size != 2 * sizeof(uint64_t) || invocation->results_size != sizeof(float))
        return VERNON_STATUS_INVALID_ARGUMENT;
    float x = 0.0f;
    VernonAdTapeAllocator *allocator = nullptr;
    std::memcpy(&x, invocation->arguments, sizeof(x));
    std::memcpy(&allocator, static_cast<const std::byte *>(invocation->arguments) + sizeof(uint64_t),
                sizeof(VernonAdTapeAllocator *));
    VernonAdRegionHandle region = VERNON_AD_INVALID_REGION_HANDLE;
    VernonAdRecordHandle record = VERNON_AD_INVALID_RECORD_HANDLE;
    if (!allocator ||
        allocator->begin_region(allocator, VERNON_AD_INVALID_REGION_HANDLE, &region) != VERNON_AD_TAPE_ALLOCATOR_OK ||
        allocator->reserve_record(allocator, region, sizeof(x), alignof(float), 0, &record) !=
            VERNON_AD_TAPE_ALLOCATOR_OK ||
        allocator->write_leaf(allocator, record, 0, &x, sizeof(x)) != VERNON_AD_TAPE_ALLOCATOR_OK ||
        allocator->end_region(allocator, region, 1, 0) != VERNON_AD_TAPE_ALLOCATOR_OK ||
        allocator->seal(allocator) != VERNON_AD_TAPE_ALLOCATOR_OK)
        return VERNON_STATUS_INVALID_ARGUMENT;
    const float result = x * x;
    std::memcpy(invocation->results, &result, sizeof(result));
    return VERNON_STATUS_OK;
}

VernonStatus squareBackward(const VernonCpuInvocation *invocation) {
    if (!invocation || invocation->arguments_size != 3 * sizeof(uint64_t) || invocation->results_size != sizeof(float))
        return VERNON_STATUS_INVALID_ARGUMENT;
    VernonAdTapeAllocator *allocator = nullptr;
    VernonAdRegionHandle region = VERNON_AD_INVALID_REGION_HANDLE;
    float x = 0.0f;
    float seed = 0.0f;
    std::memcpy(&allocator, invocation->arguments, sizeof(VernonAdTapeAllocator *));
    std::memcpy(&region, static_cast<const std::byte *>(invocation->arguments) + sizeof(uint64_t), sizeof(region));
    std::memcpy(&seed, static_cast<const std::byte *>(invocation->arguments) + 2 * sizeof(uint64_t), sizeof(seed));
    if (!allocator || allocator->read_leaf(allocator, region, 0, 0, &x, sizeof(x)) != VERNON_AD_TAPE_ALLOCATOR_OK)
        return VERNON_STATUS_INVALID_ARGUMENT;
    const float gradient = 2.0f * x * seed;
    std::memcpy(invocation->results, &gradient, sizeof(gradient));
    return VERNON_STATUS_OK;
}

VernonStatus tensorSquareForward(const VernonCpuInvocation *invocation) {
    if (!invocation || invocation->arguments_size != 2 * sizeof(uint64_t) ||
        invocation->results_size != 2 * sizeof(float))
        return VERNON_STATUS_INVALID_ARGUMENT;
    float x[2]{};
    VernonAdTapeAllocator *allocator = nullptr;
    std::memcpy(x, invocation->arguments, sizeof(x));
    std::memcpy(&allocator, static_cast<const std::byte *>(invocation->arguments) + sizeof(uint64_t),
                sizeof(VernonAdTapeAllocator *));
    VernonAdRegionHandle region = VERNON_AD_INVALID_REGION_HANDLE;
    VernonAdRecordHandle record = VERNON_AD_INVALID_RECORD_HANDLE;
    if (!allocator ||
        allocator->begin_region(allocator, VERNON_AD_INVALID_REGION_HANDLE, &region) != VERNON_AD_TAPE_ALLOCATOR_OK ||
        allocator->reserve_record(allocator, region, sizeof(x), alignof(float), 0, &record) !=
            VERNON_AD_TAPE_ALLOCATOR_OK ||
        allocator->write_leaf(allocator, record, 0, x, sizeof(x)) != VERNON_AD_TAPE_ALLOCATOR_OK ||
        allocator->end_region(allocator, region, 1, 0) != VERNON_AD_TAPE_ALLOCATOR_OK ||
        allocator->seal(allocator) != VERNON_AD_TAPE_ALLOCATOR_OK)
        return VERNON_STATUS_INVALID_ARGUMENT;
    const float results[]{x[0] * x[0], x[1] * x[1]};
    std::memcpy(invocation->results, results, sizeof(results));
    return VERNON_STATUS_OK;
}

VernonStatus tensorSquareBackward(const VernonCpuInvocation *invocation) {
    if (!invocation || invocation->arguments_size != 3 * sizeof(uint64_t) ||
        invocation->results_size != 2 * sizeof(float))
        return VERNON_STATUS_INVALID_ARGUMENT;
    VernonAdTapeAllocator *allocator = nullptr;
    VernonAdRegionHandle region = VERNON_AD_INVALID_REGION_HANDLE;
    float x[2]{};
    float seed[2]{};
    std::memcpy(&allocator, invocation->arguments, sizeof(VernonAdTapeAllocator *));
    std::memcpy(&region, static_cast<const std::byte *>(invocation->arguments) + sizeof(uint64_t), sizeof(region));
    std::memcpy(seed, static_cast<const std::byte *>(invocation->arguments) + 2 * sizeof(uint64_t), sizeof(seed));
    if (!allocator || allocator->read_leaf(allocator, region, 0, 0, x, sizeof(x)) != VERNON_AD_TAPE_ALLOCATOR_OK)
        return VERNON_STATUS_INVALID_ARGUMENT;
    const float gradients[]{2.0f * x[0] * seed[0], 2.0f * x[1] * seed[1]};
    std::memcpy(invocation->results, gradients, sizeof(gradients));
    return VERNON_STATUS_OK;
}

VernonStatus aggregateForward(const VernonCpuInvocation *invocation) {
    if (!invocation || invocation->arguments_size != 3 * sizeof(uint64_t) || invocation->results_size != sizeof(float))
        return VERNON_STATUS_INVALID_ARGUMENT;
    float arguments[3]{};
    VernonAdTapeAllocator *allocator = nullptr;
    std::memcpy(arguments, invocation->arguments, sizeof(arguments));
    std::memcpy(&allocator, static_cast<const std::byte *>(invocation->arguments) + 2 * sizeof(uint64_t),
                sizeof(VernonAdTapeAllocator *));
    VernonAdRegionHandle region = VERNON_AD_INVALID_REGION_HANDLE;
    VernonAdRecordHandle record = VERNON_AD_INVALID_RECORD_HANDLE;
    if (!allocator ||
        allocator->begin_region(allocator, VERNON_AD_INVALID_REGION_HANDLE, &region) != VERNON_AD_TAPE_ALLOCATOR_OK ||
        allocator->reserve_record(allocator, region, 2 * sizeof(float), alignof(float), 0, &record) !=
            VERNON_AD_TAPE_ALLOCATOR_OK ||
        allocator->write_leaf(allocator, record, 0, arguments, 2 * sizeof(float)) != VERNON_AD_TAPE_ALLOCATOR_OK ||
        allocator->end_region(allocator, region, 1, 0) != VERNON_AD_TAPE_ALLOCATOR_OK ||
        allocator->seal(allocator) != VERNON_AD_TAPE_ALLOCATOR_OK)
        return VERNON_STATUS_INVALID_ARGUMENT;
    const float result = arguments[0] * arguments[1] * arguments[2];
    std::memcpy(invocation->results, &result, sizeof(result));
    return VERNON_STATUS_OK;
}

VernonStatus aggregateBackward(const VernonCpuInvocation *invocation) {
    if (!invocation || invocation->arguments_size != 2 * sizeof(uint64_t) + sizeof(float) ||
        invocation->results_size != sizeof(float))
        return VERNON_STATUS_INVALID_ARGUMENT;
    VernonAdTapeAllocator *allocator = nullptr;
    VernonAdRegionHandle region = VERNON_AD_INVALID_REGION_HANDLE;
    float tape[2]{};
    float seed = 0.0f;
    std::memcpy(&allocator, invocation->arguments, sizeof(VernonAdTapeAllocator *));
    std::memcpy(&region, static_cast<const std::byte *>(invocation->arguments) + sizeof(uint64_t), sizeof(region));
    std::memcpy(&seed, static_cast<const std::byte *>(invocation->arguments) + 2 * sizeof(uint64_t), sizeof(seed));
    if (!allocator || allocator->read_leaf(allocator, region, 0, 0, tape, sizeof(tape)) != VERNON_AD_TAPE_ALLOCATOR_OK)
        return VERNON_STATUS_INVALID_ARGUMENT;
    const float gradient = tape[0] * tape[1] * seed;
    std::memcpy(invocation->results, &gradient, sizeof(gradient));
    return VERNON_STATUS_OK;
}

VernonStatus halfIdentityForward(const VernonCpuInvocation *invocation) {
    if (!invocation || invocation->arguments_size != 2 * sizeof(uint64_t) || invocation->results_size != 2)
        return VERNON_STATUS_INVALID_ARGUMENT;
    VernonAdTapeAllocator *allocator = nullptr;
    std::memcpy(&allocator, static_cast<const std::byte *>(invocation->arguments) + sizeof(uint64_t),
                sizeof(VernonAdTapeAllocator *));
    VernonAdRegionHandle region = VERNON_AD_INVALID_REGION_HANDLE;
    VernonAdRecordHandle record = VERNON_AD_INVALID_RECORD_HANDLE;
    if (!allocator ||
        allocator->begin_region(allocator, VERNON_AD_INVALID_REGION_HANDLE, &region) != VERNON_AD_TAPE_ALLOCATOR_OK ||
        allocator->reserve_record(allocator, region, 2, 2, 0, &record) != VERNON_AD_TAPE_ALLOCATOR_OK ||
        allocator->write_leaf(allocator, record, 0, invocation->arguments, 2) != VERNON_AD_TAPE_ALLOCATOR_OK ||
        allocator->end_region(allocator, region, 1, 0) != VERNON_AD_TAPE_ALLOCATOR_OK ||
        allocator->seal(allocator) != VERNON_AD_TAPE_ALLOCATOR_OK)
        return VERNON_STATUS_INVALID_ARGUMENT;
    std::memcpy(invocation->results, invocation->arguments, 2);
    return VERNON_STATUS_OK;
}

VernonStatus halfIdentityBackward(const VernonCpuInvocation *invocation) {
    if (!invocation || invocation->arguments_size != 2 * sizeof(uint64_t) + sizeof(float) ||
        invocation->results_size != sizeof(float))
        return VERNON_STATUS_INVALID_ARGUMENT;
    std::memcpy(invocation->results, static_cast<const std::byte *>(invocation->arguments) + 2 * sizeof(uint64_t),
                sizeof(float));
    return VERNON_STATUS_OK;
}

VernonStatus allocatorSquareForward(const VernonCpuInvocation *invocation) {
    if (!invocation || invocation->arguments_size != 2 * sizeof(uint64_t) || invocation->results_size != sizeof(float))
        return VERNON_STATUS_INVALID_ARGUMENT;
    float x = 0.0f;
    VernonAdTapeAllocator *allocator = nullptr;
    std::memcpy(&x, invocation->arguments, sizeof(x));
    std::memcpy(&allocator, static_cast<const std::byte *>(invocation->arguments) + sizeof(uint64_t),
                sizeof(VernonAdTapeAllocator *));
    if (!allocator || allocator->struct_size != sizeof(*allocator) ||
        allocator->abi_version != VERNON_AD_TAPE_ALLOCATOR_ABI_VERSION)
        return VERNON_STATUS_INVALID_ARGUMENT;
    VernonAdRegionHandle region = VERNON_AD_INVALID_REGION_HANDLE;
    if (allocator->begin_region(allocator, VERNON_AD_INVALID_REGION_HANDLE, &region) != VERNON_AD_TAPE_ALLOCATOR_OK)
        return VERNON_STATUS_INVALID_ARGUMENT;
    VernonAdRecordHandle record = VERNON_AD_INVALID_RECORD_HANDLE;
    if (allocator->reserve_record(allocator, region, sizeof(x), alignof(float), 0, &record) !=
        VERNON_AD_TAPE_ALLOCATOR_OK)
        return VERNON_STATUS_INVALID_ARGUMENT;
    if (allocator->write_leaf(allocator, record, 0, &x, sizeof(x)) != VERNON_AD_TAPE_ALLOCATOR_OK)
        return VERNON_STATUS_INVALID_ARGUMENT;
    if (allocator->end_region(allocator, region, 1, 0) != VERNON_AD_TAPE_ALLOCATOR_OK ||
        allocator->seal(allocator) != VERNON_AD_TAPE_ALLOCATOR_OK)
        return VERNON_STATUS_INVALID_ARGUMENT;
    const float result = x * x;
    std::memcpy(invocation->results, &result, sizeof(result));
    return VERNON_STATUS_OK;
}

VernonStatus allocatorSquareBackward(const VernonCpuInvocation *invocation) {
    if (!invocation || invocation->arguments_size != 3 * sizeof(uint64_t) || invocation->results_size != sizeof(float))
        return VERNON_STATUS_INVALID_ARGUMENT;
    VernonAdTapeAllocator *allocator = nullptr;
    VernonAdRegionHandle root = VERNON_AD_INVALID_REGION_HANDLE;
    float seed = 0.0f;
    std::memcpy(&allocator, invocation->arguments, sizeof(VernonAdTapeAllocator *));
    std::memcpy(&root, static_cast<const std::byte *>(invocation->arguments) + sizeof(uint64_t), sizeof(root));
    std::memcpy(&seed, static_cast<const std::byte *>(invocation->arguments) + 2 * sizeof(uint64_t), sizeof(seed));
    float x = 0.0f;
    if (!allocator || allocator->read_leaf(allocator, root, 0, 0, &x, sizeof(x)) != VERNON_AD_TAPE_ALLOCATOR_OK)
        return VERNON_STATUS_INVALID_ARGUMENT;
    const float gradient = 2.0f * x * seed;
    std::memcpy(invocation->results, &gradient, sizeof(gradient));
    return VERNON_STATUS_OK;
}

VernonStatus allocatorSquareBackwardRejectsNegativeSeed(const VernonCpuInvocation *invocation) {
    if (!invocation || invocation->arguments_size != 3 * sizeof(uint64_t))
        return VERNON_STATUS_INVALID_ARGUMENT;
    float seed = 0.0f;
    std::memcpy(&seed, static_cast<const std::byte *>(invocation->arguments) + 2 * sizeof(uint64_t), sizeof(seed));
    return seed < 0.0f ? VERNON_STATUS_INVALID_ARGUMENT : allocatorSquareBackward(invocation);
}

VernonStatus storageTransactionForward(const VernonCpuInvocation *invocation, bool failCapture) {
    if (!invocation || invocation->arguments_size != 5 * sizeof(uint64_t) || invocation->results_size != sizeof(float))
        return VERNON_STATUS_INVALID_ARGUMENT;
    uintptr_t storageAddress = 0;
    VernonAdTapeAllocator *allocator = nullptr;
    std::memcpy(&storageAddress, invocation->arguments, sizeof(storageAddress));
    std::memcpy(&allocator, static_cast<const std::byte *>(invocation->arguments) + 4 * sizeof(uint64_t),
                sizeof(VernonAdTapeAllocator *));
    if (!storageAddress || !allocator)
        return VERNON_STATUS_INVALID_ARGUMENT;
    auto *storage = reinterpret_cast<float *>(storageAddress);
    const float x = *storage;
    *storage = x + 1.0f;
    VernonAdRegionHandle region = VERNON_AD_INVALID_REGION_HANDLE;
    VernonAdRecordHandle record = VERNON_AD_INVALID_RECORD_HANDLE;
    if (allocator->begin_region(allocator, VERNON_AD_INVALID_REGION_HANDLE, &region) != VERNON_AD_TAPE_ALLOCATOR_OK ||
        allocator->reserve_record(allocator, region, sizeof(x), alignof(float), 0, &record) !=
            VERNON_AD_TAPE_ALLOCATOR_OK ||
        allocator->write_leaf(allocator, record, 0, &x, sizeof(x)) != VERNON_AD_TAPE_ALLOCATOR_OK)
        return VERNON_STATUS_INVALID_ARGUMENT;
    if (failCapture)
        return VERNON_STATUS_INVALID_ARGUMENT;
    if (allocator->end_region(allocator, region, 1, 0) != VERNON_AD_TAPE_ALLOCATOR_OK ||
        allocator->seal(allocator) != VERNON_AD_TAPE_ALLOCATOR_OK)
        return VERNON_STATUS_INVALID_ARGUMENT;
    const float result = x * x;
    std::memcpy(invocation->results, &result, sizeof(result));
    return VERNON_STATUS_OK;
}

VernonStatus failingStorageForward(const VernonCpuInvocation *invocation) {
    return storageTransactionForward(invocation, true);
}

VernonStatus successfulStorageForward(const VernonCpuInvocation *invocation) {
    return storageTransactionForward(invocation, false);
}

VernonStatus throwingAutodiffForward(const VernonCpuInvocation *) { throw std::bad_alloc(); }

VernonStatus throwingAutodiffBackward(const VernonCpuInvocation *) {
    throw std::length_error("test pullback allocation failure");
}

VernonStatus aggregateStorageForward(const VernonCpuInvocation *invocation) {
    if (!invocation || invocation->arguments_size != 5 * sizeof(uint64_t) || invocation->results_size != sizeof(float))
        return VERNON_STATUS_INVALID_ARGUMENT;
    uintptr_t storageAddress = 0;
    VernonAdTapeAllocator *allocator = nullptr;
    std::memcpy(&storageAddress, invocation->arguments, sizeof(storageAddress));
    std::memcpy(&allocator, static_cast<const std::byte *>(invocation->arguments) + 4 * sizeof(uint64_t),
                sizeof(VernonAdTapeAllocator *));
    struct Element {
        float value;
        int32_t count;
    };
    auto *storage = reinterpret_cast<Element *>(storageAddress);
    if (!storage || !allocator)
        return VERNON_STATUS_INVALID_ARGUMENT;
    const float captured = storage[0].value;
    for (size_t index = 0; index < 2; ++index) {
        storage[index].value += static_cast<float>(storage[index].count);
        storage[index].count += 10;
    }
    VernonAdRegionHandle region = VERNON_AD_INVALID_REGION_HANDLE;
    VernonAdRecordHandle record = VERNON_AD_INVALID_RECORD_HANDLE;
    if (allocator->begin_region(allocator, VERNON_AD_INVALID_REGION_HANDLE, &region) != VERNON_AD_TAPE_ALLOCATOR_OK ||
        allocator->reserve_record(allocator, region, sizeof(captured), alignof(float), 0, &record) !=
            VERNON_AD_TAPE_ALLOCATOR_OK ||
        allocator->write_leaf(allocator, record, 0, &captured, sizeof(captured)) != VERNON_AD_TAPE_ALLOCATOR_OK ||
        allocator->end_region(allocator, region, 1, 0) != VERNON_AD_TAPE_ALLOCATOR_OK ||
        allocator->seal(allocator) != VERNON_AD_TAPE_ALLOCATOR_OK)
        return VERNON_STATUS_INVALID_ARGUMENT;
    const float output = storage[0].value + storage[1].value;
    std::memcpy(invocation->results, &output, sizeof(output));
    return VERNON_STATUS_OK;
}

nlohmann::json leaf(size_t offset, size_t scalarCount = 1, const char *dtype = "f32", bool shaped = false) {
    nlohmann::json result = {
        {"path", nlohmann::json::array()}, {"dtype", dtype}, {"scalar_count", scalarCount}, {"byte_offset", offset}};
    if (scalarCount != 1 || shaped)
        result["shape"] = nlohmann::json::array({scalarCount});
    return result;
}

nlohmann::json transportRoot(const nlohmann::json &leaves, size_t size, size_t alignment) {
    const auto scalarSize = [](const std::string &dtype) {
        return dtype == "f16" ? size_t{2} : dtype == "f64" ? size_t{8} : size_t{4};
    };
    const auto child = [&](const nlohmann::json &value) {
        const std::string dtype = value["dtype"].get<std::string>();
        const size_t count = value.value("scalar_count", size_t{1});
        const size_t offset = value.value("byte_offset", size_t{0});
        const size_t laneSize = scalarSize(dtype);
        if (count == 1)
            return nlohmann::json{{"kind", "scalar"},
                                  {"representation", dtype},
                                  {"offset", offset},
                                  {"size", laneSize},
                                  {"alignment", laneSize}};
        return nlohmann::json{{"kind", "array"},
                              {"offset", offset},
                              {"size", count * laneSize},
                              {"alignment", laneSize},
                              {"shape", nlohmann::json::array({count})},
                              {"byte_strides", nlohmann::json::array({laneSize})},
                              {"children", nlohmann::json::array({{{"kind", "scalar"},
                                                                   {"representation", dtype},
                                                                   {"offset", 0},
                                                                   {"size", laneSize},
                                                                   {"alignment", laneSize}}})}};
    };
    if (leaves.size() == 1) {
        nlohmann::json root = child(leaves[0]);
        root["offset"] = 0;
        root["size"] = size;
        root["alignment"] = alignment;
        return root;
    }
    nlohmann::json children = nlohmann::json::array();
    for (const nlohmann::json &value : leaves)
        children.push_back(child(value));
    return {{"kind", "product"},
            {"offset", 0},
            {"size", size},
            {"alignment", alignment},
            {"children", std::move(children)}};
}

nlohmann::json argument(const char *name, size_t offset, nlohmann::json leaves, size_t size = 4, size_t alignment = 4) {
    const std::string dtype =
        leaves.size() == 1 && leaves[0].contains("dtype") ? leaves[0]["dtype"].get<std::string>() : "";
    nlohmann::json root = transportRoot(leaves, size, alignment);
    return {{"kind", "scalar"},
            {"dtype", dtype},
            {"vernon.source_name", name},
            {"value_layout", {{"leaves", std::move(leaves)}}},
            {"physical_layouts",
             {{"host_value",
               {{"profile", "host_value"},
                {"kind", "cpu_call"},
                {"canonical_layout_hash", "test-layout"},
                {"frame_offset", offset},
                {"root", std::move(root)}}}}}};
}

nlohmann::json tapeAllocatorArgument(size_t offset) {
    return {{"kind", "builtin"},
            {"builtin", VERNON_AD_TAPE_ALLOCATOR_BUILTIN},
            {"physical_layouts",
             {{"host_value",
               {{"profile", "host_value"},
                {"kind", "cpu_call"},
                {"canonical_layout_hash", "ad-tape-allocator-v1"},
                {"frame_offset", offset},
                {"root",
                 {{"kind", "scalar"},
                  {"representation", "index"},
                  {"offset", uint64_t{0}},
                  {"size", sizeof(VernonAdTapeAllocator *)},
                  {"alignment", alignof(VernonAdTapeAllocator *)}}}}}}}};
}

nlohmann::json tapeRootRegionArgument(size_t offset) {
    return {{"kind", "builtin"},
            {"builtin", "ad_tape_root_region"},
            {"physical_layouts",
             {{"host_value",
               {{"profile", "host_value"},
                {"kind", "cpu_call"},
                {"canonical_layout_hash", "ad-tape-root-region-v1"},
                {"frame_offset", offset},
                {"root",
                 {{"kind", "scalar"},
                  {"representation", "index"},
                  {"offset", uint64_t{0}},
                  {"size", sizeof(VernonAdRegionHandle)},
                  {"alignment", alignof(VernonAdRegionHandle)}}}}}}}};
}

nlohmann::json tensorViewArgument(const char *name, size_t offset, const char *access) {
    return {{"kind", "tensor"},
            {"access", access},
            {"dtype", "f32"},
            {"rank", 1},
            {"vernon.source_name", name},
            {"source_shape", nlohmann::json::array({1})},
            {"element_layout",
             {{"alignment", alignof(float)},
              {"byte_size", sizeof(float)},
              {"layout_hash", "test-layout"},
              {"logical_type", "f32"},
              {"leaves", nlohmann::json::array({leaf(0)})}}},
            {"physical_layouts",
             {{"host_value",
               {{"profile", "host_value"},
                {"kind", "resource_binding"},
                {"resource_kind", "tensor_view_descriptor"},
                {"frame_offset", offset},
                {"size", 4 * sizeof(uint64_t)},
                {"alignment", alignof(uint64_t)}}}}}};
}

nlohmann::json aggregateTensorViewArgument(const char *name, size_t offset, const char *access) {
    nlohmann::json valueLeaf = leaf(0);
    valueLeaf["path"] = nlohmann::json::array({"value"});
    nlohmann::json countLeaf = leaf(sizeof(float), 1, "i32");
    countLeaf["path"] = nlohmann::json::array({"count"});
    return {{"kind", "tensor"},
            {"access", access},
            {"dtype", ""},
            {"rank", 1},
            {"vernon.source_name", name},
            {"source_shape", nlohmann::json::array({2})},
            {"element_layout",
             {{"alignment", alignof(float)},
              {"byte_size", 2 * sizeof(uint32_t)},
              {"layout_hash", "aggregate-test-layout"},
              {"logical_type", "!vernon.struct<value:f32,count:i32>"},
              {"leaves", nlohmann::json::array({valueLeaf, countLeaf})}}},
            {"physical_layouts",
             {{"host_value",
               {{"profile", "host_value"},
                {"kind", "resource_binding"},
                {"resource_kind", "tensor_view_descriptor"},
                {"frame_offset", offset},
                {"size", 4 * sizeof(uint64_t)},
                {"alignment", alignof(uint64_t)}}}}}};
}

nlohmann::json profileReflection(const char *entry, size_t argumentBytes, size_t resultBytes, nlohmann::json arguments,
                                 nlohmann::json resultLeaves, size_t resultAlignment = 4) {
    nlohmann::json resultRoot = transportRoot(resultLeaves, resultBytes, resultAlignment);
    return {
        {"compiler_contract_version", VERNON_COMPILER_CONTRACT_VERSION},
        {"pipeline_version", VERNON_PIPELINE_VERSION},
        {"entries", nlohmann::json::array(
                        {{{"name", entry},
                          {"workgroup_size", nlohmann::json::array({1, 1, 1})},
                          {"physical_layouts",
                           {{"host_value",
                             {{"profile", "host_value"},
                              {"packed_arguments_size", argumentBytes},
                              {"packed_results_size", resultBytes}}}}},
                          {"arguments", std::move(arguments)},
                          {"results", nlohmann::json::array({{{"value_layout", {{"leaves", std::move(resultLeaves)}}},
                                                              {"physical_layouts",
                                                               {{"host_value",
                                                                 {{"profile", "host_value"},
                                                                  {"kind", "cpu_call"},
                                                                  {"canonical_layout_hash", "test-layout"},
                                                                  {"frame_offset", uint64_t{0}},
                                                                  {"root", std::move(resultRoot)}}}}}}})}}})}};
}

vernon::runtime::CpuNativeArtifact artifact(const std::filesystem::path &root, const char *entry, const char *symbol,
                                            nlohmann::json reflection) {
    vernon::runtime::CpuNativeArtifact result;
    result.root = root;
    result.relativeLibrary = "fixture.o";
    result.format = "relocatable_object";
    result.entry = entry;
    result.symbol = symbol;
#if defined(_WIN32)
    result.targetTriple = "x86_64-pc-windows-msvc";
    result.objectFormat = "coff";
#elif defined(__APPLE__)
#if defined(__aarch64__)
    result.targetTriple = "aarch64-apple-darwin";
#else
    result.targetTriple = "x86_64-apple-darwin";
#endif
    result.objectFormat = "macho";
#else
#if defined(__aarch64__)
    result.targetTriple = "aarch64-unknown-linux-gnu";
#else
    result.targetTriple = "x86_64-unknown-linux-gnu";
#endif
    result.objectFormat = "elf";
#endif
    constexpr char bytes[] = "registered object fixture";
    result.size = sizeof(bytes) - 1;
    result.sha256 = vernon::runtime::sha256Hex(bytes, sizeof(bytes) - 1);
    result.reflection = std::move(reflection);
    return result;
}

vernon::runtime::Stage reflectedStage(const char *entry, const nlohmann::json &reflection) {
    vernon::runtime::Stage result;
    result.stage = "compute";
    result.entry = entry;
    result.reflection = reflection.dump();
    return result;
}

vernon::runtime::Stage stage(vernon::runtime::CpuNativeArtifact artifact) {
    vernon::runtime::Stage result = reflectedStage(artifact.entry.c_str(), artifact.reflection);
    result.autodiff = vernon::runtime::AutodiffStageMetadata{"", "dynamic_v2", ""};
    result.cpuArtifact = std::move(artifact);
    return result;
}

void attachCpuAutodiff(VernonRuntimeContext &context, VernonLoadedPipeline &pipeline, vernon::runtime::Stage forward,
                       vernon::runtime::Stage backward, std::vector<std::string> gradientPaths) {
    std::shared_ptr<vernon::runtime::ad::Executable> executable;
    const bool created =
        vernon::runtime::ad::createCpuExecutable(context, forward, backward, gradientPaths, executable);
    ASSERT_TRUE(created) << lastError(&context);
    pipeline.autodiff = VernonLoadedAutodiff{std::move(executable)};
}

class MetadataExecutable final : public vernon::runtime::ad::HostExecutable {
public:
    explicit MetadataExecutable(vernon::runtime::ad::Signature signature) : signature_(std::move(signature)) {}

    const vernon::runtime::ad::Signature &signature() const override { return signature_; }

    VernonStatus forward(VernonLaunchSize, const VernonAdValueSet &, VernonAdValueSet &,
                         std::unique_ptr<vernon::runtime::ad::PullbackExecution> &) override {
        return VERNON_STATUS_UNSUPPORTED_TARGET;
    }

private:
    vernon::runtime::ad::Signature signature_;
};

TEST(RuntimeAutodiffTapeAllocator, HasFrozenVersionedAbi) {
    EXPECT_EQ(VERNON_AD_TAPE_ALLOCATOR_ABI_VERSION, 2u);
    EXPECT_EQ(sizeof(VernonAdTapeAllocatorStatus), 4u);
    EXPECT_EQ(sizeof(VernonAdRegionHandle), 8u);
    EXPECT_EQ(sizeof(VernonAdRecordHandle), 8u);
    EXPECT_EQ(sizeof(VernonAdTapeAllocator), sizeof(void *) == 8 ? 128u : 68u);
    EXPECT_EQ(offsetof(VernonAdTapeAllocator, abi_version), sizeof(void *) == 8 ? 8u : 4u);
    EXPECT_EQ(offsetof(VernonAdTapeAllocator, status), sizeof(void *) == 8 ? 12u : 8u);
    EXPECT_EQ(offsetof(VernonAdTapeAllocator, user_data), sizeof(void *) == 8 ? 16u : 12u);
    EXPECT_EQ(offsetof(VernonAdTapeAllocator, capacity_bytes), sizeof(void *) == 8 ? 24u : 16u);
    EXPECT_EQ(offsetof(VernonAdTapeAllocator, required_bytes), sizeof(void *) == 8 ? 32u : 20u);
    EXPECT_EQ(offsetof(VernonAdTapeAllocator, reset), sizeof(void *) == 8 ? 40u : 24u);
    EXPECT_EQ(offsetof(VernonAdTapeAllocator, begin_region), sizeof(void *) == 8 ? 48u : 28u);
    EXPECT_EQ(offsetof(VernonAdTapeAllocator, reserve_record), sizeof(void *) == 8 ? 56u : 32u);
    EXPECT_EQ(offsetof(VernonAdTapeAllocator, write_leaf), sizeof(void *) == 8 ? 64u : 36u);
    EXPECT_EQ(offsetof(VernonAdTapeAllocator, set_child), sizeof(void *) == 8 ? 72u : 40u);
    EXPECT_EQ(offsetof(VernonAdTapeAllocator, end_region), sizeof(void *) == 8 ? 80u : 44u);
    EXPECT_EQ(offsetof(VernonAdTapeAllocator, seal), sizeof(void *) == 8 ? 88u : 48u);
    EXPECT_EQ(offsetof(VernonAdTapeAllocator, read_leaf), sizeof(void *) == 8 ? 96u : 52u);
    EXPECT_EQ(offsetof(VernonAdTapeAllocator, read_child), sizeof(void *) == 8 ? 104u : 56u);
    EXPECT_EQ(offsetof(VernonAdTapeAllocator, read_executed_count), sizeof(void *) == 8 ? 112u : 60u);
    EXPECT_EQ(offsetof(VernonAdTapeAllocator, read_exit_kind), sizeof(void *) == 8 ? 120u : 64u);

    vernon::runtime::ad::HostDynamicTape storage;
    const VernonAdTapeAllocator &descriptor = storage.descriptor();
    EXPECT_NE(descriptor.reset, nullptr);
    EXPECT_NE(descriptor.begin_region, nullptr);
    EXPECT_NE(descriptor.reserve_record, nullptr);
    EXPECT_NE(descriptor.write_leaf, nullptr);
    EXPECT_NE(descriptor.set_child, nullptr);
    EXPECT_NE(descriptor.end_region, nullptr);
    EXPECT_NE(descriptor.seal, nullptr);
    EXPECT_NE(descriptor.read_leaf, nullptr);
    EXPECT_NE(descriptor.read_child, nullptr);
    EXPECT_NE(descriptor.read_executed_count, nullptr);
    EXPECT_NE(descriptor.read_exit_kind, nullptr);
    VernonAdTapeAllocator incompatibleSize = storage.descriptor();
    --incompatibleSize.struct_size;
    EXPECT_EQ(incompatibleSize.reset(&incompatibleSize), VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI);
    EXPECT_EQ(incompatibleSize.status, VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI);

    VernonAdTapeAllocator incompatibleVersion = storage.descriptor();
    ++incompatibleVersion.abi_version;
    EXPECT_EQ(incompatibleVersion.reset(&incompatibleVersion), VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI);
    EXPECT_EQ(incompatibleVersion.status, VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI);
}

TEST(RuntimeAutodiffEffectTransaction, CommitsOnceAndRejectsFurtherStaging) {
    float storage = 3.0f;
    float output = -1.0f;
    vernon::runtime::ad::HostEffectTransaction transaction(sizeof(output));
    auto *stagedStorage = reinterpret_cast<float *>(transaction.stageStorage(&storage, sizeof(storage), true));
    auto *stagedOutput = reinterpret_cast<float *>(transaction.stagedOutput());
    ASSERT_NE(stagedStorage, nullptr);
    ASSERT_NE(stagedOutput, nullptr);
    EXPECT_FLOAT_EQ(*stagedStorage, storage);
    *stagedStorage = 4.0f;
    *stagedOutput = 9.0f;

    EXPECT_TRUE(transaction.commit(&output));
    EXPECT_FLOAT_EQ(storage, 4.0f);
    EXPECT_FLOAT_EQ(output, 9.0f);
    EXPECT_EQ(transaction.stageStorage(&storage, sizeof(storage), true), nullptr);
    EXPECT_EQ(transaction.stagedOutput(), nullptr);
    EXPECT_FALSE(transaction.commit(&output));

    *stagedStorage = 7.0f;
    *stagedOutput = 11.0f;
    EXPECT_FLOAT_EQ(storage, 4.0f);
    EXPECT_FLOAT_EQ(output, 9.0f);

    vernon::runtime::ad::HostEffectTransaction discarded(sizeof(output));
    ASSERT_NE(discarded.stageStorage(&storage, sizeof(storage), true), nullptr);
    EXPECT_TRUE(discarded.discard());
    EXPECT_EQ(discarded.stageStorage(&storage, sizeof(storage), true), nullptr);
    EXPECT_EQ(discarded.stagedOutput(), nullptr);
    EXPECT_FALSE(discarded.commit(&output));
}

TEST(RuntimeAutodiff, CpuTapePolicyIsLazyForOrdinaryRuntimeContexts) {
    VernonRuntimeContext context;
    EXPECT_EQ(context.cpuTapePolicy, nullptr);
}

TEST(RuntimeAutodiffTapeAllocator, WritesReservedOffsetsAfterVectorGrowth) {
    vernon::runtime::ad::HostDynamicTape storage;
    VernonAdTapeAllocator &allocator = storage.descriptor();
    VernonAdRegionHandle root = VERNON_AD_INVALID_REGION_HANDLE;
    ASSERT_EQ(allocator.begin_region(&allocator, VERNON_AD_INVALID_REGION_HANDLE, &root), VERNON_AD_TAPE_ALLOCATOR_OK);
    VernonAdRecordHandle rootRecord = VERNON_AD_INVALID_RECORD_HANDLE;
    ASSERT_EQ(allocator.reserve_record(&allocator, root, sizeof(uint32_t), alignof(uint32_t), 0, &rootRecord),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    VernonAdRecordHandle growthRecord = VERNON_AD_INVALID_RECORD_HANDLE;
    ASSERT_EQ(allocator.reserve_record(&allocator, root, 4096, 8, 0, &growthRecord), VERNON_AD_TAPE_ALLOCATOR_OK);
    const uint32_t value = 0x12345678u;
    ASSERT_EQ(allocator.write_leaf(&allocator, rootRecord, 0, &value, sizeof(value)), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator.end_region(&allocator, root, 2, 7), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator.seal(&allocator), VERNON_AD_TAPE_ALLOCATOR_OK);
    std::shared_ptr<const vernon::runtime::ad::HostTapeSnapshot> snapshot = storage.takeSnapshot();
    ASSERT_NE(snapshot, nullptr);
    uint32_t read = 0;
    VernonAdTapeAllocator *reader = snapshot->descriptor();
    ASSERT_EQ(reader->read_leaf(reader, root, 0, 0, &read, sizeof(read)), VERNON_AD_TAPE_ALLOCATOR_OK);
    EXPECT_EQ(read, value);
}

TEST(RuntimeAutodiffTapeAllocator, DiagnosticVisitsEachIndexedRecordOnce) {
    const auto readCount = [](size_t recordCount) {
        vernon::runtime::ad::HostDynamicTape storage;
        VernonAdTapeAllocator &allocator = storage.descriptor();
        VernonAdRegionHandle root = VERNON_AD_INVALID_REGION_HANDLE;
        EXPECT_EQ(allocator.begin_region(&allocator, VERNON_AD_INVALID_REGION_HANDLE, &root),
                  VERNON_AD_TAPE_ALLOCATOR_OK);
        for (size_t index = 0; index < recordCount; ++index) {
            VernonAdRecordHandle record = VERNON_AD_INVALID_RECORD_HANDLE;
            EXPECT_EQ(allocator.reserve_record(&allocator, root, sizeof(uint32_t), alignof(uint32_t), 0, &record),
                      VERNON_AD_TAPE_ALLOCATOR_OK);
            const uint32_t value = static_cast<uint32_t>(index);
            EXPECT_EQ(allocator.write_leaf(&allocator, record, 0, &value, sizeof(value)), VERNON_AD_TAPE_ALLOCATOR_OK);
        }
        EXPECT_EQ(allocator.end_region(&allocator, root, recordCount, 0), VERNON_AD_TAPE_ALLOCATOR_OK);
        EXPECT_EQ(allocator.seal(&allocator), VERNON_AD_TAPE_ALLOCATOR_OK);
        std::shared_ptr<const vernon::runtime::ad::HostTapeSnapshot> snapshot = storage.takeSnapshot();
        EXPECT_NE(snapshot, nullptr);
        VernonAdTapeAllocator reader = *snapshot->descriptor();
        instrumentedReadLeafTarget = reader.read_leaf;
        instrumentedReadLeafCalls = 0;
        reader.read_leaf = instrumentedReadLeaf;
        for (size_t index = recordCount; index-- > 0;) {
            uint32_t value = 0;
            EXPECT_EQ(reader.read_leaf(&reader, root, index, 0, &value, sizeof(value)), VERNON_AD_TAPE_ALLOCATOR_OK);
            EXPECT_EQ(value, index);
        }
        instrumentedReadLeafTarget = nullptr;
        return instrumentedReadLeafCalls;
    };

    EXPECT_EQ(readCount(32), 32u);
    EXPECT_EQ(readCount(2048), 2048u);
}

TEST(RuntimeAutodiffTapeAllocator, ReportsDistinctFailuresAndExactRequiredBytes) {
    vernon::runtime::ad::HostDynamicTape bounded(8);
    VernonAdTapeAllocator &allocator = bounded.descriptor();
    VernonAdRegionHandle region = VERNON_AD_INVALID_REGION_HANDLE;
    ASSERT_EQ(allocator.begin_region(&allocator, VERNON_AD_INVALID_REGION_HANDLE, &region),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    VernonAdRecordHandle record = VERNON_AD_INVALID_RECORD_HANDLE;
    ASSERT_EQ(allocator.reserve_record(&allocator, region, 5, 4, 0, &record), VERNON_AD_TAPE_ALLOCATOR_OK);
    EXPECT_NE(record, VERNON_AD_INVALID_RECORD_HANDLE);
    EXPECT_EQ(allocator.required_bytes, 5u);

    EXPECT_EQ(allocator.reserve_record(&allocator, region, 8, 8, 0, &record),
              VERNON_AD_TAPE_ALLOCATOR_CAPACITY_EXHAUSTED);
    EXPECT_EQ(record, VERNON_AD_INVALID_RECORD_HANDLE);
    EXPECT_EQ(allocator.required_bytes, 16u);
    EXPECT_EQ(allocator.reserve_record(&allocator, region, 3, 4, 0, &record),
              VERNON_AD_TAPE_ALLOCATOR_CAPACITY_EXHAUSTED);
    EXPECT_EQ(allocator.required_bytes, 19u);
    EXPECT_EQ(allocator.end_region(&allocator, region, 0, 0), VERNON_AD_TAPE_ALLOCATOR_CAPACITY_EXHAUSTED);

    ASSERT_EQ(allocator.reset(&allocator), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator.begin_region(&allocator, VERNON_AD_INVALID_REGION_HANDLE, &region),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    EXPECT_EQ(allocator.reserve_record(&allocator, region, std::numeric_limits<size_t>::max(), 1, 0, &record),
              VERNON_AD_TAPE_ALLOCATOR_CAPACITY_EXHAUSTED);
    EXPECT_EQ(allocator.required_bytes, std::numeric_limits<size_t>::max());
    EXPECT_EQ(allocator.reserve_record(&allocator, region, 1, 1, 0, &record),
              VERNON_AD_TAPE_ALLOCATOR_ARITHMETIC_OVERFLOW);

    vernon::runtime::ad::HostDynamicTape unbounded(
        std::numeric_limits<size_t>::max(),
        std::make_shared<vernon::runtime::ad::HostTapeMemoryPolicy>(std::numeric_limits<size_t>::max(),
                                                                    std::numeric_limits<size_t>::max()));
    VernonAdTapeAllocator &hostFailure = unbounded.descriptor();
    ASSERT_EQ(hostFailure.begin_region(&hostFailure, VERNON_AD_INVALID_REGION_HANDLE, &region),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    const size_t maximumStorageSize = std::vector<std::byte>{}.max_size();
    ASSERT_LT(maximumStorageSize, std::numeric_limits<size_t>::max());
    const size_t unsupportedStorageSize = maximumStorageSize + 1;
    EXPECT_EQ(hostFailure.reserve_record(&hostFailure, region, unsupportedStorageSize, 1, 0, &record),
              VERNON_AD_TAPE_ALLOCATOR_HOST_ALLOCATION_FAILURE);
    EXPECT_EQ(hostFailure.required_bytes, unsupportedStorageSize);
    EXPECT_EQ(hostFailure.seal(&hostFailure), VERNON_AD_TAPE_ALLOCATOR_HOST_ALLOCATION_FAILURE);
    EXPECT_EQ(hostFailure.reset(&hostFailure), VERNON_AD_TAPE_ALLOCATOR_OK);
}

TEST(RuntimeAutodiffTapeAllocator, RetainsAndReleasesSharedMemoryPolicyChargeWithSnapshot) {
    auto invocationPolicy = std::make_shared<vernon::runtime::ad::HostTapeMemoryPolicy>(8, 64);
    vernon::runtime::ad::HostDynamicTape invocationRejected(std::numeric_limits<size_t>::max(), invocationPolicy);
    VernonAdTapeAllocator &invocationAllocator = invocationRejected.descriptor();
    VernonAdRegionHandle invocationRegion = VERNON_AD_INVALID_REGION_HANDLE;
    VernonAdRecordHandle invocationRecord = VERNON_AD_INVALID_RECORD_HANDLE;
    ASSERT_EQ(
        invocationAllocator.begin_region(&invocationAllocator, VERNON_AD_INVALID_REGION_HANDLE, &invocationRegion),
        VERNON_AD_TAPE_ALLOCATOR_OK);
    EXPECT_EQ(invocationAllocator.reserve_record(&invocationAllocator, invocationRegion, 9, 1, 0, &invocationRecord),
              VERNON_AD_TAPE_ALLOCATOR_CAPACITY_EXHAUSTED);

    auto policy = std::make_shared<vernon::runtime::ad::HostTapeMemoryPolicy>(16, 16);
    std::shared_ptr<const vernon::runtime::ad::HostTapeSnapshot> snapshot;
    {
        vernon::runtime::ad::HostDynamicTape first(std::numeric_limits<size_t>::max(), policy);
        VernonAdTapeAllocator &allocator = first.descriptor();
        VernonAdRegionHandle region = VERNON_AD_INVALID_REGION_HANDLE;
        VernonAdRecordHandle record = VERNON_AD_INVALID_RECORD_HANDLE;
        ASSERT_EQ(allocator.begin_region(&allocator, VERNON_AD_INVALID_REGION_HANDLE, &region),
                  VERNON_AD_TAPE_ALLOCATOR_OK);
        ASSERT_EQ(allocator.reserve_record(&allocator, region, 16, 1, 0, &record), VERNON_AD_TAPE_ALLOCATOR_OK);
        ASSERT_EQ(allocator.end_region(&allocator, region, 1, 0), VERNON_AD_TAPE_ALLOCATOR_OK);
        ASSERT_EQ(allocator.seal(&allocator), VERNON_AD_TAPE_ALLOCATOR_OK);
        snapshot = first.takeSnapshot();
        ASSERT_NE(snapshot, nullptr);
    }
    vernon::runtime::ad::HostDynamicTape rejected(std::numeric_limits<size_t>::max(), policy);
    VernonAdTapeAllocator &allocator = rejected.descriptor();
    VernonAdRegionHandle region = VERNON_AD_INVALID_REGION_HANDLE;
    VernonAdRecordHandle record = VERNON_AD_INVALID_RECORD_HANDLE;
    ASSERT_EQ(allocator.begin_region(&allocator, VERNON_AD_INVALID_REGION_HANDLE, &region),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    EXPECT_EQ(allocator.reserve_record(&allocator, region, 1, 1, 0, &record),
              VERNON_AD_TAPE_ALLOCATOR_CAPACITY_EXHAUSTED);
    snapshot.reset();
    ASSERT_EQ(allocator.reset(&allocator), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator.begin_region(&allocator, VERNON_AD_INVALID_REGION_HANDLE, &region),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    EXPECT_EQ(allocator.reserve_record(&allocator, region, 1, 1, 0, &record), VERNON_AD_TAPE_ALLOCATOR_OK);
}

TEST(RuntimeAutodiffTapeAllocator, EnforcesRegionLifetimeResetAndThreadOwnership) {
    vernon::runtime::ad::HostDynamicTape storage;
    VernonAdTapeAllocator &allocator = storage.descriptor();
    VernonAdRegionHandle root = VERNON_AD_INVALID_REGION_HANDLE;
    ASSERT_EQ(allocator.begin_region(&allocator, VERNON_AD_INVALID_REGION_HANDLE, &root), VERNON_AD_TAPE_ALLOCATOR_OK);
    size_t count = 0;
    EXPECT_EQ(allocator.read_executed_count(&allocator, root, &count), VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);

    ASSERT_EQ(allocator.reset(&allocator), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator.begin_region(&allocator, VERNON_AD_INVALID_REGION_HANDLE, &root), VERNON_AD_TAPE_ALLOCATOR_OK);
    VernonAdRecordHandle rootRecord = VERNON_AD_INVALID_RECORD_HANDLE;
    ASSERT_EQ(allocator.reserve_record(&allocator, root, sizeof(uint32_t), alignof(uint32_t), 1, &rootRecord),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    const uint32_t value = 0x12345678u;
    ASSERT_EQ(allocator.write_leaf(&allocator, rootRecord, 0, &value, sizeof(value)), VERNON_AD_TAPE_ALLOCATOR_OK);
    VernonAdRegionHandle nested = VERNON_AD_INVALID_REGION_HANDLE;
    ASSERT_EQ(allocator.begin_region(&allocator, root, &nested), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator.set_child(&allocator, rootRecord, 0, nested), VERNON_AD_TAPE_ALLOCATOR_OK);
    VernonAdRecordHandle nestedRecord = VERNON_AD_INVALID_RECORD_HANDLE;
    ASSERT_EQ(allocator.reserve_record(&allocator, nested, sizeof(uint32_t), alignof(uint32_t), 0, &nestedRecord),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    const uint32_t nestedValue = 0xabcdef01u;
    ASSERT_EQ(allocator.write_leaf(&allocator, nestedRecord, 0, &nestedValue, sizeof(nestedValue)),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator.end_region(&allocator, nested, 1, 2), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator.end_region(&allocator, root, 1, 3), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator.seal(&allocator), VERNON_AD_TAPE_ALLOCATOR_OK);

    std::shared_ptr<const vernon::runtime::ad::HostTapeSnapshot> snapshot = storage.takeSnapshot();
    ASSERT_NE(snapshot, nullptr);
    EXPECT_EQ(storage.takeSnapshot(), nullptr);
    const VernonAdRegionHandle previousRoot = root;
    ASSERT_EQ(allocator.reset(&allocator), VERNON_AD_TAPE_ALLOCATOR_OK);
    EXPECT_EQ(allocator.read_executed_count(&allocator, previousRoot, &count), VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    VernonAdTapeAllocator *reader = snapshot->descriptor();
    uint32_t read = 0;
    ASSERT_EQ(reader->read_leaf(reader, previousRoot, 0, 0, &read, sizeof(read)), VERNON_AD_TAPE_ALLOCATOR_OK);
    EXPECT_EQ(read, value);
    VernonAdRegionHandle readChild = VERNON_AD_INVALID_REGION_HANDLE;
    ASSERT_EQ(reader->read_child(reader, previousRoot, 0, 0, &readChild), VERNON_AD_TAPE_ALLOCATOR_OK);
    EXPECT_EQ(readChild, nested);
    ASSERT_EQ(reader->read_leaf(reader, nested, 0, 0, &read, sizeof(read)), VERNON_AD_TAPE_ALLOCATOR_OK);
    EXPECT_EQ(read, nestedValue);
    uint32_t exitKind = 0;
    ASSERT_EQ(reader->read_executed_count(reader, previousRoot, &count), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(reader->read_exit_kind(reader, previousRoot, &exitKind), VERNON_AD_TAPE_ALLOCATOR_OK);
    EXPECT_EQ(count, 1u);
    EXPECT_EQ(exitKind, 3u);

    ASSERT_EQ(allocator.reset(&allocator), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator.begin_region(&allocator, VERNON_AD_INVALID_REGION_HANDLE, &root), VERNON_AD_TAPE_ALLOCATOR_OK);
    EXPECT_NE(root, previousRoot);
    ASSERT_EQ(allocator.end_region(&allocator, root, 0, 0), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator.seal(&allocator), VERNON_AD_TAPE_ALLOCATOR_OK);
    EXPECT_NE(storage.takeSnapshot(), nullptr);

    ASSERT_EQ(allocator.reset(&allocator), VERNON_AD_TAPE_ALLOCATOR_OK);
    VernonAdTapeAllocatorStatus threadStatus = VERNON_AD_TAPE_ALLOCATOR_OK;
    std::thread other([&] { threadStatus = allocator.reset(&allocator); });
    other.join();
    EXPECT_EQ(threadStatus, VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
}

TEST(RuntimeAutodiffTapeAllocator, RejectsCopiedDescriptorWithoutTrustingUserData) {
    vernon::runtime::ad::HostDynamicTape storage;
    VernonAdTapeAllocator copied = storage.descriptor();
    copied.user_data = reinterpret_cast<void *>(uintptr_t{1});
    EXPECT_EQ(copied.reset(&copied), VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    EXPECT_EQ(copied.status, VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);

    VernonAdTapeAllocator &allocator = storage.descriptor();
    allocator.user_data = reinterpret_cast<void *>(uintptr_t{1});
    EXPECT_EQ(allocator.reset(&allocator), VERNON_AD_TAPE_ALLOCATOR_OK);
}

TEST(RuntimeAutodiff, OrdinaryRuntimeFailureReplacesThreadLocalInvocationDiagnostic) {
    VernonRuntimeContext context;
    context.backend = VERNON_RUNTIME_CPU;
    vernon::runtime::invocationDiagnostic(context) = "stale autodiff diagnostic";
    constexpr char invalidBundle[] = "{";
    EXPECT_EQ(vernonRuntimeLoadPipelineBundleWithOptions(&context, invalidBundle, sizeof(invalidBundle) - 1, nullptr),
              nullptr);
    const VernonStringView error = vernonRuntimeGetLastError(&context);
    const std::string message(error.data, error.size);
    EXPECT_FALSE(message.empty());
    EXPECT_NE(message, "stale autodiff diagnostic");
    vernon::runtime::clearInvocationDiagnostic(context);
}

TEST(RuntimeAutodiff, RejectsExplicitProtocolMismatchBeforeLoadingArtifacts) {
    VernonRuntimeContext context;
    context.backend = VERNON_RUNTIME_CPU;
    vernon::runtime::Stage forward;
    vernon::runtime::Stage backward;
    forward.autodiff = vernon::runtime::AutodiffStageMetadata{"", "dynamic_v2", ""};
    backward.autodiff = vernon::runtime::AutodiffStageMetadata{"", "legacy_fixed", ""};
    std::shared_ptr<vernon::runtime::ad::Executable> executable;
    EXPECT_FALSE(vernon::runtime::ad::createCpuExecutable(context, forward, backward, {"x"}, executable));
    EXPECT_EQ(vernon::runtime::invocationDiagnostic(context),
              "structured CPU autodiff requires explicit dynamic_v2 profiles");

    backward.autodiff->protocol = "unknown";
    forward.autodiff->protocol = "unknown";
    EXPECT_FALSE(vernon::runtime::ad::createCpuExecutable(context, forward, backward, {"x"}, executable));
    EXPECT_EQ(vernon::runtime::invocationDiagnostic(context),
              "structured CPU autodiff requires explicit dynamic_v2 profiles");
    vernon::runtime::clearInvocationDiagnostic(context);
}

TEST(RuntimeAutodiff, RejectsMissingAndLayoutMismatchedDirectProtocols) {
    VernonRuntimeContext context;
    context.backend = VERNON_RUNTIME_CPU;
    const std::string forwardJson =
        profileReflection("forward", 2 * sizeof(uint64_t), sizeof(float),
                          nlohmann::json::array({argument("x", 0, nlohmann::json::array({leaf(0)})),
                                                 tapeAllocatorArgument(sizeof(uint64_t))}),
                          nlohmann::json::array({leaf(0)}))
            .dump();
    const std::string backwardJson =
        profileReflection(
            "backward", 3 * sizeof(uint64_t), sizeof(float),
            nlohmann::json::array({tapeAllocatorArgument(0), tapeRootRegionArgument(sizeof(uint64_t)),
                                   argument("output", 2 * sizeof(uint64_t), nlohmann::json::array({leaf(0)}))}),
            nlohmann::json::array({leaf(0)}))
            .dump();
    const auto view = [](const std::string &value) { return VernonStringView{value.data(), value.size()}; };
    const std::string forwardName = "forward";
    const std::string backwardName = "backward";
    const std::string dynamicProtocol = "dynamic_v2";
    const std::string fixedProtocol = "legacy_fixed";
    std::shared_ptr<vernon::runtime::ad::Executable> executable;

    EXPECT_FALSE(vernon::runtime::ad::createCpuEntryExecutable(
        context, allocatorSquareForward, view(forwardJson), view(forwardName), allocatorSquareBackward,
        view(backwardJson), view(backwardName), {}, view(dynamicProtocol), {"x"}, executable));
    EXPECT_FALSE(vernon::runtime::ad::createCpuEntryExecutable(
        context, allocatorSquareForward, view(forwardJson), view(forwardName), allocatorSquareBackward,
        view(backwardJson), view(backwardName), view(dynamicProtocol), view(fixedProtocol), {"x"}, executable));
    EXPECT_EQ(vernon::runtime::invocationDiagnostic(context),
              "direct structured CPU autodiff requires explicit dynamic_v2 profiles");
    EXPECT_FALSE(vernon::runtime::ad::createCpuEntryExecutable(
        context, allocatorSquareForward, view(forwardJson), view(forwardName), allocatorSquareBackward,
        view(backwardJson), view(backwardName), view(fixedProtocol), view(fixedProtocol), {"x"}, executable));
    EXPECT_EQ(vernon::runtime::invocationDiagnostic(context),
              "direct structured CPU autodiff requires explicit dynamic_v2 profiles");
}

TEST(RuntimeAutodiff, InjectsHiddenTapeAllocatorAndTransfersSealedTape) {
    static constexpr char forwardSymbol[] = "__vernon_cpu_test_allocator_square_forward";
    static constexpr char backwardSymbol[] = "__vernon_cpu_test_allocator_square_backward";
    ASSERT_EQ(vernonRuntimeRegisterStaticCpuEntry({forwardSymbol, sizeof(forwardSymbol) - 1}, allocatorSquareForward),
              VERNON_STATUS_OK);
    ASSERT_EQ(
        vernonRuntimeRegisterStaticCpuEntry({backwardSymbol, sizeof(backwardSymbol) - 1}, allocatorSquareBackward),
        VERNON_STATUS_OK);

    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "vernon_runtime_autodiff_allocator_profile";
    std::filesystem::create_directories(root);
    {
        std::ofstream output(root / "fixture.o", std::ios::binary);
        output.write("registered object fixture", 25);
    }
    const nlohmann::json forwardReflection =
        profileReflection("forward", 2 * sizeof(uint64_t), sizeof(float),
                          nlohmann::json::array({argument("x", 0, nlohmann::json::array({leaf(0)})),
                                                 tapeAllocatorArgument(sizeof(uint64_t))}),
                          nlohmann::json::array({leaf(0)}));
    const nlohmann::json backwardReflection = profileReflection(
        "backward", 3 * sizeof(uint64_t), sizeof(float),
        nlohmann::json::array({tapeAllocatorArgument(0), tapeRootRegionArgument(sizeof(uint64_t)),
                               argument("output", 2 * sizeof(uint64_t), nlohmann::json::array({leaf(0)}))}),
        nlohmann::json::array({leaf(0)}));

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    VernonLoadedPipeline pipeline;
    pipeline.context = context;
    attachCpuAutodiff(*context, pipeline, stage(artifact(root, "forward", forwardSymbol, forwardReflection)),
                      stage(artifact(root, "backward", backwardSymbol, backwardReflection)), {"x"});

    float x = 3.0f;
    float outputValue = 0.0f;
    VernonAdValue input{sizeof(VernonAdValue), {"x", 1}, VERNON_DATA_F32, &x, sizeof(x), 0, nullptr};
    VernonAdValue output{
        sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, &outputValue, sizeof(outputValue), 0, nullptr};
    VernonAdValueSet inputs{sizeof(VernonAdValueSet), &input, 1, {}};
    VernonAdValueSet outputs{sizeof(VernonAdValueSet), &output, 1, {}};
    VernonPullback *pullback = nullptr;
    ASSERT_EQ(vernonAdPipelineForward(&pipeline, {1, 1, 1}, &inputs, &outputs, &pullback), VERNON_STATUS_OK)
        << lastError(context);
    ASSERT_NE(pullback, nullptr);
    EXPECT_FLOAT_EQ(outputValue, 9.0f);

    x = 100.0f;
    float gradientValue = 0.0f;
    VernonAdValue gradient{sizeof(VernonAdValue), {"x", 1}, VERNON_DATA_F32, &gradientValue,
                           sizeof(gradientValue), 0,        nullptr};
    VernonAdValueSet gradients{sizeof(VernonAdValueSet), &gradient, 1, {}};
    ASSERT_EQ(vernonPullbackApply(pullback, nullptr, &gradients), VERNON_STATUS_OK) << lastError(context);
    EXPECT_FLOAT_EQ(gradientValue, 6.0f);

    vernonPullbackDestroy(pullback);

    context->cpuTapePolicy = std::make_shared<vernon::runtime::ad::HostTapeMemoryPolicy>(0, 0);
    VernonLoadedPipeline boundedPipeline;
    boundedPipeline.context = context;
    attachCpuAutodiff(*context, boundedPipeline, stage(artifact(root, "forward", forwardSymbol, forwardReflection)),
                      stage(artifact(root, "backward", backwardSymbol, backwardReflection)), {"x"});
    outputValue = -11.0f;
    pullback = nullptr;
    EXPECT_EQ(vernonAdPipelineForward(&boundedPipeline, {1, 1, 1}, &inputs, &outputs, &pullback),
              VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(pullback, nullptr);
    EXPECT_FLOAT_EQ(outputValue, -11.0f);

    context->cpuTapePolicy = std::make_shared<vernon::runtime::ad::HostTapeMemoryPolicy>(sizeof(float), sizeof(float));
    VernonLoadedPipeline sharedPipeline;
    sharedPipeline.context = context;
    attachCpuAutodiff(*context, sharedPipeline, stage(artifact(root, "forward", forwardSymbol, forwardReflection)),
                      stage(artifact(root, "backward", backwardSymbol, backwardReflection)), {"x"});
    outputValue = 0.0f;
    ASSERT_EQ(vernonAdPipelineForward(&sharedPipeline, {1, 1, 1}, &inputs, &outputs, &pullback), VERNON_STATUS_OK)
        << lastError(context);
    VernonPullback *secondPullback = nullptr;
    outputValue = -13.0f;
    EXPECT_EQ(vernonAdPipelineForward(&sharedPipeline, {1, 1, 1}, &inputs, &outputs, &secondPullback),
              VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(secondPullback, nullptr);
    EXPECT_FLOAT_EQ(outputValue, -13.0f);
    vernonPullbackDestroy(pullback);
    pullback = nullptr;
    ASSERT_EQ(vernonAdPipelineForward(&sharedPipeline, {1, 1, 1}, &inputs, &outputs, &pullback), VERNON_STATUS_OK)
        << lastError(context);
    vernonPullbackDestroy(pullback);

    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
    std::filesystem::remove_all(root);
}

TEST(RuntimeAutodiff, ContainsCppExceptionsAtPublicForwardAndPullbackBoundaries) {
    static constexpr char throwingForwardSymbol[] = "__vernon_cpu_test_throwing_forward";
    static constexpr char successfulForwardSymbol[] = "__vernon_cpu_test_successful_exception_forward";
    static constexpr char successfulBackwardSymbol[] = "__vernon_cpu_test_successful_exception_backward";
    static constexpr char throwingBackwardSymbol[] = "__vernon_cpu_test_throwing_backward";
    ASSERT_EQ(vernonRuntimeRegisterStaticCpuEntry({throwingForwardSymbol, sizeof(throwingForwardSymbol) - 1},
                                                  throwingAutodiffForward),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeRegisterStaticCpuEntry({successfulForwardSymbol, sizeof(successfulForwardSymbol) - 1},
                                                  allocatorSquareForward),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeRegisterStaticCpuEntry({successfulBackwardSymbol, sizeof(successfulBackwardSymbol) - 1},
                                                  allocatorSquareBackward),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeRegisterStaticCpuEntry({throwingBackwardSymbol, sizeof(throwingBackwardSymbol) - 1},
                                                  throwingAutodiffBackward),
              VERNON_STATUS_OK);

    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "vernon_runtime_autodiff_exception_boundary";
    std::filesystem::create_directories(root);
    {
        std::ofstream output(root / "fixture.o", std::ios::binary);
        output.write("registered object fixture", 25);
    }
    const nlohmann::json forwardReflection =
        profileReflection("forward", 2 * sizeof(uint64_t), sizeof(float),
                          nlohmann::json::array({argument("x", 0, nlohmann::json::array({leaf(0)})),
                                                 tapeAllocatorArgument(sizeof(uint64_t))}),
                          nlohmann::json::array({leaf(0)}));
    const nlohmann::json backwardReflection = profileReflection(
        "backward", 3 * sizeof(uint64_t), sizeof(float),
        nlohmann::json::array({tapeAllocatorArgument(0), tapeRootRegionArgument(sizeof(uint64_t)),
                               argument("output", 2 * sizeof(uint64_t), nlohmann::json::array({leaf(0)}))}),
        nlohmann::json::array({leaf(0)}));

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    VernonLoadedPipeline throwingForwardPipeline;
    throwingForwardPipeline.context = context;
    attachCpuAutodiff(*context, throwingForwardPipeline,
                      stage(artifact(root, "forward", throwingForwardSymbol, forwardReflection)),
                      stage(artifact(root, "backward", successfulBackwardSymbol, backwardReflection)), {"x"});

    float x = 2.0f;
    float outputValue = -1.0f;
    VernonAdValue input{sizeof(VernonAdValue), {"x", 1}, VERNON_DATA_F32, &x, sizeof(x), {}};
    VernonAdValue output{sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, &outputValue, sizeof(outputValue), {}};
    VernonAdValueSet inputs{sizeof(VernonAdValueSet), &input, 1, {}};
    VernonAdValueSet outputs{sizeof(VernonAdValueSet), &output, 1, {}};
    VernonPullback *pullback = nullptr;
    EXPECT_EQ(vernonAdPipelineForward(&throwingForwardPipeline, {1, 1, 1}, &inputs, &outputs, &pullback),
              VERNON_STATUS_INTERNAL_ERROR);
    EXPECT_EQ(pullback, nullptr);
    EXPECT_FLOAT_EQ(outputValue, -1.0f);

    VernonLoadedPipeline throwingBackwardPipeline;
    throwingBackwardPipeline.context = context;
    attachCpuAutodiff(*context, throwingBackwardPipeline,
                      stage(artifact(root, "forward", successfulForwardSymbol, forwardReflection)),
                      stage(artifact(root, "backward", throwingBackwardSymbol, backwardReflection)), {"x"});
    ASSERT_EQ(vernonAdPipelineForward(&throwingBackwardPipeline, {1, 1, 1}, &inputs, &outputs, &pullback),
              VERNON_STATUS_OK)
        << lastError(context);
    ASSERT_NE(pullback, nullptr);
    float gradientValue = 7.0f;
    VernonAdValue gradient{sizeof(VernonAdValue), {"x", 1}, VERNON_DATA_F32, &gradientValue, sizeof(gradientValue), {}};
    VernonAdValueSet gradients{sizeof(VernonAdValueSet), &gradient, 1, {}};
    EXPECT_EQ(vernonPullbackApply(pullback, nullptr, &gradients), VERNON_STATUS_INTERNAL_ERROR);
    EXPECT_FLOAT_EQ(gradientValue, 7.0f);

    vernonPullbackDestroy(pullback);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
    std::filesystem::remove_all(root);
}

TEST(RuntimeAutodiff, CommitsStorageAndOutputOnlyAfterSuccessfulCapture) {
    static constexpr char failingSymbol[] = "__vernon_cpu_test_storage_transaction_failing";
    static constexpr char successfulSymbol[] = "__vernon_cpu_test_storage_transaction_successful";
    static constexpr char backwardSymbol[] = "__vernon_cpu_test_storage_transaction_backward";
    ASSERT_EQ(vernonRuntimeRegisterStaticCpuEntry({failingSymbol, sizeof(failingSymbol) - 1}, failingStorageForward),
              VERNON_STATUS_OK);
    ASSERT_EQ(
        vernonRuntimeRegisterStaticCpuEntry({successfulSymbol, sizeof(successfulSymbol) - 1}, successfulStorageForward),
        VERNON_STATUS_OK);
    ASSERT_EQ(
        vernonRuntimeRegisterStaticCpuEntry({backwardSymbol, sizeof(backwardSymbol) - 1}, allocatorSquareBackward),
        VERNON_STATUS_OK);

    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "vernon_runtime_autodiff_effect_transaction";
    std::filesystem::create_directories(root);
    {
        std::ofstream output(root / "fixture.o", std::ios::binary);
        output.write("registered object fixture", 25);
    }
    const nlohmann::json forwardReflection =
        profileReflection("forward", 5 * sizeof(uint64_t), sizeof(float),
                          nlohmann::json::array({tensorViewArgument("storage", 0, "read_write"),
                                                 tapeAllocatorArgument(4 * sizeof(uint64_t))}),
                          nlohmann::json::array({leaf(0)}));
    const nlohmann::json backwardReflection = profileReflection(
        "backward", 3 * sizeof(uint64_t), sizeof(float),
        nlohmann::json::array({tapeAllocatorArgument(0), tapeRootRegionArgument(sizeof(uint64_t)),
                               argument("output", 2 * sizeof(uint64_t), nlohmann::json::array({leaf(0)}))}),
        nlohmann::json::array({leaf(0, 1, "f32", true)}));

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    VernonLoadedPipeline failingPipeline;
    failingPipeline.context = context;
    attachCpuAutodiff(*context, failingPipeline, stage(artifact(root, "forward", failingSymbol, forwardReflection)),
                      stage(artifact(root, "backward", backwardSymbol, backwardReflection)), {"storage"});

    const uint64_t shape[]{1};
    float storageValue = 3.0f;
    float outputValue = -7.0f;
    VernonAdValue input{
        sizeof(VernonAdValue), {"storage", 7}, VERNON_DATA_F32, &storageValue, sizeof(storageValue), 1, shape};
    VernonAdValue output{
        sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, &outputValue, sizeof(outputValue), 0, nullptr};
    VernonAdValueSet inputs{sizeof(VernonAdValueSet), &input, 1, {}};
    VernonAdValueSet outputs{sizeof(VernonAdValueSet), &output, 1, {}};
    VernonPullback *pullback = nullptr;
    std::array<std::byte, sizeof(storageValue)> storageBefore{};
    std::array<std::byte, sizeof(outputValue)> outputBefore{};
    std::memcpy(storageBefore.data(), &storageValue, sizeof(storageValue));
    std::memcpy(outputBefore.data(), &outputValue, sizeof(outputValue));
    EXPECT_EQ(vernonAdPipelineForward(&failingPipeline, {1, 1, 1}, &inputs, &outputs, &pullback),
              VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(pullback, nullptr);
    EXPECT_EQ(std::memcmp(storageBefore.data(), &storageValue, sizeof(storageValue)), 0);
    EXPECT_EQ(std::memcmp(outputBefore.data(), &outputValue, sizeof(outputValue)), 0);

    VernonLoadedPipeline successfulPipeline;
    successfulPipeline.context = context;
    attachCpuAutodiff(*context, successfulPipeline,
                      stage(artifact(root, "forward", successfulSymbol, forwardReflection)),
                      stage(artifact(root, "backward", backwardSymbol, backwardReflection)), {"storage"});
    output.data = &storageValue;
    std::memcpy(storageBefore.data(), &storageValue, sizeof(storageValue));
    EXPECT_EQ(vernonAdPipelineForward(&successfulPipeline, {1, 1, 1}, &inputs, &outputs, &pullback),
              VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(pullback, nullptr);
    EXPECT_EQ(std::memcmp(storageBefore.data(), &storageValue, sizeof(storageValue)), 0);
    output.data = &outputValue;
    ASSERT_EQ(vernonAdPipelineForward(&successfulPipeline, {1, 1, 1}, &inputs, &outputs, &pullback), VERNON_STATUS_OK)
        << lastError(context);
    ASSERT_NE(pullback, nullptr);
    EXPECT_FLOAT_EQ(storageValue, 4.0f);
    EXPECT_FLOAT_EQ(outputValue, 9.0f);

    float gradientValue = 0.0f;
    VernonAdValue gradient{
        sizeof(VernonAdValue), {"storage", 7}, VERNON_DATA_F32, &gradientValue, sizeof(gradientValue), 1, shape};
    VernonAdValueSet gradients{sizeof(VernonAdValueSet), &gradient, 1, {}};
    ASSERT_EQ(vernonPullbackApply(pullback, nullptr, &gradients), VERNON_STATUS_OK) << lastError(context);
    EXPECT_FLOAT_EQ(gradientValue, 6.0f);
    float seedValue = 2.0f;
    VernonAdValue seed{
        sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, &seedValue, sizeof(seedValue), 0, nullptr};
    VernonAdValueSet seeds{sizeof(VernonAdValueSet), &seed, 1, {}};
    ASSERT_EQ(vernonPullbackApply(pullback, &seeds, &gradients), VERNON_STATUS_OK) << lastError(context);
    EXPECT_FLOAT_EQ(gradientValue, 12.0f);
    EXPECT_FLOAT_EQ(storageValue, 4.0f);
    vernonPullbackDestroy(pullback);

    const nlohmann::json readOnlyForwardReflection = profileReflection(
        "forward", 5 * sizeof(uint64_t), sizeof(float),
        nlohmann::json::array({tensorViewArgument("storage", 0, "read"), tapeAllocatorArgument(4 * sizeof(uint64_t))}),
        nlohmann::json::array({leaf(0)}));
    VernonLoadedPipeline readOnlyPipeline;
    readOnlyPipeline.context = context;
    attachCpuAutodiff(*context, readOnlyPipeline,
                      stage(artifact(root, "forward", successfulSymbol, readOnlyForwardReflection)),
                      stage(artifact(root, "backward", backwardSymbol, backwardReflection)), {"storage"});
    storageValue = 3.0f;
    outputValue = -7.0f;
    pullback = nullptr;
    ASSERT_EQ(vernonAdPipelineForward(&readOnlyPipeline, {1, 1, 1}, &inputs, &outputs, &pullback), VERNON_STATUS_OK)
        << lastError(context);
    ASSERT_NE(pullback, nullptr);
    EXPECT_FLOAT_EQ(storageValue, 3.0f);
    EXPECT_FLOAT_EQ(outputValue, 9.0f);
    vernonPullbackDestroy(pullback);

    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
    std::filesystem::remove_all(root);
}

TEST(RuntimeAutodiff, PacksAggregateTensorViewLeavesWithCanonicalValueLayout) {
    static constexpr char forwardSymbol[] = "__vernon_cpu_test_aggregate_storage_forward";
    static constexpr char backwardSymbol[] = "__vernon_cpu_test_aggregate_storage_backward";
    ASSERT_EQ(vernonRuntimeRegisterStaticCpuEntry({forwardSymbol, sizeof(forwardSymbol) - 1}, aggregateStorageForward),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeRegisterStaticCpuEntry({backwardSymbol, sizeof(backwardSymbol) - 1}, squareBackward),
              VERNON_STATUS_OK);

    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "vernon_runtime_autodiff_aggregate_storage";
    std::filesystem::create_directories(root);
    {
        std::ofstream output(root / "fixture.o", std::ios::binary);
        output.write("registered object fixture", 25);
    }
    const nlohmann::json forwardReflection =
        profileReflection("forward", 5 * sizeof(uint64_t), sizeof(float),
                          nlohmann::json::array({aggregateTensorViewArgument("storage", 0, "read_write"),
                                                 tapeAllocatorArgument(4 * sizeof(uint64_t))}),
                          nlohmann::json::array({leaf(0)}));
    const nlohmann::json backwardReflection = profileReflection(
        "backward", 3 * sizeof(uint64_t), 2 * sizeof(float),
        nlohmann::json::array({tapeAllocatorArgument(0), tapeRootRegionArgument(sizeof(uint64_t)),
                               argument("output", 2 * sizeof(uint64_t), nlohmann::json::array({leaf(0)}))}),
        nlohmann::json::array({leaf(0, 2)}));

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    VernonLoadedPipeline pipeline;
    pipeline.context = context;
    attachCpuAutodiff(*context, pipeline, stage(artifact(root, "forward", forwardSymbol, forwardReflection)),
                      stage(artifact(root, "backward", backwardSymbol, backwardReflection)), {"storage.value"});

    const uint64_t shape[]{2};
    std::array<float, 2> values{1.0f, 2.0f};
    std::array<int32_t, 2> counts{3, 4};
    float outputValue = 0.0f;
    std::array<VernonAdValue, 2> inputValues{
        VernonAdValue{
            sizeof(VernonAdValue), {"storage.value", 13}, VERNON_DATA_F32, values.data(), sizeof(values), 1, shape},
        VernonAdValue{
            sizeof(VernonAdValue), {"storage.count", 13}, VERNON_DATA_I32, counts.data(), sizeof(counts), 1, shape}};
    VernonAdValue output{
        sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, &outputValue, sizeof(outputValue), 0, nullptr};
    VernonAdValueSet inputs{sizeof(VernonAdValueSet), inputValues.data(), inputValues.size(), {}};
    VernonAdValueSet outputs{sizeof(VernonAdValueSet), &output, 1, {}};
    VernonPullback *pullback = nullptr;
    ASSERT_EQ(vernonAdPipelineForward(&pipeline, {1, 1, 1}, &inputs, &outputs, &pullback), VERNON_STATUS_OK)
        << lastError(context);
    EXPECT_EQ(values, (std::array<float, 2>{4.0f, 6.0f}));
    EXPECT_EQ(counts, (std::array<int32_t, 2>{13, 14}));
    EXPECT_FLOAT_EQ(outputValue, 10.0f);
    vernonPullbackDestroy(pullback);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
    std::filesystem::remove_all(root);
}

TEST(RuntimeAutodiff, ExecutesReusableScalarPullbackFromRegisteredProfiles) {
    static constexpr char forwardSymbol[] = "__vernon_cpu_test_square_forward";
    static constexpr char backwardSymbol[] = "__vernon_cpu_test_square_backward";
    static constexpr char failingBackwardSymbol[] = "__vernon_cpu_test_square_backward_rejects_negative_seed";
    ASSERT_EQ(vernonRuntimeRegisterStaticCpuEntry({forwardSymbol, sizeof(forwardSymbol) - 1}, squareForward),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeRegisterStaticCpuEntry({backwardSymbol, sizeof(backwardSymbol) - 1}, squareBackward),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeRegisterStaticCpuEntry({failingBackwardSymbol, sizeof(failingBackwardSymbol) - 1},
                                                  allocatorSquareBackwardRejectsNegativeSeed),
              VERNON_STATUS_OK);

    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "vernon_runtime_autodiff_registered_profiles";
    std::filesystem::create_directories(root);
    {
        std::ofstream output(root / "fixture.o", std::ios::binary);
        output.write("registered object fixture", 25);
    }

    const nlohmann::json forwardReflection =
        profileReflection("forward", 2 * sizeof(uint64_t), sizeof(float),
                          nlohmann::json::array({argument("x", 0, nlohmann::json::array({leaf(0)})),
                                                 tapeAllocatorArgument(sizeof(uint64_t))}),
                          nlohmann::json::array({leaf(0)}));
    const nlohmann::json backwardReflection = profileReflection(
        "backward", 3 * sizeof(uint64_t), sizeof(float),
        nlohmann::json::array({tapeAllocatorArgument(0), tapeRootRegionArgument(sizeof(uint64_t)),
                               argument("output", 2 * sizeof(uint64_t), nlohmann::json::array({leaf(0)}))}),
        nlohmann::json::array({leaf(0)}));

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    VernonLoadedPipeline pipeline;
    pipeline.context = context;
    attachCpuAutodiff(*context, pipeline, stage(artifact(root, "forward", forwardSymbol, forwardReflection)),
                      stage(artifact(root, "backward", backwardSymbol, backwardReflection)), {"x"});
    VernonDataType outputType{};
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetAdOutputDataType(&pipeline, &outputType), VERNON_STATUS_OK);
    EXPECT_EQ(outputType, VERNON_DATA_F32);
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetAdGradientCount(&pipeline), 1u);
    VernonStringView gradientPath{};
    VernonDataType gradientType{};
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetAdGradient(&pipeline, 0, &gradientPath, &gradientType), VERNON_STATUS_OK);
    EXPECT_EQ(std::string(gradientPath.data, gradientPath.size), "x");
    EXPECT_EQ(gradientType, VERNON_DATA_F32);

    float x = 3.0f;
    float outputValue = 0.0f;
    VernonAdValue input{sizeof(VernonAdValue), {"x", 1}, VERNON_DATA_F32, &x, sizeof(x), 0, nullptr};
    VernonAdValue output{
        sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, &outputValue, sizeof(outputValue), 0, nullptr};
    VernonAdValueSet inputs{sizeof(VernonAdValueSet), &input, 1, {0, 0, 0, 0}};
    VernonAdValueSet outputs{sizeof(VernonAdValueSet), &output, 1, {0, 0, 0, 0}};
    VernonPullback *pullback = nullptr;
    ASSERT_EQ(vernonAdPipelineForward(&pipeline, {1, 1, 1}, &inputs, &outputs, &pullback), VERNON_STATUS_OK)
        << lastError(context);
    ASSERT_NE(pullback, nullptr);
    EXPECT_FLOAT_EQ(outputValue, 9.0f);

    x = 100.0f;
    float gradientValue = 0.0f;
    VernonAdValue gradient{sizeof(VernonAdValue), {"x", 1}, VERNON_DATA_F32, &gradientValue,
                           sizeof(gradientValue), 0,        nullptr};
    VernonAdValueSet gradients{sizeof(VernonAdValueSet), &gradient, 1, {0, 0, 0, 0}};
    ASSERT_EQ(vernonPullbackApply(pullback, nullptr, &gradients), VERNON_STATUS_OK) << lastError(context);
    EXPECT_FLOAT_EQ(gradientValue, 6.0f);

    float seedValue = 2.0f;
    VernonAdValue seed{
        sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, &seedValue, sizeof(seedValue), 0, nullptr};
    VernonAdValueSet seeds{sizeof(VernonAdValueSet), &seed, 1, {0, 0, 0, 0}};
    ASSERT_EQ(vernonPullbackApply(pullback, &seeds, &gradients), VERNON_STATUS_OK) << lastError(context);
    EXPECT_FLOAT_EQ(gradientValue, 12.0f);

    vernonPullbackDestroy(pullback);

    outputValue = 0.0f;
    gradientValue = 0.0f;
    vernon::runtime::Pullback cppPullback = vernon::runtime::vjp(&pipeline, inputs, outputs, {1, 1, 1});
    vernon::runtime::Pullback movedPullback = std::move(cppPullback);
    EXPECT_FALSE(static_cast<bool>(cppPullback));
    movedPullback.apply(nullptr, gradients);
    EXPECT_FLOAT_EQ(outputValue, 10000.0f);
    EXPECT_FLOAT_EQ(gradientValue, 200.0f);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_INVALID_ARGUMENT);
    movedPullback = {};

    x = 2.0f;
    std::array<float, 24> batchedOutput{};
    const uint64_t batchedShape[]{4, 3, 2};
    output.data = batchedOutput.data();
    output.size = sizeof(batchedOutput);
    output.rank = 3;
    output.shape = batchedShape;
    VernonPullback *batchedPullback = nullptr;
    EXPECT_EQ(vernonAdPipelineForward(&pipeline, {0, 3, 4}, &inputs, &outputs, &batchedPullback),
              VERNON_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(vernonAdPipelineForward(&pipeline, {2, 3, 4}, &inputs, &outputs, &batchedPullback), VERNON_STATUS_OK)
        << lastError(context);
    for (float value : batchedOutput)
        EXPECT_FLOAT_EQ(value, 4.0f);
    std::array<float, 24> batchedSeed{};
    for (size_t index = 0; index < batchedSeed.size(); ++index)
        batchedSeed[index] = static_cast<float>(index + 1);
    seed.data = batchedSeed.data();
    seed.size = sizeof(batchedSeed);
    seed.rank = 3;
    seed.shape = batchedShape;
    gradientValue = 0.0f;
    ASSERT_EQ(vernonPullbackApply(batchedPullback, &seeds, &gradients), VERNON_STATUS_OK) << lastError(context);
    EXPECT_FLOAT_EQ(gradientValue, 1200.0f);
    vernonPullbackDestroy(batchedPullback);

    VernonLoadedPipeline failingBackwardPipeline;
    failingBackwardPipeline.context = context;
    attachCpuAutodiff(*context, failingBackwardPipeline,
                      stage(artifact(root, "forward", forwardSymbol, forwardReflection)),
                      stage(artifact(root, "backward", failingBackwardSymbol, backwardReflection)), {"x"});
    std::array<float, 2> failureOutput{};
    const uint64_t failureShape[]{1, 1, 2};
    output.data = failureOutput.data();
    output.size = sizeof(failureOutput);
    output.rank = 3;
    output.shape = failureShape;
    VernonPullback *failingPullback = nullptr;
    ASSERT_EQ(vernonAdPipelineForward(&failingBackwardPipeline, {2, 1, 1}, &inputs, &outputs, &failingPullback),
              VERNON_STATUS_OK)
        << lastError(context);
    std::array<float, 2> failingSeeds{1.0f, -1.0f};
    seed.data = failingSeeds.data();
    seed.size = sizeof(failingSeeds);
    seed.rank = 3;
    seed.shape = failureShape;
    gradientValue = 17.0f;
    EXPECT_EQ(vernonPullbackApply(failingPullback, &seeds, &gradients), VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_FLOAT_EQ(gradientValue, 17.0f);
    vernonPullbackDestroy(failingPullback);

    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);

    std::filesystem::remove_all(root);
}

TEST(RuntimeAutodiff, ExecutesContiguousTensorValuePullback) {
    static constexpr char forwardSymbol[] = "__vernon_cpu_test_tensor_square_forward";
    static constexpr char backwardSymbol[] = "__vernon_cpu_test_tensor_square_backward";
    ASSERT_EQ(vernonRuntimeRegisterStaticCpuEntry({forwardSymbol, sizeof(forwardSymbol) - 1}, tensorSquareForward),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeRegisterStaticCpuEntry({backwardSymbol, sizeof(backwardSymbol) - 1}, tensorSquareBackward),
              VERNON_STATUS_OK);

    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "vernon_runtime_autodiff_tensor_profiles";
    std::filesystem::create_directories(root);
    {
        std::ofstream output(root / "fixture.o", std::ios::binary);
        output.write("registered object fixture", 25);
    }
    const nlohmann::json forwardReflection =
        profileReflection("forward", 2 * sizeof(uint64_t), 2 * sizeof(float),
                          nlohmann::json::array({argument("values", 0, nlohmann::json::array({leaf(0, 2)}), 8),
                                                 tapeAllocatorArgument(sizeof(uint64_t))}),
                          nlohmann::json::array({leaf(0, 2)}));
    const nlohmann::json backwardReflection = profileReflection(
        "backward", 3 * sizeof(uint64_t), 2 * sizeof(float),
        nlohmann::json::array({tapeAllocatorArgument(0), tapeRootRegionArgument(sizeof(uint64_t)),
                               argument("output", 2 * sizeof(uint64_t), nlohmann::json::array({leaf(0, 2)}), 8)}),
        nlohmann::json::array({leaf(0, 2)}));

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    VernonLoadedPipeline pipeline;
    pipeline.context = context;
    attachCpuAutodiff(*context, pipeline, stage(artifact(root, "forward", forwardSymbol, forwardReflection)),
                      stage(artifact(root, "backward", backwardSymbol, backwardReflection)), {"values"});
    size_t outputRank{};
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetAdOutputRank(&pipeline, &outputRank), VERNON_STATUS_OK);
    EXPECT_EQ(outputRank, 1u);
    uint64_t outputExtent{};
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetAdOutputDimension(&pipeline, 0, &outputExtent), VERNON_STATUS_OK);
    EXPECT_EQ(outputExtent, 2u);

    float values[]{2.0f, 3.0f};
    float outputsBuffer[2]{};
    const uint64_t tensorShape[]{2};
    VernonAdValue input{sizeof(VernonAdValue), {"values", 6}, VERNON_DATA_F32, values, sizeof(values), 1, tensorShape};
    VernonAdValue output{
        sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, outputsBuffer, sizeof(outputsBuffer), 1, tensorShape};
    VernonAdValueSet inputs{sizeof(VernonAdValueSet), &input, 1, {}};
    VernonAdValueSet outputs{sizeof(VernonAdValueSet), &output, 1, {}};
    VernonPullback *pullback = nullptr;
    const uint64_t wrongShape[]{1, 2};
    output.rank = 2;
    output.shape = wrongShape;
    EXPECT_EQ(vernonAdPipelineForward(&pipeline, {1, 1, 1}, &inputs, &outputs, &pullback),
              VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(pullback, nullptr);
    output.rank = 1;
    output.shape = tensorShape;
    ASSERT_EQ(vernonAdPipelineForward(&pipeline, {1, 1, 1}, &inputs, &outputs, &pullback), VERNON_STATUS_OK)
        << lastError(context);
    EXPECT_FLOAT_EQ(outputsBuffer[0], 4.0f);
    EXPECT_FLOAT_EQ(outputsBuffer[1], 9.0f);

    float gradientsBuffer[2]{};
    VernonAdValue gradient{sizeof(VernonAdValue),
                           {"values", 6},
                           VERNON_DATA_F32,
                           gradientsBuffer,
                           sizeof(gradientsBuffer),
                           1,
                           tensorShape};
    VernonAdValueSet gradients{sizeof(VernonAdValueSet), &gradient, 1, {}};
    EXPECT_EQ(vernonPullbackApply(pullback, nullptr, &gradients), VERNON_STATUS_INVALID_ARGUMENT);

    float seedBuffer[]{1.0f, 2.0f};
    VernonAdValue seed{
        sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, seedBuffer, sizeof(seedBuffer), 1, tensorShape};
    VernonAdValueSet seeds{sizeof(VernonAdValueSet), &seed, 1, {}};
    ASSERT_EQ(vernonPullbackApply(pullback, &seeds, &gradients), VERNON_STATUS_OK) << lastError(context);
    EXPECT_FLOAT_EQ(gradientsBuffer[0], 4.0f);
    EXPECT_FLOAT_EQ(gradientsBuffer[1], 12.0f);

    vernonPullbackDestroy(pullback);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
    std::filesystem::remove_all(root);
}

TEST(RuntimeAutodiff, ExecutesFlattenedAggregateInputLeaves) {
    static constexpr char forwardSymbol[] = "__vernon_cpu_test_aggregate_forward";
    static constexpr char backwardSymbol[] = "__vernon_cpu_test_aggregate_backward";
    ASSERT_EQ(vernonRuntimeRegisterStaticCpuEntry({forwardSymbol, sizeof(forwardSymbol) - 1}, aggregateForward),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeRegisterStaticCpuEntry({backwardSymbol, sizeof(backwardSymbol) - 1}, aggregateBackward),
              VERNON_STATUS_OK);

    const std::filesystem::path root = std::filesystem::temp_directory_path() / "vernon_runtime_autodiff_aggregate";
    std::filesystem::create_directories(root);
    {
        std::ofstream output(root / "fixture.o", std::ios::binary);
        output.write("registered object fixture", 25);
    }
    nlohmann::json pairX = leaf(0);
    pairX["path"] = nlohmann::json::array({"x"});
    nlohmann::json pairY = leaf(4);
    pairY["path"] = nlohmann::json::array({"y"});
    const nlohmann::json forwardReflection =
        profileReflection("forward", 3 * sizeof(uint64_t), sizeof(float),
                          nlohmann::json::array({argument("pair", 0, nlohmann::json::array({pairX, pairY}), 8),
                                                 argument("scale", 8, nlohmann::json::array({leaf(0)})),
                                                 tapeAllocatorArgument(2 * sizeof(uint64_t))}),
                          nlohmann::json::array({leaf(0)}));
    const nlohmann::json backwardReflection = profileReflection(
        "backward", 2 * sizeof(uint64_t) + sizeof(float), sizeof(float),
        nlohmann::json::array({tapeAllocatorArgument(0), tapeRootRegionArgument(sizeof(uint64_t)),
                               argument("output", 2 * sizeof(uint64_t), nlohmann::json::array({leaf(0)}))}),
        nlohmann::json::array({leaf(0)}));

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    VernonLoadedPipeline pipeline;
    pipeline.context = context;
    attachCpuAutodiff(*context, pipeline, stage(artifact(root, "forward", forwardSymbol, forwardReflection)),
                      stage(artifact(root, "backward", backwardSymbol, backwardReflection)), {"scale"});

    float pairXValue = 2.0f;
    float pairYValue = 3.0f;
    float scale = 4.0f;
    VernonAdValue inputsBuffer[]{
        {sizeof(VernonAdValue), {"pair.x", 6}, VERNON_DATA_F32, &pairXValue, sizeof(pairXValue), 0, nullptr},
        {sizeof(VernonAdValue), {"pair.y", 6}, VERNON_DATA_F32, &pairYValue, sizeof(pairYValue), 0, nullptr},
        {sizeof(VernonAdValue), {"scale", 5}, VERNON_DATA_F32, &scale, sizeof(scale), 0, nullptr},
    };
    VernonAdValueSet inputs{sizeof(VernonAdValueSet), inputsBuffer, std::size(inputsBuffer), {}};
    float outputValue = 0.0f;
    VernonAdValue output{
        sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, &outputValue, sizeof(outputValue), 0, nullptr};
    VernonAdValueSet outputs{sizeof(VernonAdValueSet), &output, 1, {}};
    VernonPullback *pullback = nullptr;
    ASSERT_EQ(vernonAdPipelineForward(&pipeline, {1, 1, 1}, &inputs, &outputs, &pullback), VERNON_STATUS_OK)
        << lastError(context);
    EXPECT_FLOAT_EQ(outputValue, 24.0f);

    float gradientValue = 0.0f;
    VernonAdValue gradient{
        sizeof(VernonAdValue), {"scale", 5}, VERNON_DATA_F32, &gradientValue, sizeof(gradientValue), 0, nullptr};
    VernonAdValueSet gradients{sizeof(VernonAdValueSet), &gradient, 1, {}};
    ASSERT_EQ(vernonPullbackApply(pullback, nullptr, &gradients), VERNON_STATUS_OK) << lastError(context);
    EXPECT_FLOAT_EQ(gradientValue, 6.0f);

    vernonPullbackDestroy(pullback);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
    std::filesystem::remove_all(root);
}

TEST(RuntimeAutodiff, ExposesMetadataFromResolvedSignature) {
    VernonRuntimeContext context;
    context.backend = VERNON_RUNTIME_VULKAN;
    VernonLoadedPipeline pipeline;
    pipeline.context = &context;
    vernon::runtime::ad::Signature signature;
    signature.outputs.push_back({"output", VERNON_DATA_F32, 2 * sizeof(float), alignof(float), {2}});
    signature.cotangents = signature.outputs;
    signature.gradients.push_back({"values", VERNON_DATA_F32, 2 * sizeof(float), alignof(float), {2}});
    pipeline.autodiff = VernonLoadedAutodiff{std::make_shared<MetadataExecutable>(std::move(signature))};

    VernonDataType outputType{};
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetAdOutputDataType(&pipeline, &outputType), VERNON_STATUS_OK);
    EXPECT_EQ(outputType, VERNON_DATA_F32);
    size_t rank{};
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetAdOutputRank(&pipeline, &rank), VERNON_STATUS_OK);
    EXPECT_EQ(rank, 1u);
    uint64_t extent{};
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetAdOutputDimension(&pipeline, 0, &extent), VERNON_STATUS_OK);
    EXPECT_EQ(extent, 2u);
    VernonStringView path{};
    VernonDataType gradientType{};
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetAdGradient(&pipeline, 0, &path, &gradientType), VERNON_STATUS_OK);
    EXPECT_EQ(std::string(path.data, path.size), "values");
    EXPECT_EQ(gradientType, VERNON_DATA_F32);
}

TEST(RuntimeAutodiff, OneElementTensorStillRequiresExplicitCotangent) {
    static constexpr char forwardSymbol[] = "__vernon_cpu_test_one_tensor_forward";
    static constexpr char backwardSymbol[] = "__vernon_cpu_test_one_tensor_backward";
    ASSERT_EQ(vernonRuntimeRegisterStaticCpuEntry({forwardSymbol, sizeof(forwardSymbol) - 1}, squareForward),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeRegisterStaticCpuEntry({backwardSymbol, sizeof(backwardSymbol) - 1}, squareBackward),
              VERNON_STATUS_OK);

    const std::filesystem::path root = std::filesystem::temp_directory_path() / "vernon_runtime_autodiff_one_tensor";
    std::filesystem::create_directories(root);
    {
        std::ofstream output(root / "fixture.o", std::ios::binary);
        output.write("registered object fixture", 25);
    }
    const nlohmann::json forwardReflection =
        profileReflection("forward", 2 * sizeof(uint64_t), sizeof(float),
                          nlohmann::json::array({argument("x", 0, nlohmann::json::array({leaf(0, 1, "f32", true)})),
                                                 tapeAllocatorArgument(sizeof(uint64_t))}),
                          nlohmann::json::array({leaf(0, 1, "f32", true)}));
    const nlohmann::json backwardReflection =
        profileReflection("backward", 3 * sizeof(uint64_t), sizeof(float),
                          nlohmann::json::array({tapeAllocatorArgument(0), tapeRootRegionArgument(sizeof(uint64_t)),
                                                 argument("output", 2 * sizeof(uint64_t),
                                                          nlohmann::json::array({leaf(0, 1, "f32", true)}))}),
                          nlohmann::json::array({leaf(0, 1, "f32", true)}));

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    VernonLoadedPipeline pipeline;
    pipeline.context = context;
    attachCpuAutodiff(*context, pipeline, stage(artifact(root, "forward", forwardSymbol, forwardReflection)),
                      stage(artifact(root, "backward", backwardSymbol, backwardReflection)), {"x"});
    float x = 3.0f;
    float outputValue = 0.0f;
    const uint64_t tensorShape[]{1};
    VernonAdValue input{sizeof(VernonAdValue), {"x", 1}, VERNON_DATA_F32, &x, sizeof(x), 1, tensorShape};
    VernonAdValue output{
        sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, &outputValue, sizeof(outputValue), 1, tensorShape};
    VernonAdValueSet inputs{sizeof(VernonAdValueSet), &input, 1, {}};
    VernonAdValueSet outputs{sizeof(VernonAdValueSet), &output, 1, {}};
    VernonPullback *pullback = nullptr;
    ASSERT_EQ(vernonAdPipelineForward(&pipeline, {1, 1, 1}, &inputs, &outputs, &pullback), VERNON_STATUS_OK)
        << lastError(context);

    float gradientValue = 0.0f;
    VernonAdValue gradient{sizeof(VernonAdValue), {"x", 1}, VERNON_DATA_F32, &gradientValue,
                           sizeof(gradientValue), 1,        tensorShape};
    VernonAdValueSet gradients{sizeof(VernonAdValueSet), &gradient, 1, {}};
    EXPECT_EQ(vernonPullbackApply(pullback, nullptr, &gradients), VERNON_STATUS_INVALID_ARGUMENT);

    vernonPullbackDestroy(pullback);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
    std::filesystem::remove_all(root);
}

TEST(RuntimeAutodiff, PromotesF16ImplicitCotangentAndGradientToF32) {
    static constexpr char forwardSymbol[] = "__vernon_cpu_test_half_identity_forward";
    static constexpr char backwardSymbol[] = "__vernon_cpu_test_half_identity_backward";
    ASSERT_EQ(vernonRuntimeRegisterStaticCpuEntry({forwardSymbol, sizeof(forwardSymbol) - 1}, halfIdentityForward),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeRegisterStaticCpuEntry({backwardSymbol, sizeof(backwardSymbol) - 1}, halfIdentityBackward),
              VERNON_STATUS_OK);

    const std::filesystem::path root = std::filesystem::temp_directory_path() / "vernon_runtime_autodiff_f16";
    std::filesystem::create_directories(root);
    {
        std::ofstream output(root / "fixture.o", std::ios::binary);
        output.write("registered object fixture", 25);
    }
    const nlohmann::json forwardReflection =
        profileReflection("forward", 2 * sizeof(uint64_t), 2,
                          nlohmann::json::array({argument("x", 0, nlohmann::json::array({leaf(0, 1, "f16")}), 2, 2),
                                                 tapeAllocatorArgument(sizeof(uint64_t))}),
                          nlohmann::json::array({leaf(0, 1, "f16")}), 2);
    const nlohmann::json backwardReflection = profileReflection(
        "backward", 2 * sizeof(uint64_t) + sizeof(float), sizeof(float),
        nlohmann::json::array({tapeAllocatorArgument(0), tapeRootRegionArgument(sizeof(uint64_t)),
                               argument("output", 2 * sizeof(uint64_t), nlohmann::json::array({leaf(0, 1, "f32")}))}),
        nlohmann::json::array({leaf(0, 1, "f32")}));

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    VernonLoadedPipeline pipeline;
    pipeline.context = context;
    attachCpuAutodiff(*context, pipeline, stage(artifact(root, "forward", forwardSymbol, forwardReflection)),
                      stage(artifact(root, "backward", backwardSymbol, backwardReflection)), {"x"});
    uint16_t x = 0x3c00;
    uint16_t outputValue = 0;
    VernonAdValue input{sizeof(VernonAdValue), {"x", 1}, VERNON_DATA_F16, &x, sizeof(x), {}};
    VernonAdValue output{sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F16, &outputValue, sizeof(outputValue), {}};
    VernonAdValueSet inputs{sizeof(VernonAdValueSet), &input, 1, {}};
    VernonAdValueSet outputs{sizeof(VernonAdValueSet), &output, 1, {}};
    VernonPullback *pullback = nullptr;
    ASSERT_EQ(vernonAdPipelineForward(&pipeline, {1, 1, 1}, &inputs, &outputs, &pullback), VERNON_STATUS_OK)
        << lastError(context);

    float gradientValue = 0.0f;
    VernonAdValue gradient{sizeof(VernonAdValue), {"x", 1}, VERNON_DATA_F32, &gradientValue, sizeof(gradientValue), {}};
    VernonAdValueSet gradients{sizeof(VernonAdValueSet), &gradient, 1, {}};
    EXPECT_EQ(vernonPullbackApply(pullback, nullptr, &gradients), VERNON_STATUS_OK) << lastError(context);
    EXPECT_FLOAT_EQ(gradientValue, 1.0f);

    vernonPullbackDestroy(pullback);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
    std::filesystem::remove_all(root);
}

} // namespace
