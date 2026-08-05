#include "runtime_autodiff_internal.h"

#include "host_tape_allocator.h"
#include "runtime/backend_cpu.h"
#include "runtime/pipeline_metadata.h"
#include "runtime/runtime_state.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vernon::runtime::ad {

namespace {

struct HostLeaf {
    ValueAbi value;
    size_t offset{};
};

struct HostArgument {
    std::string name;
    std::string builtin;
    size_t offset{};
    size_t size{};
    bool tensorView{};
    std::string access;
    std::vector<HostLeaf> leaves;
};

struct HostProfileLayout {
    size_t argumentsSize{};
    size_t resultsSize{};
    bool hasTapeAllocator{};
    std::vector<HostArgument> arguments;
    std::vector<HostLeaf> results;
};

VernonStatus fail(VernonRuntimeContext &context, std::string message,
                  VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT) {
    invocationDiagnostic(context) = std::move(message);
    return status;
}

const char *allocatorFailure(VernonAdTapeAllocatorStatus status) {
    switch (status) {
    case VERNON_AD_TAPE_ALLOCATOR_CAPACITY_EXHAUSTED:
        return "capacity exhausted";
    case VERNON_AD_TAPE_ALLOCATOR_ARITHMETIC_OVERFLOW:
        return "arithmetic overflow";
    case VERNON_AD_TAPE_ALLOCATOR_HOST_ALLOCATION_FAILURE:
        return "host allocation failed";
    case VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE:
        return "invalid state";
    case VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI:
        return "invalid ABI";
    case VERNON_AD_TAPE_ALLOCATOR_OK:
        return "no failure";
    }
    return "unknown failure";
}

bool appendLeafPath(const nlohmann::json &value, std::string &path, std::string &error) {
    if (!value.contains("path") || !value["path"].is_array()) {
        error = "autodiff Value leaf path is invalid";
        return false;
    }
    for (const nlohmann::json &component : value["path"]) {
        std::string name;
        if (component.is_string())
            name = component.get<std::string>();
        else if (component.is_number_unsigned())
            name = std::to_string(component.get<uint64_t>());
        else {
            error = "autodiff Value leaf path is invalid";
            return false;
        }
        if (name.empty() || name.find('.') != std::string::npos) {
            error = "autodiff Value leaf path is invalid";
            return false;
        }
        path += "." + name;
    }
    return true;
}

bool parseLeaf(const nlohmann::json &value, const std::string &path, size_t baseOffset, HostLeaf &leaf,
               std::string &error) {
    if (!value.is_object() || !value.contains("dtype") || !value["dtype"].is_string() ||
        !value.contains("byte_offset") || !value["byte_offset"].is_number_unsigned() ||
        !value.contains("scalar_count") || !value["scalar_count"].is_number_unsigned()) {
        error = "autodiff profile contains an invalid Value leaf";
        return false;
    }
    const auto dtype = pipelineDataType(value["dtype"].get<std::string>());
    const size_t scalarSize = dtype ? dtypeSize(*dtype) : 0;
    const size_t scalarCount = value["scalar_count"].get<size_t>();
    if (!scalarSize || !scalarCount || scalarCount > SIZE_MAX / scalarSize) {
        error = "autodiff Runtime profile contains an unsupported or empty Value leaf";
        return false;
    }
    const size_t relativeOffset = value["byte_offset"].get<size_t>();
    if (relativeOffset > SIZE_MAX - baseOffset) {
        error = "autodiff scalar leaf offset overflows";
        return false;
    }
    std::vector<uint64_t> shape;
    if (value.contains("shape")) {
        if (!value["shape"].is_array()) {
            error = "autodiff Value leaf shape is invalid";
            return false;
        }
        size_t shapedCount = 1;
        for (const nlohmann::json &extentValue : value["shape"]) {
            if (!extentValue.is_number_unsigned()) {
                error = "autodiff Value leaf shape is invalid";
                return false;
            }
            const uint64_t extent = extentValue.get<uint64_t>();
            if (!extent || extent > SIZE_MAX / shapedCount) {
                error = "autodiff Value leaf shape is empty or overflows";
                return false;
            }
            shapedCount *= static_cast<size_t>(extent);
            shape.push_back(extent);
        }
        if (shapedCount != scalarCount) {
            error = "autodiff Value leaf shape does not match scalar_count";
            return false;
        }
    } else if (scalarCount != 1) {
        error = "autodiff non-scalar Value leaf has no shape";
        return false;
    }
    std::string leafPath = path;
    if (!appendLeafPath(value, leafPath, error))
        return false;
    leaf = {{std::move(leafPath), *dtype, scalarCount * scalarSize, scalarSize, std::move(shape)},
            baseOffset + relativeOffset};
    return true;
}

bool parseProfile(const Stage &stage, HostProfileLayout &layout, std::string &error) {
    const nlohmann::json root = nlohmann::json::parse(stage.reflection, nullptr, false);
    if (root.is_discarded()) {
        error = "autodiff artifact reflection is invalid JSON";
        return false;
    }
    const nlohmann::json *entry = nullptr;
    if (root.contains("entries") && root["entries"].is_array())
        for (const nlohmann::json &candidate : root["entries"])
            if (candidate.is_object() && candidate.value("name", "") == stage.entry) {
                entry = &candidate;
                break;
            }
    if (!entry || !entry->contains("arguments") || !(*entry)["arguments"].is_array() || !entry->contains("results") ||
        !(*entry)["results"].is_array() || !entry->contains("physical_layouts") ||
        !(*entry)["physical_layouts"].is_object()) {
        error = "autodiff artifact reflection has no selected profile ABI";
        return false;
    }
    const auto host = (*entry)["physical_layouts"].find("host_value");
    if (host == (*entry)["physical_layouts"].end() || !host->is_object() || !host->contains("packed_arguments_size") ||
        !(*host)["packed_arguments_size"].is_number_unsigned() || !host->contains("packed_results_size") ||
        !(*host)["packed_results_size"].is_number_unsigned()) {
        error = "autodiff profile has no host Value ABI";
        return false;
    }
    layout.argumentsSize = (*host)["packed_arguments_size"].get<size_t>();
    layout.resultsSize = (*host)["packed_results_size"].get<size_t>();
    if (!layout.argumentsSize || !layout.resultsSize) {
        error = "autodiff profile has an empty host Value ABI";
        return false;
    }
    for (const nlohmann::json &value : (*entry)["arguments"]) {
        if (!value.is_object() || !value.contains("physical_layouts") || !value["physical_layouts"].is_object()) {
            error = "autodiff profile contains an invalid argument ABI";
            return false;
        }
        const auto physical = value["physical_layouts"].find("host_value");
        const std::string physicalKind =
            physical != value["physical_layouts"].end() && physical->is_object() ? physical->value("kind", "") : "";
        const nlohmann::json *physicalExtent =
            physicalKind == "cpu_call" && physical->contains("root") && (*physical)["root"].is_object()
                ? &(*physical)["root"]
                : (physicalKind == "resource_binding" ? &*physical : nullptr);
        if (physical == value["physical_layouts"].end() || !physical->is_object() ||
            !physical->contains("frame_offset") || !(*physical)["frame_offset"].is_number_unsigned() ||
            !physicalExtent || !physicalExtent->contains("size") || !(*physicalExtent)["size"].is_number_unsigned()) {
            error = "autodiff argument has no host Value layout";
            return false;
        }
        HostArgument argument;
        argument.name = value.value("vernon.source_name", "");
        argument.builtin = value.value("builtin", "");
        argument.offset = (*physical)["frame_offset"].get<size_t>();
        argument.size = (*physicalExtent)["size"].get<size_t>();
        argument.tensorView = physical->value("kind", "") == "resource_binding" &&
                              physical->value("resource_kind", "") == "tensor_view_descriptor";
        argument.access = value.value("access", "");
        if (argument.offset > layout.argumentsSize || argument.size > layout.argumentsSize - argument.offset) {
            error = "autodiff argument host Value layout is out of bounds";
            return false;
        }
        if (argument.builtin == VERNON_AD_TAPE_ALLOCATOR_BUILTIN) {
            if (value.value("kind", "") != "builtin" || argument.size != sizeof(VernonAdTapeAllocator *) ||
                layout.hasTapeAllocator) {
                error = "autodiff tape allocator builtin has an invalid host ABI";
                return false;
            }
            layout.hasTapeAllocator = true;
            layout.arguments.push_back(std::move(argument));
            continue;
        }
        if (argument.name.empty()) {
            error = "autodiff argument has no source name";
            return false;
        }
        if (argument.tensorView) {
            if (!value.contains("source_shape") || !value["source_shape"].is_array() ||
                !value.contains("element_layout") || !value["element_layout"].is_object() ||
                !value["element_layout"].contains("leaves") || !value["element_layout"]["leaves"].is_array() ||
                value["element_layout"]["leaves"].size() != 1) {
                error = "autodiff TensorView argument has an invalid element ABI";
                return false;
            }
            const nlohmann::json &element = value["element_layout"]["leaves"][0];
            if (!element.is_object() || !element.contains("dtype") || !element["dtype"].is_string() ||
                element.value("scalar_count", 0u) != 1) {
                error = "autodiff TensorView argument requires one scalar element leaf";
                return false;
            }
            const auto dtype = pipelineDataType(element["dtype"].get<std::string>());
            const size_t scalarSize = dtype ? dtypeSize(*dtype) : 0;
            size_t scalarCount = 1;
            std::vector<uint64_t> shape;
            for (const nlohmann::json &extentValue : value["source_shape"]) {
                if (!extentValue.is_number_unsigned()) {
                    error = "autodiff TensorView argument has an invalid shape";
                    return false;
                }
                const uint64_t extent = extentValue.get<uint64_t>();
                if (!extent || extent > SIZE_MAX / scalarCount) {
                    error = "autodiff TensorView argument shape is empty or overflows";
                    return false;
                }
                scalarCount *= static_cast<size_t>(extent);
                shape.push_back(extent);
            }
            if (!scalarSize || shape.empty() || scalarCount > SIZE_MAX / scalarSize ||
                argument.size != sizeof(uint64_t) * (2 + 2 * shape.size()) ||
                (argument.access != "read" && argument.access != "write" && argument.access != "read_write")) {
                error = "autodiff TensorView argument has an invalid host descriptor";
                return false;
            }
            argument.leaves.push_back(
                {{argument.name, *dtype, scalarCount * scalarSize, scalarSize, std::move(shape)}, argument.offset});
            layout.arguments.push_back(std::move(argument));
            continue;
        }
        if (!value.contains("value_layout") || !value["value_layout"].is_object() ||
            !value["value_layout"].contains("leaves") || !value["value_layout"]["leaves"].is_array()) {
            error = "autodiff profile contains an invalid argument Value ABI";
            return false;
        }
        for (const nlohmann::json &leafValue : value["value_layout"]["leaves"]) {
            HostLeaf leaf;
            if (!parseLeaf(leafValue, argument.name, argument.offset, leaf, error))
                return false;
            if (leaf.offset < argument.offset || leaf.offset > argument.offset + argument.size ||
                leaf.value.byteSize > argument.offset + argument.size - leaf.offset) {
                error = "autodiff argument Value leaf is out of bounds";
                return false;
            }
            argument.leaves.push_back(std::move(leaf));
        }
        if (argument.leaves.empty()) {
            error = "autodiff argument contains no Value leaves";
            return false;
        }
        layout.arguments.push_back(std::move(argument));
    }
    if ((*entry)["results"].size() != 1) {
        error = "CPU autodiff profile requires one result";
        return false;
    }
    const nlohmann::json &result = (*entry)["results"][0];
    if (!result.is_object() || !result.contains("physical_layouts") || !result["physical_layouts"].is_object() ||
        !result.contains("value_layout") || !result["value_layout"].is_object() ||
        !result["value_layout"].contains("leaves") || !result["value_layout"]["leaves"].is_array()) {
        error = "autodiff profile contains an invalid result ABI";
        return false;
    }
    const auto physical = result["physical_layouts"].find("host_value");
    if (physical == result["physical_layouts"].end() || !physical->is_object() || !physical->contains("frame_offset") ||
        !(*physical)["frame_offset"].is_number_unsigned() || physical->value("kind", "") != "cpu_call" ||
        !physical->contains("root") || !(*physical)["root"].is_object() || !(*physical)["root"].contains("size") ||
        !(*physical)["root"]["size"].is_number_unsigned()) {
        error = "autodiff result has no host Value layout";
        return false;
    }
    const size_t resultOffset = (*physical)["frame_offset"].get<size_t>();
    const size_t resultSize = (*physical)["root"]["size"].get<size_t>();
    if (resultOffset > layout.resultsSize || resultSize > layout.resultsSize - resultOffset) {
        error = "autodiff result host Value layout is out of bounds";
        return false;
    }
    size_t index = 0;
    for (const nlohmann::json &leafValue : result["value_layout"]["leaves"]) {
        HostLeaf leaf;
        if (!parseLeaf(leafValue, "result." + std::to_string(index++), resultOffset, leaf, error))
            return false;
        if (leaf.offset < resultOffset || leaf.offset > resultOffset + resultSize ||
            leaf.value.byteSize > resultOffset + resultSize - leaf.offset) {
            error = "autodiff result Value leaf is out of bounds";
            return false;
        }
        layout.results.push_back(std::move(leaf));
    }
    if (layout.results.empty()) {
        error = "autodiff profile result contains no scalar leaves";
        return false;
    }
    return true;
}

class CpuPullbackExecution final : public PullbackExecution {
public:
    CpuPullbackExecution(VernonRuntimeContext &context, std::shared_ptr<CpuKernelState> backward,
                         HostProfileLayout layout, Signature signature, VernonLaunchSize computeGrid,
                         std::vector<std::vector<uint8_t>> arguments,
                         std::vector<std::shared_ptr<const HostTapeSnapshot>> tapeSnapshots)
        : context_(context), backward_(std::move(backward)), layout_(std::move(layout)),
          signature_(std::move(signature)), computeGrid_(computeGrid), arguments_(std::move(arguments)),
          tapeSnapshots_(std::move(tapeSnapshots)) {}

    VernonStatus apply(const VernonAdValueSet *cotangents, VernonAdValueSet &gradients) override {
        if (gradients.value_count != signature_.gradients.size())
            return fail(context_, "invalid pullback invocation");
        ValueAbi cotangentAbi = signature_.cotangent;
        if (!materializeCarrierValue(cotangentAbi, computeGrid_))
            return fail(context_, "CPU pullback cotangent size overflows");
        std::vector<uint8_t> cotangent;
        if (!makeCotangentBytes(cotangents, cotangentAbi, cotangent, invocationDiagnostic(context_)))
            return VERNON_STATUS_INVALID_ARGUMENT;
        const HostLeaf &cotangentLeaf = layout_.arguments[1].leaves.front();
        std::vector<VernonAdValue *> destinations;
        destinations.reserve(signature_.gradients.size());
        for (size_t index = 0; index < signature_.gradients.size(); ++index) {
            VernonAdValue *gradient = findValue(gradients, signature_.gradients[index].path);
            if (!gradient || !valueMatches(*gradient, signature_.gradients[index]))
                return fail(context_, "gradient output does not match backward reflection");
            std::memset(gradient->data, 0, gradient->size);
            destinations.push_back(gradient);
        }
        for (size_t invocationIndex = 0; invocationIndex < arguments_.size(); ++invocationIndex) {
            std::vector<uint8_t> &arguments = arguments_[invocationIndex];
            std::memcpy(arguments.data() + cotangentLeaf.offset,
                        cotangent.data() + invocationIndex * signature_.cotangent.byteSize,
                        signature_.cotangent.byteSize);
            std::vector<uint8_t> results(layout_.resultsSize);
            const VernonCpuInvocation invocation{arguments.data(), arguments.size(), results.data(), results.size(),
                                                 nullptr};
            const VernonStatus status = backward_->entry(&invocation);
            if (status != VERNON_STATUS_OK)
                return fail(context_, "autodiff backward profile invocation failed", status);
            for (size_t index = 0; index < signature_.gradients.size(); ++index) {
                const HostLeaf &source = layout_.results[index];
                uint8_t *destination = static_cast<uint8_t *>(destinations[index]->data);
                const uint8_t *contribution = results.data() + source.offset;
                const size_t scalarSize = dtypeSize(source.value.dtype);
                for (size_t offset = 0; offset < source.value.byteSize; offset += scalarSize) {
                    if (source.value.dtype == VERNON_DATA_F32) {
                        float current;
                        float value;
                        std::memcpy(&current, destination + offset, sizeof(current));
                        std::memcpy(&value, contribution + offset, sizeof(value));
                        current += value;
                        std::memcpy(destination + offset, &current, sizeof(current));
                    } else if (source.value.dtype == VERNON_DATA_F64) {
                        double current;
                        double value;
                        std::memcpy(&current, destination + offset, sizeof(current));
                        std::memcpy(&value, contribution + offset, sizeof(value));
                        current += value;
                        std::memcpy(destination + offset, &current, sizeof(current));
                    } else {
                        return fail(context_, "CPU pullback can only reduce floating gradients");
                    }
                }
            }
        }
        return VERNON_STATUS_OK;
    }

private:
    VernonRuntimeContext &context_;
    std::shared_ptr<CpuKernelState> backward_;
    HostProfileLayout layout_;
    Signature signature_;
    VernonLaunchSize computeGrid_{};
    std::vector<std::vector<uint8_t>> arguments_;
    std::vector<std::shared_ptr<const HostTapeSnapshot>> tapeSnapshots_;
};

class CpuExecutable final : public HostExecutable {
public:
    CpuExecutable(VernonRuntimeContext &context, HostProfileLayout forwardLayout, HostProfileLayout backwardLayout,
                  std::shared_ptr<CpuKernelState> forward, std::shared_ptr<CpuKernelState> backward,
                  Signature signature)
        : context_(context), forwardLayout_(std::move(forwardLayout)), backwardLayout_(std::move(backwardLayout)),
          forward_(std::move(forward)), backward_(std::move(backward)), signature_(std::move(signature)) {}

    const Signature &signature() const override { return signature_; }

    VernonStatus forward(VernonLaunchSize computeGrid, const VernonAdValueSet &inputs, VernonAdValueSet &outputs,
                         std::unique_ptr<PullbackExecution> &pullback) override {
        size_t invocationCount = 0;
        if (!carrierCount(computeGrid, invocationCount))
            return fail(context_, "native CPU autodiff launch size overflows");
        if (inputs.value_count != signature_.inputs.size() || outputs.value_count != 1)
            return fail(context_, "autodiff forward values do not match the host profile");
        std::vector<uint8_t> arguments(forwardLayout_.argumentsSize);
        struct StorageRange {
            uintptr_t begin;
            uintptr_t end;
            bool writable;
        };
        std::vector<StorageRange> storageRanges;
        for (const HostArgument &argument : forwardLayout_.arguments) {
            if (!argument.builtin.empty())
                continue;
            if (argument.tensorView) {
                const HostLeaf &leaf = argument.leaves.front();
                const VernonAdValue *value = findValue(inputs, leaf.value.path);
                if (!value || !valueMatches(*value, leaf.value))
                    return fail(context_, "autodiff forward TensorView input '" + leaf.value.path +
                                              "' does not match profile reflection");
                const uintptr_t begin = reinterpret_cast<uintptr_t>(value->data);
                if (value->size > std::numeric_limits<uintptr_t>::max() - begin)
                    return fail(context_, "autodiff forward TensorView input address range overflows");
                const StorageRange range{begin, begin + value->size, argument.access != "read"};
                for (const StorageRange &existing : storageRanges)
                    if ((range.writable || existing.writable) && range.begin < existing.end &&
                        existing.begin < range.end)
                        return fail(context_, "writable native CPU autodiff Storage inputs overlap");
                storageRanges.push_back(range);
                uint8_t *descriptor = arguments.data() + argument.offset;
                const uintptr_t pointer = reinterpret_cast<uintptr_t>(value->data);
                const uint64_t zero = 0;
                std::memcpy(descriptor, &pointer, sizeof(pointer));
                std::memcpy(descriptor + sizeof(uint64_t), &zero, sizeof(zero));
                const size_t rank = leaf.value.logicalShape.size();
                for (size_t dimension = 0; dimension < rank; ++dimension) {
                    const uint64_t extent = leaf.value.logicalShape[dimension];
                    std::memcpy(descriptor + sizeof(uint64_t) * (2 + dimension), &extent, sizeof(extent));
                }
                uint64_t stride = 1;
                for (size_t dimension = rank; dimension-- > 0;) {
                    std::memcpy(descriptor + sizeof(uint64_t) * (2 + rank + dimension), &stride, sizeof(stride));
                    stride *= leaf.value.logicalShape[dimension];
                }
                continue;
            }
            for (const HostLeaf &leaf : argument.leaves) {
                const VernonAdValue *value = findValue(inputs, leaf.value.path);
                if (!value || !valueMatches(*value, leaf.value))
                    return fail(
                        context_,
                        "autodiff forward input '" + leaf.value.path +
                            "' does not match profile reflection (expected " + std::to_string(leaf.value.byteSize) +
                            " bytes at rank " + std::to_string(leaf.value.logicalShape.size()) + ", received " +
                            (value ? std::to_string(value->size) + " bytes at rank " + std::to_string(value->rank)
                                   : std::string("no value")) +
                            ")");
                std::memcpy(arguments.data() + leaf.offset, value->data, value->size);
            }
        }
        VernonAdValue *output = findValue(outputs, signature_.output.path);
        ValueAbi outputAbi = signature_.output;
        if (!materializeCarrierValue(outputAbi, computeGrid))
            return fail(context_, "native CPU autodiff output size overflows");
        if (!output || !valueMatches(*output, outputAbi))
            return fail(context_, "autodiff output does not match profile reflection");

        std::vector<std::vector<uint8_t>> backwardArguments;
        backwardArguments.reserve(invocationCount);
        std::vector<std::shared_ptr<const HostTapeSnapshot>> tapeSnapshots;
        if (forwardLayout_.hasTapeAllocator)
            tapeSnapshots.reserve(invocationCount);
        for (size_t invocationIndex = 0; invocationIndex < invocationCount; ++invocationIndex) {
            std::unique_ptr<HostTapeAllocator> tapeAllocator;
            for (const HostArgument &argument : forwardLayout_.arguments) {
                if (argument.builtin.empty())
                    continue;
                if (argument.builtin == VERNON_AD_TAPE_ALLOCATOR_BUILTIN) {
                    tapeAllocator = std::make_unique<HostTapeAllocator>();
                    VernonAdTapeAllocator *descriptor = &tapeAllocator->descriptor();
                    std::memcpy(arguments.data() + argument.offset, &descriptor, sizeof(descriptor));
                    continue;
                }
                if (argument.builtin != "global_invocation_id" || argument.leaves.size() != 1 ||
                    argument.leaves.front().value.dtype != VERNON_DATA_U32 ||
                    argument.leaves.front().value.logicalShape != std::vector<uint64_t>{3})
                    return fail(context_, "native CPU autodiff has an unsupported builtin ABI");
                const uint32_t coordinates[] = {
                    static_cast<uint32_t>(invocationIndex % computeGrid.x),
                    static_cast<uint32_t>((invocationIndex / computeGrid.x) % computeGrid.y),
                    static_cast<uint32_t>(invocationIndex / (static_cast<size_t>(computeGrid.x) * computeGrid.y)),
                };
                std::memcpy(arguments.data() + argument.leaves.front().offset, coordinates, sizeof(coordinates));
            }
            std::vector<uint8_t> results(forwardLayout_.resultsSize);
            const VernonCpuInvocation invocation{arguments.data(), arguments.size(), results.data(), results.size(),
                                                 nullptr};
            const VernonStatus status = forward_->entry(&invocation);
            if (status != VERNON_STATUS_OK)
                return fail(context_, "autodiff forward profile invocation failed", status);
            if (tapeAllocator) {
                const VernonAdTapeAllocatorStatus allocatorStatus = tapeAllocator->descriptor().status;
                if (allocatorStatus != VERNON_AD_TAPE_ALLOCATOR_OK)
                    return fail(context_, std::string("autodiff tape allocator ") + allocatorFailure(allocatorStatus),
                                allocatorStatus == VERNON_AD_TAPE_ALLOCATOR_HOST_ALLOCATION_FAILURE
                                    ? VERNON_STATUS_INTERNAL_ERROR
                                    : VERNON_STATUS_INVALID_ARGUMENT);
                std::shared_ptr<const HostTapeSnapshot> snapshot = tapeAllocator->takeSnapshot();
                if (!snapshot)
                    return fail(context_, "autodiff forward profile did not seal its tape");
                tapeSnapshots.push_back(std::move(snapshot));
            }
            std::memcpy(static_cast<uint8_t *>(output->data) + invocationIndex * signature_.output.byteSize,
                        results.data() + forwardLayout_.results.front().offset, signature_.output.byteSize);
            backwardArguments.emplace_back(backwardLayout_.argumentsSize);
            for (size_t index = 0; index < signature_.tape.size(); ++index) {
                const HostLeaf &source = forwardLayout_.results[index + 1];
                const HostLeaf &destination = backwardLayout_.arguments[0].leaves[index];
                std::memcpy(backwardArguments.back().data() + destination.offset, results.data() + source.offset,
                            source.value.byteSize);
            }
        }
        pullback = std::make_unique<CpuPullbackExecution>(context_, backward_, backwardLayout_, signature_, computeGrid,
                                                          std::move(backwardArguments), std::move(tapeSnapshots));
        return VERNON_STATUS_OK;
    }

private:
    VernonRuntimeContext &context_;
    HostProfileLayout forwardLayout_;
    HostProfileLayout backwardLayout_;
    std::shared_ptr<CpuKernelState> forward_;
    std::shared_ptr<CpuKernelState> backward_;
    Signature signature_;
};

} // namespace

bool createCpuExecutable(VernonRuntimeContext &context, const Stage &forwardStage, const Stage &backwardStage,
                         const std::vector<std::string> &gradientPaths, std::shared_ptr<Executable> &executable) {
    if (!forwardStage.cpuArtifact || !backwardStage.cpuArtifact) {
        invocationDiagnostic(context) = "CPU autodiff profiles have no loadable artifacts";
        return false;
    }
    HostProfileLayout forward;
    HostProfileLayout backward;
    auto forwardKernel = std::make_shared<CpuKernelState>();
    auto backwardKernel = std::make_shared<CpuKernelState>();
    ReflectedEntry reflection;
    if (!parseProfile(forwardStage, forward, invocationDiagnostic(context)) ||
        !parseProfile(backwardStage, backward, invocationDiagnostic(context)) ||
        !loadCpuNativeArtifact(*forwardStage.cpuArtifact, *forwardKernel, reflection, invocationDiagnostic(context)) ||
        !loadCpuNativeArtifact(*backwardStage.cpuArtifact, *backwardKernel, reflection, invocationDiagnostic(context)))
        return false;
    if (forward.results.size() < 1 || backward.arguments.size() != 2 ||
        backward.arguments[0].leaves.size() + 1 != forward.results.size() || backward.arguments[1].leaves.size() != 1 ||
        backward.results.size() != gradientPaths.size()) {
        invocationDiagnostic(context) = "autodiff profile ABI does not match the contiguous Value Runtime contract";
        return false;
    }
    Signature signature;
    for (const HostArgument &argument : forward.arguments) {
        if (!argument.builtin.empty())
            continue;
        for (const HostLeaf &leaf : argument.leaves)
            signature.inputs.push_back(leaf.value);
    }
    signature.output = forward.results.front().value;
    signature.output.path = "output";
    for (size_t index = 1; index < forward.results.size(); ++index) {
        ValueAbi tape = forward.results[index].value;
        tape.path = "tape." + std::to_string(index - 1);
        signature.tape.push_back(std::move(tape));
    }
    signature.cotangent = backward.arguments[1].leaves.front().value;
    signature.cotangent.path = "output";
    if (!derivativeAbiMatches(signature.output, signature.cotangent)) {
        invocationDiagnostic(context) = "autodiff output and cotangent profile ABIs do not match";
        return false;
    }
    for (size_t index = 0; index < signature.tape.size(); ++index) {
        const ValueAbi &backwardTape = backward.arguments[0].leaves[index].value;
        if (signature.tape[index].dtype != backwardTape.dtype ||
            signature.tape[index].byteSize != backwardTape.byteSize ||
            signature.tape[index].logicalShape != backwardTape.logicalShape) {
            invocationDiagnostic(context) = "autodiff tape leaves do not match backward reflection";
            return false;
        }
    }
    for (size_t index = 0; index < gradientPaths.size(); ++index) {
        ValueAbi gradient = backward.results[index].value;
        gradient.path = gradientPaths[index];
        signature.gradients.push_back(std::move(gradient));
    }
    executable =
        std::make_shared<CpuExecutable>(context, std::move(forward), std::move(backward), std::move(forwardKernel),
                                        std::move(backwardKernel), std::move(signature));
    return true;
}

} // namespace vernon::runtime::ad
