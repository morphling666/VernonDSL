#include "runtime/autodiff/runtime_gpu_derivatives.h"
#include "runtime_gpu_failure_injection.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

namespace vernon::runtime::ad::gpu {
namespace {

bool checkedAdd(size_t left, size_t right, size_t &result) {
    if (right > std::numeric_limits<size_t>::max() - left)
        return false;
    result = left + right;
    return true;
}

bool samePath(const VernonStringView &path, const std::string &expected) {
    return path.data && path.size == expected.size() && std::memcmp(path.data, expected.data(), path.size) == 0;
}

bool indexValues(const std::vector<ValueAbi> &expected, const VernonAdValueSet &set,
                 std::vector<const VernonAdValue *> &slots) {
    slots.assign(expected.size(), nullptr);
    for (size_t valueIndex = 0; valueIndex < set.value_count; ++valueIndex) {
        const VernonAdValue &value = set.values[valueIndex];
        if (value.struct_size < sizeof(VernonAdValue))
            return false;
        const auto abi = std::find_if(expected.begin(), expected.end(),
                                      [&](const ValueAbi &candidate) { return samePath(value.path, candidate.path); });
        if (abi == expected.end())
            return false;
        const size_t index = static_cast<size_t>(std::distance(expected.begin(), abi));
        if (slots[index])
            return false;
        slots[index] = &value;
    }
    return std::none_of(slots.begin(), slots.end(), [](const VernonAdValue *value) { return value == nullptr; });
}

bool indexValues(const std::vector<ValueAbi> &expected, VernonAdValueSet &set, std::vector<VernonAdValue *> &slots) {
    slots.assign(expected.size(), nullptr);
    for (size_t valueIndex = 0; valueIndex < set.value_count; ++valueIndex) {
        VernonAdValue &value = set.values[valueIndex];
        if (value.struct_size < sizeof(VernonAdValue))
            return false;
        const auto abi = std::find_if(expected.begin(), expected.end(),
                                      [&](const ValueAbi &candidate) { return samePath(value.path, candidate.path); });
        if (abi == expected.end())
            return false;
        const size_t index = static_cast<size_t>(std::distance(expected.begin(), abi));
        if (slots[index])
            return false;
        slots[index] = &value;
    }
    return std::none_of(slots.begin(), slots.end(), [](const VernonAdValue *value) { return value == nullptr; });
}

bool indexValues(const std::vector<ValueAbi> &expected, const VernonAdDeviceValueSet &set,
                 std::vector<const VernonAdDeviceValue *> &slots) {
    slots.assign(expected.size(), nullptr);
    for (size_t valueIndex = 0; valueIndex < set.value_count; ++valueIndex) {
        const VernonAdDeviceValue &value = set.values[valueIndex];
        const auto abi = std::find_if(expected.begin(), expected.end(),
                                      [&](const ValueAbi &candidate) { return samePath(value.path, candidate.path); });
        if (abi == expected.end())
            return false;
        const size_t index = static_cast<size_t>(std::distance(expected.begin(), abi));
        if (slots[index])
            return false;
        slots[index] = &value;
    }
    return std::none_of(slots.begin(), slots.end(), [](const VernonAdDeviceValue *value) { return value == nullptr; });
}

bool indexValues(const std::vector<ValueAbi> &expected, VernonAdDeviceValueSet &set,
                 std::vector<VernonAdDeviceValue *> &slots) {
    slots.assign(expected.size(), nullptr);
    for (size_t valueIndex = 0; valueIndex < set.value_count; ++valueIndex) {
        VernonAdDeviceValue &value = set.values[valueIndex];
        const auto abi = std::find_if(expected.begin(), expected.end(),
                                      [&](const ValueAbi &candidate) { return samePath(value.path, candidate.path); });
        if (abi == expected.end())
            return false;
        const size_t index = static_cast<size_t>(std::distance(expected.begin(), abi));
        if (slots[index])
            return false;
        slots[index] = &value;
    }
    return std::none_of(slots.begin(), slots.end(), [](const VernonAdDeviceValue *value) { return value == nullptr; });
}

bool valueMatches(const VernonAdDeviceValue &value, const ValueAbi &abi) {
    return value.dtype == abi.dtype && value.size == abi.byteSize && value.rank == abi.logicalShape.size() &&
           (!value.rank || std::equal(value.shape, value.shape + value.rank, abi.logicalShape.begin()));
}

} // namespace

bool PreparedDerivativeValues::prepare(VernonRuntimeContext &context, const Signature &signature,
                                       const VernonAdValueSet *sourceCotangents, VernonAdValueSet &gradients,
                                       const BindingSpecPlan &bindingSpecs, VernonLaunchSize invocationExtent,
                                       size_t baseTemporaryBytes, size_t temporaryLimit, std::string &error) {
    temporaryBytes = baseTemporaryBytes;
    failureStatus = VERNON_STATUS_INVALID_ARGUMENT;
    cotangents = sourceCotangents;
    cotangentSources.clear();
    destinations.clear();
    deviceCotangentSources.clear();
    deviceDestinations.clear();
    deviceGradientOwners.clear();
    stagedGradients.clear();
    gradientSlots.clear();
    devices.clear();
    const bool implicit = !cotangents && signature.cotangents.size() == 1;
    if ((!cotangents && !implicit) || (cotangents && cotangents->value_count != signature.cotangents.size()) ||
        gradients.value_count != signature.gradients.size()) {
        error = "GPU pullback values do not match the reflected derivative signature";
        return false;
    }

    if (implicit) {
        if (!makeCotangentBytes(nullptr, signature.cotangents.front(), implicitCotangentBytes, error))
            return false;
        const ValueAbi &abi = signature.cotangents.front();
        implicitCotangent = {sizeof(VernonAdValue),
                             {abi.path.data(), abi.path.size()},
                             abi.dtype,
                             implicitCotangentBytes.data(),
                             implicitCotangentBytes.size(),
                             static_cast<uint32_t>(abi.logicalShape.size()),
                             abi.logicalShape.data()};
        implicitCotangentSet = {sizeof(VernonAdValueSet), &implicitCotangent, 1, {}};
        cotangents = &implicitCotangentSet;
    }
    if (!indexValues(signature.cotangents, *cotangents, cotangentSources) ||
        !indexValues(signature.gradients, gradients, destinations)) {
        error = "GPU pullback values cannot be indexed by the reflected derivative signature";
        return false;
    }
    if (bindingSpecs.cotangentBindingBySignature.size() != signature.cotangents.size() ||
        bindingSpecs.gradientSlotBySignature.size() != signature.gradients.size()) {
        error = "GPU pullback binding plan does not match the reflected derivative signature";
        return false;
    }

    for (size_t index = 0; index < signature.cotangents.size(); ++index) {
        const ValueAbi &abi = signature.cotangents[index];
        const VernonAdValue *source = cotangentSources[index];
        const size_t bindingIndex = bindingSpecs.cotangentBindingBySignature[index];
        if (bindingIndex >= bindingSpecs.bindings.size()) {
            error = "GPU pullback cotangent has no indexed physical binding specification";
            return false;
        }
        const BindingSpec &bindingSpec = bindingSpecs.bindings[bindingIndex];
        if (bindingSpec.sourceSlot >= bindingSpecs.derivativeSlotCount) {
            error = "GPU pullback cotangent binding slot is invalid";
            return false;
        }
        const bool carrierDimension = bindingSpec.carrierDimension;
        ValueAbi carried = abi;
        const bool carrierValid = !carrierDimension || materializeCarrierValue(carried, invocationExtent);
        size_t physicalCarrierCount = 0;
        const bool carrierCountValid =
            !carrierDimension || (carrierCount(invocationExtent, physicalCarrierCount) &&
                                  physicalCarrierCount <= std::numeric_limits<uint32_t>::max());
        const bool shared = implicit || (source && valueMatches(*source, abi));
        const size_t sourceSize = implicit ? abi.byteSize : source ? source->size : 0;
        if ((!source && !implicit) || !carrierValid || !carrierCountValid ||
            (!shared && (!carrierDimension || !valueMatches(*source, carried))) ||
            !checkedAdd(temporaryBytes, sourceSize, temporaryBytes) ||
            (implicit && !checkedAdd(temporaryBytes, sourceSize, temporaryBytes))) {
            error = "GPU pullback cotangent does not match reflection";
            return false;
        }
    }
    for (size_t index = 0; index < signature.gradients.size(); ++index) {
        const ValueAbi &abi = signature.gradients[index];
        VernonAdValue *destination = destinations[index];
        if (bindingSpecs.gradientSlotBySignature[index] >= bindingSpecs.derivativeSlotCount || !destination ||
            !valueMatches(*destination, abi) || !checkedAdd(temporaryBytes, destination->size, temporaryBytes) ||
            !checkedAdd(temporaryBytes, destination->size, temporaryBytes)) {
            error = "GPU pullback gradient does not match reflection";
            return false;
        }
    }
    if (temporaryBytes > temporaryLimit) {
        error = "GPU pullback temporary memory exceeds the apply-time budget";
        return false;
    }

    devices.resize(bindingSpecs.derivativeSlotCount);
    for (size_t index = 0; index < signature.cotangents.size(); ++index) {
        const ValueAbi &abi = signature.cotangents[index];
        const VernonAdValue *source = cotangentSources[index];
        const BindingSpec &bindingSpec = bindingSpecs.bindings[bindingSpecs.cotangentBindingBySignature[index]];
        const bool carrierDimension = bindingSpec.carrierDimension;
        const bool shared = valueMatches(*source, abi);
        DeviceValue value(context, *source);
        if (carrierDimension) {
            size_t count = 0;
            if (!carrierCount(invocationExtent, count)) {
                error = "GPU pullback cotangent carrier size overflows";
                return false;
            }
            if (!bindingSpec.tensorViewRank ||
                abi.logicalShape.size() != bindingSpec.tensorViewRank - 1 + bindingSpec.leafShape.size() ||
                value.strides.size() < bindingSpec.tensorViewRank - 1) {
                error = "GPU pullback cotangent carrier layout is incomplete";
                return false;
            }
            const size_t logicalTensorRank = bindingSpec.tensorViewRank - 1;
            std::vector<uint64_t> shape;
            shape.reserve(bindingSpec.tensorViewRank);
            shape.push_back(count);
            shape.insert(shape.end(), abi.logicalShape.begin(), abi.logicalShape.begin() + logicalTensorRank);
            std::vector<int64_t> strides;
            strides.reserve(shape.size());
            strides.push_back(shared ? 0 : static_cast<int64_t>(abi.byteSize));
            strides.insert(strides.end(), value.strides.begin(), value.strides.begin() + logicalTensorRank);
            value.shape = std::move(shape);
            value.strides = std::move(strides);
        }
        if (!value.buffer.upload(source->data, source->size)) {
            error = "cannot upload GPU pullback cotangent";
            failureStatus = VERNON_STATUS_INTERNAL_ERROR;
            return false;
        }
        devices[bindingSpec.sourceSlot].emplace(std::move(value));
    }
    stagedGradients.resize(signature.gradients.size());
    gradientSlots = bindingSpecs.gradientSlotBySignature;
    for (size_t index = 0; index < signature.gradients.size(); ++index) {
        VernonAdValue *destination = destinations[index];
        stagedGradients[index].resize(destination->size);
        DeviceValue value(context, *destination);
        if (!value.buffer.upload(stagedGradients[index].data(), stagedGradients[index].size())) {
            error = "cannot allocate transactional GPU gradient";
            failureStatus = VERNON_STATUS_INTERNAL_ERROR;
            return false;
        }
        devices[gradientSlots[index]].emplace(std::move(value));
    }
    return true;
}

bool PreparedDerivativeValues::prepareDevice(VernonRuntimeContext &context, const Signature &signature,
                                             const VernonAdDeviceValueSet *sourceCotangents,
                                             VernonAdDeviceValueSet &gradients, const BindingSpecPlan &bindingSpecs,
                                             VernonLaunchSize invocationExtent, size_t baseTemporaryBytes,
                                             size_t temporaryLimit, std::string &error) {
    temporaryBytes = baseTemporaryBytes;
    failureStatus = VERNON_STATUS_INVALID_ARGUMENT;
    cotangents = nullptr;
    cotangentSources.clear();
    destinations.clear();
    deviceCotangentSources.clear();
    deviceDestinations.clear();
    deviceGradientOwners.clear();
    stagedGradients.clear();
    gradientSlots.clear();
    devices.clear();
    const bool implicit = !sourceCotangents && signature.cotangents.size() == 1;
    if ((!sourceCotangents && !implicit) ||
        (sourceCotangents && sourceCotangents->value_count != signature.cotangents.size()) ||
        gradients.value_count != signature.gradients.size() ||
        (sourceCotangents && !indexValues(signature.cotangents, *sourceCotangents, deviceCotangentSources)) ||
        !indexValues(signature.gradients, gradients, deviceDestinations)) {
        error = "GPU device pullback values do not match the reflected derivative signature";
        return false;
    }
    if (bindingSpecs.cotangentBindingBySignature.size() != signature.cotangents.size() ||
        bindingSpecs.gradientSlotBySignature.size() != signature.gradients.size()) {
        error = "GPU device pullback binding plan does not match the reflected derivative signature";
        return false;
    }
    if (implicit) {
        if (!makeCotangentBytes(nullptr, signature.cotangents.front(), implicitCotangentBytes, error))
            return false;
        const ValueAbi &abi = signature.cotangents.front();
        implicitCotangent = {sizeof(VernonAdValue),
                             {abi.path.data(), abi.path.size()},
                             abi.dtype,
                             implicitCotangentBytes.data(),
                             implicitCotangentBytes.size(),
                             static_cast<uint32_t>(abi.logicalShape.size()),
                             abi.logicalShape.data()};
    }
    for (size_t index = 0; index < signature.cotangents.size(); ++index) {
        const ValueAbi &abi = signature.cotangents[index];
        const VernonAdDeviceValue *source = implicit ? nullptr : deviceCotangentSources[index];
        const size_t bindingIndex = bindingSpecs.cotangentBindingBySignature[index];
        if (bindingIndex >= bindingSpecs.bindings.size()) {
            error = "GPU device pullback cotangent has no indexed physical binding specification";
            return false;
        }
        const BindingSpec &bindingSpec = bindingSpecs.bindings[bindingIndex];
        if (bindingSpec.sourceSlot >= bindingSpecs.derivativeSlotCount) {
            error = "GPU device pullback cotangent binding slot is invalid";
            return false;
        }
        ValueAbi carried = abi;
        const bool carrierValid = !bindingSpec.carrierDimension || materializeCarrierValue(carried, invocationExtent);
        size_t physicalCarrierCount = 0;
        const bool carrierCountValid =
            !bindingSpec.carrierDimension || (carrierCount(invocationExtent, physicalCarrierCount) &&
                                              physicalCarrierCount <= std::numeric_limits<uint32_t>::max());
        const bool shared = implicit || (source && valueMatches(*source, abi));
        if ((!source && !implicit) || !carrierValid || !carrierCountValid ||
            (!shared && (!bindingSpec.carrierDimension || !valueMatches(*source, carried))) ||
            (implicit && (!checkedAdd(temporaryBytes, abi.byteSize, temporaryBytes) ||
                          !checkedAdd(temporaryBytes, abi.byteSize, temporaryBytes)))) {
            error = "GPU device pullback cotangent does not match reflection";
            return false;
        }
    }
    for (size_t index = 0; index < signature.gradients.size(); ++index) {
        const VernonAdDeviceValue *destination = deviceDestinations[index];
        if (bindingSpecs.gradientSlotBySignature[index] >= bindingSpecs.derivativeSlotCount || !destination ||
            !valueMatches(*destination, signature.gradients[index]) ||
            !vernonRhiDeviceIsBufferValid(context.rhiDevice, destination->buffer)) {
            error = "GPU device pullback gradient does not match reflection";
            return false;
        }
        const auto owner = std::find_if(deviceGradientOwners.begin(), deviceGradientOwners.end(),
                                        [&](const DeviceGradientOwner &candidate) {
                                            return candidate.destination.index == destination->buffer.index &&
                                                   candidate.destination.generation == destination->buffer.generation;
                                        });
        if (owner != deviceGradientOwners.end()) {
            if (owner->size != destination->buffer_size) {
                error = "GPU device gradient leaves disagree about their physical owner";
                return false;
            }
        } else {
            const size_t ownerBytes = static_cast<size_t>(destination->buffer_size);
            if (!checkedAdd(temporaryBytes, ownerBytes, temporaryBytes) ||
                !checkedAdd(temporaryBytes, ownerBytes, temporaryBytes)) {
                error = "GPU device gradient owner size overflows";
                return false;
            }
            deviceGradientOwners.push_back({destination->buffer, ownerBytes, {}});
        }
    }
    if (temporaryBytes > temporaryLimit) {
        error = "GPU device pullback temporary memory exceeds the apply-time budget";
        return false;
    }
    devices.resize(bindingSpecs.derivativeSlotCount);
    for (size_t index = 0; index < signature.cotangents.size(); ++index) {
        const ValueAbi &abi = signature.cotangents[index];
        const BindingSpec &bindingSpec = bindingSpecs.bindings[bindingSpecs.cotangentBindingBySignature[index]];
        const bool shared = implicit || valueMatches(*deviceCotangentSources[index], abi);
        DeviceValue value =
            implicit ? DeviceValue(context, implicitCotangent) : DeviceValue(context, *deviceCotangentSources[index]);
        if (!value.buffer.valid()) {
            error = "GPU device pullback cotangent buffer is invalid";
            return false;
        }
        if (implicit && !value.buffer.upload(implicitCotangent.data, implicitCotangent.size)) {
            error = "cannot upload implicit GPU device pullback cotangent";
            failureStatus = VERNON_STATUS_INTERNAL_ERROR;
            return false;
        }
        if (bindingSpec.carrierDimension) {
            size_t count = 0;
            if (!carrierCount(invocationExtent, count) || !bindingSpec.tensorViewRank ||
                abi.logicalShape.size() != bindingSpec.tensorViewRank - 1 + bindingSpec.leafShape.size() ||
                value.strides.size() < bindingSpec.tensorViewRank - 1) {
                error = "GPU device pullback cotangent carrier layout is incomplete";
                return false;
            }
            const size_t logicalTensorRank = bindingSpec.tensorViewRank - 1;
            std::vector<uint64_t> shape{count};
            shape.insert(shape.end(), abi.logicalShape.begin(), abi.logicalShape.begin() + logicalTensorRank);
            std::vector<int64_t> strides{shared ? 0 : static_cast<int64_t>(abi.byteSize)};
            strides.insert(strides.end(), value.strides.begin(), value.strides.begin() + logicalTensorRank);
            value.shape = std::move(shape);
            value.strides = std::move(strides);
        }
        devices[bindingSpec.sourceSlot].emplace(std::move(value));
    }
    gradientSlots = bindingSpecs.gradientSlotBySignature;
    for (DeviceGradientOwner &owner : deviceGradientOwners) {
        owner.shadow = std::make_shared<DeviceBuffer>(context, owner.size);
        std::vector<uint8_t> zeros(owner.size);
        if (!owner.shadow->upload(zeros.data(), zeros.size())) {
            error = "cannot allocate transactional GPU device gradient owner";
            failureStatus = VERNON_STATUS_INTERNAL_ERROR;
            return false;
        }
    }
    for (size_t index = 0; index < signature.gradients.size(); ++index) {
        const VernonAdDeviceValue &destination = *deviceDestinations[index];
        const auto owner = std::find_if(deviceGradientOwners.begin(), deviceGradientOwners.end(),
                                        [&](const DeviceGradientOwner &candidate) {
                                            return candidate.destination.index == destination.buffer.index &&
                                                   candidate.destination.generation == destination.buffer.generation;
                                        });
        if (owner == deviceGradientOwners.end() || !owner->shadow) {
            error = "GPU device gradient owner has no transactional shadow";
            return false;
        }
        devices[gradientSlots[index]].emplace(context, owner->shadow, destination);
    }
    return true;
}

bool PreparedDerivativeValues::stageGradients(const Signature &signature, std::string &error) {
    for (size_t index = 0; index < signature.gradients.size(); ++index) {
        const size_t slot = index < gradientSlots.size() ? gradientSlots[index] : devices.size();
        if (slot >= devices.size() || !devices[slot] ||
            !devices[slot]->buffer.download(stagedGradients[index].data(), stagedGradients[index].size())) {
            error = "cannot download GPU pullback gradient";
            failureStatus = VERNON_STATUS_INTERNAL_ERROR;
            return false;
        }
    }
    return true;
}

bool PreparedDerivativeValues::publishGradients() const {
    if (injectFailure(FailureBoundary::Publication))
        return false;
    for (size_t index = 0; index < destinations.size(); ++index)
        std::memcpy(destinations[index]->data, stagedGradients[index].data(), stagedGradients[index].size());
    return true;
}

bool PreparedDerivativeValues::devicePublicationCopies(std::vector<DeviceBufferCopy> &copies,
                                                       std::string &error) const {
    copies.clear();
    copies.reserve(deviceGradientOwners.size());
    for (const DeviceGradientOwner &owner : deviceGradientOwners) {
        if (!owner.shadow || !owner.shadow->valid() || !owner.size) {
            error = "GPU device gradient publication plan is incomplete";
            return false;
        }
        copies.push_back({owner.shadow->handle(), owner.destination, 0, 0, owner.size});
    }
    return true;
}

} // namespace vernon::runtime::ad::gpu
