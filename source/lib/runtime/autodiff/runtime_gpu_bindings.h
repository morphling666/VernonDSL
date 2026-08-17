#ifndef VERNON_RUNTIME_AUTODIFF_RUNTIME_GPU_BINDINGS_H
#define VERNON_RUNTIME_AUTODIFF_RUNTIME_GPU_BINDINGS_H

#include "runtime/autodiff/runtime_gpu_resources.h"

#include <optional>
#include <utility>

namespace vernon::runtime::ad {
struct Signature;
}

namespace vernon::runtime::ad::gpu {

enum class BindingSource {
    ReplaySegment,
    ReplayStatus,
    Tape,
    Launch,
    Device,
    Host,
};

struct Binding {
    const Parameter *parameter{};
    BindingSource source{};
    DeviceValue *device{};
    HostValue *host{};
    std::vector<uint64_t> materializedShape;
    std::vector<int64_t> materializedStrides;
    bool materializedLayout{};

    const std::vector<uint64_t> &shape() const {
        return materializedLayout ? materializedShape : device ? device->shape : host->shape;
    }
    const std::vector<int64_t> &strides() const {
        return materializedLayout ? materializedStrides : device ? device->strides : host->strides;
    }
};

struct BindingSpec {
    size_t parameterIndex{};
    BindingSource source{};
    enum class ValueClass {
        None,
        Derivative,
        RetainedPrimal,
        Primal,
    } valueClass{};
    AutodiffResourceRole derivativeRole{AutodiffResourceRole::None};
    size_t sourceSlot{};
    size_t signatureIndex{};
    std::string sourceName;
    std::string sourcePath;
    size_t tensorViewRank{};
    std::vector<uint64_t> tensorShape;
    std::vector<uint64_t> leafShape;
    size_t recordByteStride{};
    bool carrierDimension{};
};

struct BindingSpecPlan {
    std::vector<BindingSpec> bindings;
    size_t derivativeSlotCount{};
    std::vector<size_t> cotangentBindingBySignature;
    std::vector<size_t> gradientSlotBySignature;
    std::vector<std::string> retainedNames;
    std::vector<std::string> primalNames;

    void clear() {
        bindings.clear();
        derivativeSlotCount = 0;
        cotangentBindingBySignature.clear();
        gradientSlotBySignature.clear();
        retainedNames.clear();
        primalNames.clear();
    }
    void reserve(size_t size) { bindings.reserve(size); }
    size_t size() const { return bindings.size(); }
    auto begin() const { return bindings.begin(); }
    auto end() const { return bindings.end(); }
    void push_back(BindingSpec spec) { bindings.push_back(std::move(spec)); }
};
using BindingPlan = std::vector<Binding>;
using DerivativeSlots = std::vector<std::optional<DeviceValue>>;

bool buildBindingSpecPlan(const Variant &variant, const Signature &signature, BindingSpecPlan &plan,
                          std::string &error);
bool materializeBindingPlan(const Variant &variant, const BindingSpecPlan &specs, DeviceValues &retainedDevices,
                            HostValues &retainedHosts, DeviceValues &working, DerivativeSlots &derivatives,
                            BindingPlan &plan, std::string &error);

} // namespace vernon::runtime::ad::gpu

#endif
