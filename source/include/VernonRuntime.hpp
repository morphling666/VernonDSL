#ifndef VERNON_RUNTIME_HPP
#define VERNON_RUNTIME_HPP

#include "VernonRuntime.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace vernon::runtime {

struct Float16 {
    uint16_t bits{};
};

class Pullback {
public:
    Pullback() = default;
    explicit Pullback(VernonPullback *handle) : handle_(handle) {}
    Pullback(const Pullback &) = delete;
    Pullback &operator=(const Pullback &) = delete;
    Pullback(Pullback &&other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
    Pullback &operator=(Pullback &&other) noexcept {
        if (this != &other) {
            vernonProgramPullbackDestroy(handle_);
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }
    ~Pullback() { vernonProgramPullbackDestroy(handle_); }

    void apply(const VernonProgramArgument *arguments, size_t argumentCount, uint64_t maximumTemporaryBytes) const {
        if (!handle_)
            throw std::logic_error("pullback is empty");
        const VernonPullbackApplyOptions options{
            sizeof(VernonPullbackApplyOptions), VERNON_PULLBACK_APPLY_OPTIONS_VERSION, maximumTemporaryBytes, {}};
        if (vernonProgramPullbackApplyWithOptions(handle_, arguments, argumentCount, &options, nullptr) !=
            VERNON_STATUS_OK)
            throw std::runtime_error("pullback application failed");
    }

    void apply(const VernonProgramArgument *arguments, size_t argumentCount) const {
        apply(arguments, argumentCount, std::numeric_limits<uint64_t>::max());
    }

    explicit operator bool() const noexcept { return handle_ != nullptr; }
    VernonPullback *get() const noexcept { return handle_; }

private:
    VernonPullback *handle_{};
};

class ProgramExecutable;

class ProgramVariant {
public:
    ProgramVariant &set(std::string name, bool value) {
        return set(std::move(name), VERNON_PROGRAM_SPECIALIZATION_BOOL, value);
    }
    ProgramVariant &set(std::string name, int32_t value) {
        return set(std::move(name), VERNON_PROGRAM_SPECIALIZATION_I32, value);
    }
    ProgramVariant &set(std::string name, uint32_t value) {
        return set(std::move(name), VERNON_PROGRAM_SPECIALIZATION_U32, value);
    }
    ProgramVariant &set(std::string name, float value) {
        if (!std::isfinite(value))
            throw std::invalid_argument("Program f32 specialization must be finite");
        return set(std::move(name), VERNON_PROGRAM_SPECIALIZATION_F32, value);
    }
    ProgramVariant &set(std::string name, double value) {
        if (!std::isfinite(value))
            throw std::invalid_argument("Program f64 specialization must be finite");
        return set(std::move(name), VERNON_PROGRAM_SPECIALIZATION_F64, value);
    }

private:
    using Value = std::variant<bool, int32_t, uint32_t, float, double>;
    struct Entry {
        std::string name;
        VernonProgramSpecializationKind kind{};
        Value value;
    };

    ProgramVariant &set(std::string name, VernonProgramSpecializationKind kind, Value value) {
        if (name.empty())
            throw std::invalid_argument("Program specialization name must not be empty");
        for (const Entry &entry : entries_)
            if (entry.name == name)
                throw std::invalid_argument("Program specialization name is duplicated");
        entries_.push_back({std::move(name), kind, std::move(value)});
        return *this;
    }

    std::vector<VernonProgramSpecialization> native() const {
        std::vector<VernonProgramSpecialization> result;
        result.reserve(entries_.size());
        for (const Entry &entry : entries_) {
            VernonProgramSpecialization specialization{};
            specialization.struct_size = sizeof(specialization);
            specialization.name = {entry.name.data(), entry.name.size()};
            specialization.kind = entry.kind;
            std::visit(
                [&](const auto &value) {
                    using T = std::decay_t<decltype(value)>;
                    if constexpr (std::is_same_v<T, bool>)
                        specialization.value.boolean_value = value ? 1 : 0;
                    else if constexpr (std::is_same_v<T, int32_t>)
                        specialization.value.i32_value = value;
                    else if constexpr (std::is_same_v<T, uint32_t>)
                        specialization.value.u32_value = value;
                    else if constexpr (std::is_same_v<T, float>)
                        specialization.value.f32_value = value;
                    else
                        specialization.value.f64_value = value;
                },
                entry.value);
            result.push_back(specialization);
        }
        return result;
    }

    std::vector<Entry> entries_;
    friend class ProgramAsset;
    friend class ProgramGraph;
};

class ProgramAsset {
public:
    static ProgramAsset load(VernonRuntimeContext *context, const void *bundle, size_t bundleSize,
                             const VernonProgramBundleLoadOptions *options = nullptr) {
        VernonProgramBundle *handle = vernonRuntimeLoadProgramBundleWithOptions(context, bundle, bundleSize, options);
        if (!handle)
            throw std::runtime_error("failed to load Program bundle");
        return ProgramAsset(handle);
    }

    VernonProgramBundle *get() const noexcept { return handle_.get(); }
    ProgramExecutable resolve() const;
    ProgramExecutable resolve(const ProgramVariant &variant) const;

private:
    explicit ProgramAsset(VernonProgramBundle *handle) : handle_(handle, vernonRuntimeProgramBundleDestroy) {}
    std::shared_ptr<VernonProgramBundle> handle_;
};

struct ProgramNodeBinding {
    VernonProgramNodeBindingToken token{sizeof(VernonProgramNodeBindingToken), 0, 0, 0, VERNON_PROGRAM_TENSOR};
};

struct ProgramGraphValue {
    VernonProgramGraphValue value{sizeof(VernonProgramGraphValue), 0, 0, VERNON_PROGRAM_TENSOR};
};

struct ProgramGraphStorage {
    VernonProgramGraphStorage storage{sizeof(VernonProgramGraphStorage), 0, 0, VERNON_PROGRAM_TENSOR};
};

class ProgramGraph;
class ProgramInvocation;
namespace detail {
struct ProgramGraphState {
    explicit ProgramGraphState(VernonRuntimeContext *context) : handle(vernonRuntimeProgramGraphCreate(context)) {}
    ~ProgramGraphState() { vernonRuntimeProgramGraphDestroy(handle); }
    VernonProgramGraph *handle{};
};
} // namespace detail

class ProgramNode {
public:
    ProgramNodeBinding boundary(VernonProgramBoundaryRole role, std::string_view name) const;
    VernonProgramNodeId id() const noexcept { return id_; }

private:
    ProgramNode(std::shared_ptr<detail::ProgramGraphState> owner, VernonProgramNodeId id)
        : owner_(std::move(owner)), id_(id) {}
    std::shared_ptr<detail::ProgramGraphState> owner_;
    VernonProgramNodeId id_{};
    friend class ProgramGraph;
    friend class ProgramInvocation;
};

using ProgramNodeHandle = ProgramNode;

class ProgramExecutable {
public:
    ProgramExecutable() = default;
    ProgramExecutable(const ProgramExecutable &) = default;
    ProgramExecutable &operator=(const ProgramExecutable &) = default;
    ProgramExecutable(ProgramExecutable &&other) noexcept = default;
    ProgramExecutable &operator=(ProgramExecutable &&other) noexcept = default;

    explicit operator bool() const noexcept { return static_cast<bool>(handle_); }
    VernonProgramExecutable *get() const noexcept { return handle_.get(); }
    std::string_view id() const noexcept {
        if (!handle_)
            return {};
        const VernonStringView value = vernonRuntimeProgramExecutableGetId(handle_.get());
        return {value.data, value.size};
    }

private:
    explicit ProgramExecutable(VernonProgramExecutable *handle,
                               std::shared_ptr<detail::ProgramGraphState> graphOwner = {})
        : handle_(handle, vernonRuntimeProgramExecutableDestroy), graphOwner_(std::move(graphOwner)) {}

    std::shared_ptr<VernonProgramExecutable> handle_;
    std::shared_ptr<detail::ProgramGraphState> graphOwner_;
    friend class ProgramAsset;
    friend class ProgramInstance;
    friend class ProgramGraph;
};

inline ProgramExecutable ProgramAsset::resolve() const {
    VernonProgramExecutable *executable = vernonRuntimeResolveProgram(handle_.get(), nullptr);
    if (!executable)
        throw std::runtime_error("failed to resolve Program variant");
    return ProgramExecutable(executable);
}

inline ProgramExecutable ProgramAsset::resolve(const ProgramVariant &variant) const {
    const std::vector<VernonProgramSpecialization> specializations = variant.native();
    const VernonProgramVariantSelector selector{
        sizeof(VernonProgramVariantSelector), specializations.data(), specializations.size(), {}};
    VernonProgramExecutable *executable = vernonRuntimeResolveProgram(handle_.get(), &selector);
    if (!executable)
        throw std::runtime_error("failed to resolve Program variant");
    return ProgramExecutable(executable);
}

class ProgramGraph {
public:
    explicit ProgramGraph(VernonRuntimeContext *context)
        : state_(std::make_shared<detail::ProgramGraphState>(context)) {
        if (!state_->handle)
            throw std::runtime_error("failed to create ProgramGraph");
    }
    ProgramGraph(const ProgramGraph &) = delete;
    ProgramGraph &operator=(const ProgramGraph &) = delete;
    ProgramGraph(ProgramGraph &&) noexcept = default;
    ProgramGraph &operator=(ProgramGraph &&) noexcept = default;

    ProgramNode add(const ProgramAsset &asset) { return add(asset, nullptr); }

    ProgramNode add(const ProgramAsset &asset, const ProgramVariant &variant) {
        const std::vector<VernonProgramSpecialization> specializations = variant.native();
        const VernonProgramVariantSelector selector{
            sizeof(VernonProgramVariantSelector), specializations.data(), specializations.size(), {}};
        return add(asset, &selector);
    }

private:
    ProgramNode add(const ProgramAsset &asset, const VernonProgramVariantSelector *selector) {
        VernonProgramNodeId id{};
        if (!state_ || !state_->handle ||
            vernonRuntimeProgramGraphAddProgram(state_->handle, asset.get(), selector, &id) != VERNON_STATUS_OK)
            throw std::runtime_error("failed to add ProgramGraph node");
        return ProgramNode(state_, id);
    }

public:
    ProgramGraphValue createValue(ProgramNodeBinding source) {
        ProgramGraphValue value;
        if (!state_ || !state_->handle ||
            vernonRuntimeProgramGraphCreateValue(state_->handle, &source.token, &value.value) != VERNON_STATUS_OK)
            throw std::runtime_error("failed to create ProgramGraph Value");
        return value;
    }

    ProgramGraph &connect(ProgramGraphValue value, ProgramNodeBinding destination) {
        if (!state_ || !state_->handle ||
            vernonRuntimeProgramGraphConnectValue(state_->handle, &value.value, &destination.token) != VERNON_STATUS_OK)
            throw std::runtime_error("failed to connect ProgramGraph Value");
        return *this;
    }

    ProgramGraphStorage createStorage(ProgramNodeBinding firstVersion) {
        ProgramGraphStorage storage;
        if (!state_ || !state_->handle ||
            vernonRuntimeProgramGraphCreateStorage(state_->handle, &firstVersion.token, &storage.storage) !=
                VERNON_STATUS_OK)
            throw std::runtime_error("failed to create ProgramGraph Storage");
        return storage;
    }

    ProgramGraph &append(ProgramGraphStorage storage, ProgramNodeBinding nextVersion) {
        if (!state_ || !state_->handle ||
            vernonRuntimeProgramGraphAppendStorage(state_->handle, &storage.storage, &nextVersion.token) !=
                VERNON_STATUS_OK)
            throw std::runtime_error("failed to append ProgramGraph Storage version");
        return *this;
    }

    ProgramGraph &exportBoundary(ProgramNodeBinding boundary, std::string_view name) {
        const VernonStringView view{name.data(), name.size()};
        if (!state_ || !state_->handle ||
            vernonRuntimeProgramGraphExportBoundary(state_->handle, &boundary.token, view) != VERNON_STATUS_OK)
            throw std::runtime_error("failed to export ProgramGraph boundary");
        return *this;
    }

    ProgramGraph &exportValue(ProgramGraphValue value, std::string_view name) {
        const VernonStringView view{name.data(), name.size()};
        if (!state_ || !state_->handle ||
            vernonRuntimeProgramGraphExportValue(state_->handle, &value.value, view) != VERNON_STATUS_OK)
            throw std::runtime_error("failed to export ProgramGraph Value");
        return *this;
    }

    ProgramGraph &exportStorage(ProgramGraphStorage storage, std::string_view name) {
        const VernonStringView view{name.data(), name.size()};
        if (!state_ || !state_->handle ||
            vernonRuntimeProgramGraphExportStorage(state_->handle, &storage.storage, view) != VERNON_STATUS_OK)
            throw std::runtime_error("failed to export ProgramGraph Storage");
        return *this;
    }

    ProgramExecutable compile() {
        VernonProgramExecutable *executable = vernonRuntimeResolveProgramGraph(state_ ? state_->handle : nullptr);
        if (!executable)
            throw std::runtime_error("failed to compile ProgramGraph");
        return ProgramExecutable(executable, state_);
    }

private:
    std::shared_ptr<detail::ProgramGraphState> state_;
};

inline ProgramNodeBinding ProgramNode::boundary(VernonProgramBoundaryRole role, std::string_view name) const {
    if (!owner_)
        throw std::logic_error("ProgramGraph node is empty");
    ProgramNodeBinding result;
    const VernonStringView view{name.data(), name.size()};
    if (vernonRuntimeProgramGraphFindBoundary(owner_->handle, id_, role, view, &result.token) != VERNON_STATUS_OK)
        throw std::runtime_error("ProgramGraph node boundary was not found");
    return result;
}

class ProgramInvocation {
public:
    ProgramInvocation() = default;
    explicit ProgramInvocation(VernonProgramInvocation *handle,
                               std::shared_ptr<detail::ProgramGraphState> graphOwner = {})
        : handle_(handle), graphOwner_(std::move(graphOwner)) {}
    ProgramInvocation(const ProgramInvocation &) = delete;
    ProgramInvocation &operator=(const ProgramInvocation &) = delete;
    ProgramInvocation(ProgramInvocation &&other) noexcept
        : handle_(std::exchange(other.handle_, nullptr)), graphOwner_(std::move(other.graphOwner_)) {}
    ProgramInvocation &operator=(ProgramInvocation &&other) noexcept {
        if (this != &other) {
            vernonRuntimeProgramInvocationDestroy(handle_);
            handle_ = std::exchange(other.handle_, nullptr);
            graphOwner_ = std::move(other.graphOwner_);
        }
        return *this;
    }
    ~ProgramInvocation() { vernonRuntimeProgramInvocationDestroy(handle_); }

    ProgramInvocation &bind(const VernonProgramBindingToken &token, const VernonProgramArgument &argument,
                            const VernonProgramResourceLease *lease = nullptr, uint64_t uploadBytes = 0,
                            uint64_t uploadRanges = 0) {
        if (!handle_ || vernonRuntimeProgramInvocationBind(handle_, &token, &argument, lease, uploadBytes,
                                                           uploadRanges) != VERNON_STATUS_OK)
            throw std::runtime_error("failed to bind Program invocation");
        return *this;
    }

    ProgramInvocation &bindRenderPass(uint32_t slot, const VernonProgramBindingToken &token,
                                      const VernonRenderPass &renderPass,
                                      const VernonProgramResourceLease *leases = nullptr, size_t leaseCount = 0) {
        if (!handle_ || vernonRuntimeProgramInvocationBindRenderPass(handle_, slot, &token, &renderPass, leases,
                                                                     leaseCount) != VERNON_STATUS_OK)
            throw std::runtime_error("failed to bind Program RenderPass control");
        return *this;
    }

    ProgramInvocation &bindDrawCommand(uint32_t slot, const VernonProgramBindingToken &token,
                                       const VernonDrawCommand &draw,
                                       const VernonProgramResourceLease *lease = nullptr) {
        if (!handle_ ||
            vernonRuntimeProgramInvocationBindDrawCommand(handle_, slot, &token, &draw, lease) != VERNON_STATUS_OK)
            throw std::runtime_error("failed to bind Program DrawCommand control");
        return *this;
    }

    ProgramInvocation &bindDynamicState(uint32_t slot, const VernonProgramBindingToken &token,
                                        const VernonDynamicState &dynamicState) {
        if (!handle_ ||
            vernonRuntimeProgramInvocationBindDynamicState(handle_, slot, &token, &dynamicState) != VERNON_STATUS_OK)
            throw std::runtime_error("failed to bind Program DynamicState control");
        return *this;
    }

    ProgramInvocation &setAutodiffOptions(const std::optional<uint64_t> &checkpointMemoryBudget,
                                          std::string_view checkpointPolicy) {
        VernonProgramAutodiffInvocationOptions options{};
        options.struct_size = sizeof(options);
        options.abi_version = VERNON_PROGRAM_AUTODIFF_INVOCATION_OPTIONS_VERSION;
        options.has_checkpoint_memory_budget = checkpointMemoryBudget.has_value();
        options.checkpoint_memory_budget = checkpointMemoryBudget.value_or(0);
        options.checkpoint_policy = {checkpointPolicy.data(), checkpointPolicy.size()};
        if (!handle_ || vernonRuntimeProgramInvocationSetAutodiffOptions(handle_, &options) != VERNON_STATUS_OK)
            throw std::runtime_error("failed to set Program invocation autodiff options");
        return *this;
    }

    void execute(bool retainPullback = true) {
        if (!handle_)
            throw std::logic_error("Program invocation is empty");
        if (vernonRuntimeProgramInvocationExecute(handle_, retainPullback, nullptr) != VERNON_STATUS_OK)
            throw std::runtime_error("Program invocation execution failed");
    }

    Pullback commit() {
        if (!handle_)
            throw std::logic_error("Program invocation is empty");
        VernonPullback *pullback = nullptr;
        if (vernonRuntimeProgramInvocationCommit(handle_, &pullback) != VERNON_STATUS_OK)
            throw std::runtime_error("Program invocation commit failed");
        return Pullback(pullback);
    }

    Pullback pullback(const ProgramNodeHandle &node) {
        if (!handle_)
            throw std::logic_error("Program invocation is empty");
        if (!graphOwner_ || graphOwner_ != node.owner_)
            throw std::invalid_argument("ProgramGraph node belongs to another graph");
        VernonPullback *pullback = nullptr;
        if (vernonRuntimeProgramInvocationGetNodePullback(handle_, node.id(), &pullback) != VERNON_STATUS_OK)
            throw std::runtime_error("ProgramGraph node pullback is unavailable");
        return Pullback(pullback);
    }

    void rollback() {
        if (handle_)
            vernonRuntimeProgramInvocationRollback(handle_);
    }

    VernonProgramInvocation *get() const noexcept { return handle_; }

private:
    VernonProgramInvocation *handle_{};
    std::shared_ptr<detail::ProgramGraphState> graphOwner_;
};

class ProgramInstance {
public:
    explicit ProgramInstance(ProgramExecutable &executable)
        : executable_(executable.handle_), graphOwner_(executable.graphOwner_),
          handle_(vernonRuntimeProgramInstanceCreate(executable_.get())) {
        if (!handle_)
            throw std::runtime_error("failed to create Program instance");
    }
    ProgramInstance(const ProgramInstance &) = delete;
    ProgramInstance &operator=(const ProgramInstance &) = delete;
    ProgramInstance(ProgramInstance &&other) noexcept
        : executable_(std::move(other.executable_)), graphOwner_(std::move(other.graphOwner_)),
          handle_(std::exchange(other.handle_, nullptr)) {}
    ProgramInstance &operator=(ProgramInstance &&other) noexcept {
        if (this != &other) {
            vernonRuntimeProgramInstanceDestroy(handle_);
            executable_ = std::move(other.executable_);
            graphOwner_ = std::move(other.graphOwner_);
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }
    ~ProgramInstance() { vernonRuntimeProgramInstanceDestroy(handle_); }

    ProgramInvocation begin() {
        VernonProgramInvocation *invocation = vernonRuntimeProgramInstanceBeginInvocation(handle_);
        if (!invocation)
            throw std::runtime_error("failed to begin Program invocation");
        return ProgramInvocation(invocation, graphOwner_);
    }

    VernonProgramBindingTelemetry telemetry() const {
        VernonProgramBindingTelemetry result{};
        result.struct_size = sizeof(result);
        if (vernonRuntimeProgramInstanceGetTelemetry(handle_, &result) != VERNON_STATUS_OK)
            throw std::runtime_error("failed to query Program binding telemetry");
        return result;
    }

    VernonProgramInstance *get() const noexcept { return handle_; }

private:
    std::shared_ptr<VernonProgramExecutable> executable_;
    std::shared_ptr<detail::ProgramGraphState> graphOwner_;
    VernonProgramInstance *handle_{};
};

} // namespace vernon::runtime

#endif
