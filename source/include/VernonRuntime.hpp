#ifndef VERNON_RUNTIME_HPP
#define VERNON_RUNTIME_HPP

#include "VernonError.hpp"
#include "VernonRuntime.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace vernon::runtime {

namespace detail {
[[nodiscard]] inline RuntimeError runtimeError(VernonStatus status, const char *operation) noexcept {
    return runtimeErrorFromStatus(status, {operation});
}

[[nodiscard]] inline RuntimeError runtimeError(VernonRuntimeOperationStatus status, const char *operation) noexcept {
    return {status == VERNON_RUNTIME_OPERATION_OK ? RuntimeErrorCode::InternalFailure
                                                  : static_cast<RuntimeErrorCode>(status),
            {operation}};
}

[[nodiscard]] inline RuntimeError invalidArgument(const char *operation) noexcept {
    return {RuntimeErrorCode::InvalidArgument, {operation}};
}

[[nodiscard]] inline RuntimeError internalFailure(const char *operation) noexcept {
    return {RuntimeErrorCode::InternalFailure, {operation}};
}

[[nodiscard]] inline RuntimeError resourceExhausted(const char *operation) noexcept {
    return {RuntimeErrorCode::ResourceExhausted, {operation}};
}
} // namespace detail

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

    [[nodiscard]] Result<void, RuntimeError> apply(const VernonProgramArgument *arguments, size_t argumentCount,
                                                   uint64_t maximumTemporaryBytes) const noexcept {
        if (!handle_)
            return Result<void, RuntimeError>{err(detail::invalidArgument("ProgramPullback.apply"))};
        const VernonPullbackApplyOptions options{
            sizeof(VernonPullbackApplyOptions), VERNON_PULLBACK_APPLY_OPTIONS_VERSION, maximumTemporaryBytes, {}};
        const VernonStatus status =
            vernonProgramPullbackApplyWithOptions(handle_, arguments, argumentCount, &options, nullptr);
        if (status != VERNON_STATUS_OK)
            return Result<void, RuntimeError>{err(detail::runtimeError(status, "ProgramPullback.apply"))};
        return Result<void, RuntimeError>{ok()};
    }

    [[nodiscard]] Result<void, RuntimeError> apply(const VernonProgramArgument *arguments,
                                                   size_t argumentCount) const noexcept {
        return apply(arguments, argumentCount, std::numeric_limits<uint64_t>::max());
    }

    explicit operator bool() const noexcept { return handle_ != nullptr; }
    VernonPullback *get() const noexcept { return handle_; }

private:
    VernonPullback *handle_{};
};

class ProgramExecutable;

class ProgramVariant {
public:
    [[nodiscard]] Result<void, RuntimeError> set(std::string name, bool value) noexcept {
        return set(std::move(name), VERNON_PROGRAM_SPECIALIZATION_BOOL, value);
    }
    [[nodiscard]] Result<void, RuntimeError> set(std::string name, int32_t value) noexcept {
        return set(std::move(name), VERNON_PROGRAM_SPECIALIZATION_I32, value);
    }
    [[nodiscard]] Result<void, RuntimeError> set(std::string name, uint32_t value) noexcept {
        return set(std::move(name), VERNON_PROGRAM_SPECIALIZATION_U32, value);
    }
    [[nodiscard]] Result<void, RuntimeError> set(std::string name, float value) noexcept {
        if (!std::isfinite(value))
            return Result<void, RuntimeError>{err(detail::invalidArgument("ProgramVariant.set"))};
        return set(std::move(name), VERNON_PROGRAM_SPECIALIZATION_F32, value);
    }
    [[nodiscard]] Result<void, RuntimeError> set(std::string name, double value) noexcept {
        if (!std::isfinite(value))
            return Result<void, RuntimeError>{err(detail::invalidArgument("ProgramVariant.set"))};
        return set(std::move(name), VERNON_PROGRAM_SPECIALIZATION_F64, value);
    }

private:
    using Value = std::variant<bool, int32_t, uint32_t, float, double>;
    struct Entry {
        std::string name;
        VernonProgramSpecializationKind kind{};
        Value value;
    };

    Result<void, RuntimeError> set(std::string name, VernonProgramSpecializationKind kind, Value value) noexcept {
        try {
            if (name.empty())
                return Result<void, RuntimeError>{err(detail::invalidArgument("ProgramVariant.set"))};
            for (const Entry &entry : entries_)
                if (entry.name == name)
                    return Result<void, RuntimeError>{err(detail::invalidArgument("ProgramVariant.set"))};
            entries_.push_back({std::move(name), kind, std::move(value)});
            return Result<void, RuntimeError>{ok()};
        } catch (const std::bad_alloc &) {
            return Result<void, RuntimeError>{err(detail::resourceExhausted("ProgramVariant.set"))};
        }
    }

    Result<std::vector<VernonProgramSpecialization>, RuntimeError> native(const char *operation) const noexcept {
        try {
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
            return Result<std::vector<VernonProgramSpecialization>, RuntimeError>{ok(std::move(result))};
        } catch (const std::bad_alloc &) {
            return Result<std::vector<VernonProgramSpecialization>, RuntimeError>{
                err(detail::resourceExhausted(operation))};
        }
    }

    std::vector<Entry> entries_;
    friend class ProgramAsset;
    friend class ProgramGraph;
};

class ProgramAsset {
public:
    [[nodiscard]] static Result<ProgramAsset, RuntimeError>
    load(VernonRuntimeContext *context, const void *bundle, size_t bundleSize,
         const VernonProgramBundleLoadOptions *options = nullptr) noexcept {
        if (!context || !bundle)
            return Result<ProgramAsset, RuntimeError>{err(detail::invalidArgument("ProgramAsset.load"))};
        VernonProgramBundle *handle = nullptr;
        const VernonRuntimeOperationStatus status =
            vernonRuntimeLoadProgramBundleWithOptionsResult(context, bundle, bundleSize, options, &handle);
        if (status != VERNON_RUNTIME_OPERATION_OK)
            return Result<ProgramAsset, RuntimeError>{err(detail::runtimeError(status, "ProgramAsset.load"))};
        try {
            return Result<ProgramAsset, RuntimeError>{ok(ProgramAsset(handle))};
        } catch (const std::bad_alloc &) {
            return Result<ProgramAsset, RuntimeError>{err(detail::resourceExhausted("ProgramAsset.load"))};
        }
    }

    VernonProgramBundle *get() const noexcept { return handle_.get(); }
    [[nodiscard]] Result<ProgramExecutable, RuntimeError> resolve() const noexcept;
    [[nodiscard]] Result<ProgramExecutable, RuntimeError> resolve(const ProgramVariant &variant) const noexcept;

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
    explicit ProgramGraphState(VernonProgramGraph *value) noexcept : handle(value) {}
    ~ProgramGraphState() { vernonRuntimeProgramGraphDestroy(handle); }
    VernonProgramGraph *handle{};
};
} // namespace detail

class ProgramNode {
public:
    [[nodiscard]] Result<ProgramNodeBinding, RuntimeError> boundary(VernonProgramBoundaryRole role,
                                                                    std::string_view name) const noexcept;
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

inline Result<ProgramExecutable, RuntimeError> ProgramAsset::resolve() const noexcept {
    if (!handle_)
        return Result<ProgramExecutable, RuntimeError>{err(detail::invalidArgument("ProgramAsset.resolve"))};
    VernonProgramExecutable *executable = nullptr;
    const VernonRuntimeOperationStatus status = vernonRuntimeResolveProgramResult(handle_.get(), nullptr, &executable);
    if (status != VERNON_RUNTIME_OPERATION_OK)
        return Result<ProgramExecutable, RuntimeError>{err(detail::runtimeError(status, "ProgramAsset.resolve"))};
    try {
        return Result<ProgramExecutable, RuntimeError>{ok(ProgramExecutable(executable))};
    } catch (const std::bad_alloc &) {
        return Result<ProgramExecutable, RuntimeError>{err(detail::resourceExhausted("ProgramAsset.resolve"))};
    }
}

inline Result<ProgramExecutable, RuntimeError> ProgramAsset::resolve(const ProgramVariant &variant) const noexcept {
    if (!handle_)
        return Result<ProgramExecutable, RuntimeError>{err(detail::invalidArgument("ProgramAsset.resolve"))};
    auto native = variant.native("ProgramAsset.resolve");
    if (native.isErr())
        return Result<ProgramExecutable, RuntimeError>{err(std::move(native).error())};
    const std::vector<VernonProgramSpecialization> specializations = std::move(native).value();
    const VernonProgramVariantSelector selector{
        sizeof(VernonProgramVariantSelector), specializations.data(), specializations.size(), {}};
    VernonProgramExecutable *executable = nullptr;
    const VernonRuntimeOperationStatus status =
        vernonRuntimeResolveProgramResult(handle_.get(), &selector, &executable);
    if (status != VERNON_RUNTIME_OPERATION_OK)
        return Result<ProgramExecutable, RuntimeError>{err(detail::runtimeError(status, "ProgramAsset.resolve"))};
    try {
        return Result<ProgramExecutable, RuntimeError>{ok(ProgramExecutable(executable))};
    } catch (const std::bad_alloc &) {
        return Result<ProgramExecutable, RuntimeError>{err(detail::resourceExhausted("ProgramAsset.resolve"))};
    }
}

class ProgramGraph {
public:
    [[nodiscard]] static Result<ProgramGraph, RuntimeError> create(VernonRuntimeContext *context) noexcept {
        if (!context)
            return Result<ProgramGraph, RuntimeError>{err(detail::invalidArgument("ProgramGraph.create"))};
        VernonProgramGraph *handle = nullptr;
        const VernonRuntimeOperationStatus status = vernonRuntimeProgramGraphCreateResult(context, &handle);
        if (status != VERNON_RUNTIME_OPERATION_OK)
            return Result<ProgramGraph, RuntimeError>{err(detail::runtimeError(status, "ProgramGraph.create"))};
        try {
            auto state = std::make_shared<detail::ProgramGraphState>(handle);
            return Result<ProgramGraph, RuntimeError>{ok(ProgramGraph(std::move(state)))};
        } catch (const std::bad_alloc &) {
            vernonRuntimeProgramGraphDestroy(handle);
            return Result<ProgramGraph, RuntimeError>{err(detail::resourceExhausted("ProgramGraph.create"))};
        }
    }
    ProgramGraph(const ProgramGraph &) = delete;
    ProgramGraph &operator=(const ProgramGraph &) = delete;
    ProgramGraph(ProgramGraph &&) noexcept = default;
    ProgramGraph &operator=(ProgramGraph &&) noexcept = default;

    [[nodiscard]] Result<ProgramNode, RuntimeError> add(const ProgramAsset &asset) noexcept {
        return add(asset, nullptr);
    }

    [[nodiscard]] Result<ProgramNode, RuntimeError> add(const ProgramAsset &asset,
                                                        const ProgramVariant &variant) noexcept {
        auto native = variant.native("ProgramGraph.add");
        if (native.isErr())
            return Result<ProgramNode, RuntimeError>{err(std::move(native).error())};
        const std::vector<VernonProgramSpecialization> specializations = std::move(native).value();
        const VernonProgramVariantSelector selector{
            sizeof(VernonProgramVariantSelector), specializations.data(), specializations.size(), {}};
        return add(asset, &selector);
    }

private:
    explicit ProgramGraph(std::shared_ptr<detail::ProgramGraphState> state) : state_(std::move(state)) {}

    Result<ProgramNode, RuntimeError> add(const ProgramAsset &asset,
                                          const VernonProgramVariantSelector *selector) noexcept {
        VernonProgramNodeId id{};
        if (!state_ || !state_->handle)
            return Result<ProgramNode, RuntimeError>{err(detail::invalidArgument("ProgramGraph.add"))};
        const VernonStatus status = vernonRuntimeProgramGraphAddProgram(state_->handle, asset.get(), selector, &id);
        if (status != VERNON_STATUS_OK)
            return Result<ProgramNode, RuntimeError>{err(detail::runtimeError(status, "ProgramGraph.add"))};
        return Result<ProgramNode, RuntimeError>{ok(ProgramNode(state_, id))};
    }

public:
    [[nodiscard]] Result<ProgramGraphValue, RuntimeError> createValue(ProgramNodeBinding source) noexcept {
        ProgramGraphValue value;
        if (!state_ || !state_->handle)
            return Result<ProgramGraphValue, RuntimeError>{err(detail::invalidArgument("ProgramGraph.createValue"))};
        const VernonStatus status = vernonRuntimeProgramGraphCreateValue(state_->handle, &source.token, &value.value);
        if (status != VERNON_STATUS_OK)
            return Result<ProgramGraphValue, RuntimeError>{
                err(detail::runtimeError(status, "ProgramGraph.createValue"))};
        return Result<ProgramGraphValue, RuntimeError>{ok(value)};
    }

    [[nodiscard]] Result<void, RuntimeError> connect(ProgramGraphValue value, ProgramNodeBinding destination) noexcept {
        if (!state_ || !state_->handle)
            return Result<void, RuntimeError>{err(detail::invalidArgument("ProgramGraph.connect"))};
        const VernonStatus status =
            vernonRuntimeProgramGraphConnectValue(state_->handle, &value.value, &destination.token);
        if (status != VERNON_STATUS_OK)
            return Result<void, RuntimeError>{err(detail::runtimeError(status, "ProgramGraph.connect"))};
        return Result<void, RuntimeError>{ok()};
    }

    [[nodiscard]] Result<ProgramGraphStorage, RuntimeError> createStorage(ProgramNodeBinding firstVersion) noexcept {
        ProgramGraphStorage storage;
        if (!state_ || !state_->handle)
            return Result<ProgramGraphStorage, RuntimeError>{
                err(detail::invalidArgument("ProgramGraph.createStorage"))};
        const VernonStatus status =
            vernonRuntimeProgramGraphCreateStorage(state_->handle, &firstVersion.token, &storage.storage);
        if (status != VERNON_STATUS_OK)
            return Result<ProgramGraphStorage, RuntimeError>{
                err(detail::runtimeError(status, "ProgramGraph.createStorage"))};
        return Result<ProgramGraphStorage, RuntimeError>{ok(storage)};
    }

    [[nodiscard]] Result<void, RuntimeError> append(ProgramGraphStorage storage,
                                                    ProgramNodeBinding nextVersion) noexcept {
        if (!state_ || !state_->handle)
            return Result<void, RuntimeError>{err(detail::invalidArgument("ProgramGraph.append"))};
        const VernonStatus status =
            vernonRuntimeProgramGraphAppendStorage(state_->handle, &storage.storage, &nextVersion.token);
        if (status != VERNON_STATUS_OK)
            return Result<void, RuntimeError>{err(detail::runtimeError(status, "ProgramGraph.append"))};
        return Result<void, RuntimeError>{ok()};
    }

    [[nodiscard]] Result<void, RuntimeError> exportBoundary(ProgramNodeBinding boundary,
                                                            std::string_view name) noexcept {
        const VernonStringView view{name.data(), name.size()};
        if (!state_ || !state_->handle)
            return Result<void, RuntimeError>{err(detail::invalidArgument("ProgramGraph.exportBoundary"))};
        const VernonStatus status = vernonRuntimeProgramGraphExportBoundary(state_->handle, &boundary.token, view);
        if (status != VERNON_STATUS_OK)
            return Result<void, RuntimeError>{err(detail::runtimeError(status, "ProgramGraph.exportBoundary"))};
        return Result<void, RuntimeError>{ok()};
    }

    [[nodiscard]] Result<void, RuntimeError> exportValue(ProgramGraphValue value, std::string_view name) noexcept {
        const VernonStringView view{name.data(), name.size()};
        if (!state_ || !state_->handle)
            return Result<void, RuntimeError>{err(detail::invalidArgument("ProgramGraph.exportValue"))};
        const VernonStatus status = vernonRuntimeProgramGraphExportValue(state_->handle, &value.value, view);
        if (status != VERNON_STATUS_OK)
            return Result<void, RuntimeError>{err(detail::runtimeError(status, "ProgramGraph.exportValue"))};
        return Result<void, RuntimeError>{ok()};
    }

    [[nodiscard]] Result<void, RuntimeError> exportStorage(ProgramGraphStorage storage,
                                                           std::string_view name) noexcept {
        const VernonStringView view{name.data(), name.size()};
        if (!state_ || !state_->handle)
            return Result<void, RuntimeError>{err(detail::invalidArgument("ProgramGraph.exportStorage"))};
        const VernonStatus status = vernonRuntimeProgramGraphExportStorage(state_->handle, &storage.storage, view);
        if (status != VERNON_STATUS_OK)
            return Result<void, RuntimeError>{err(detail::runtimeError(status, "ProgramGraph.exportStorage"))};
        return Result<void, RuntimeError>{ok()};
    }

    [[nodiscard]] Result<ProgramExecutable, RuntimeError> compile() noexcept {
        if (!state_ || !state_->handle)
            return Result<ProgramExecutable, RuntimeError>{err(detail::invalidArgument("ProgramGraph.compile"))};
        VernonProgramExecutable *executable = nullptr;
        const VernonRuntimeOperationStatus status = vernonRuntimeResolveProgramGraphResult(state_->handle, &executable);
        if (status != VERNON_RUNTIME_OPERATION_OK)
            return Result<ProgramExecutable, RuntimeError>{err(detail::runtimeError(status, "ProgramGraph.compile"))};
        try {
            return Result<ProgramExecutable, RuntimeError>{ok(ProgramExecutable(executable, state_))};
        } catch (const std::bad_alloc &) {
            return Result<ProgramExecutable, RuntimeError>{err(detail::resourceExhausted("ProgramGraph.compile"))};
        }
    }

private:
    std::shared_ptr<detail::ProgramGraphState> state_;
};

inline Result<ProgramNodeBinding, RuntimeError> ProgramNode::boundary(VernonProgramBoundaryRole role,
                                                                      std::string_view name) const noexcept {
    if (!owner_)
        return Result<ProgramNodeBinding, RuntimeError>{err(detail::invalidArgument("ProgramNode.boundary"))};
    ProgramNodeBinding result;
    const VernonStringView view{name.data(), name.size()};
    const VernonStatus status = vernonRuntimeProgramGraphFindBoundary(owner_->handle, id_, role, view, &result.token);
    if (status != VERNON_STATUS_OK)
        return Result<ProgramNodeBinding, RuntimeError>{err(detail::runtimeError(status, "ProgramNode.boundary"))};
    return Result<ProgramNodeBinding, RuntimeError>{ok(result)};
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

    [[nodiscard]] Result<void, RuntimeError> bind(const VernonProgramBindingToken &token,
                                                  const VernonProgramArgument &argument,
                                                  const VernonProgramResourceLease *lease = nullptr,
                                                  uint64_t uploadBytes = 0, uint64_t uploadRanges = 0) noexcept {
        if (!handle_)
            return Result<void, RuntimeError>{err(detail::invalidArgument("ProgramInvocation.bind"))};
        const VernonStatus status =
            vernonRuntimeProgramInvocationBind(handle_, &token, &argument, lease, uploadBytes, uploadRanges);
        if (status != VERNON_STATUS_OK)
            return Result<void, RuntimeError>{err(detail::runtimeError(status, "ProgramInvocation.bind"))};
        return Result<void, RuntimeError>{ok()};
    }

    [[nodiscard]] Result<void, RuntimeError> bindRenderPass(uint32_t slot, const VernonProgramBindingToken &token,
                                                            const VernonRenderPass &renderPass,
                                                            const VernonProgramResourceLease *leases = nullptr,
                                                            size_t leaseCount = 0) noexcept {
        if (!handle_)
            return Result<void, RuntimeError>{err(detail::invalidArgument("ProgramInvocation.bindRenderPass"))};
        const VernonStatus status =
            vernonRuntimeProgramInvocationBindRenderPass(handle_, slot, &token, &renderPass, leases, leaseCount);
        if (status != VERNON_STATUS_OK)
            return Result<void, RuntimeError>{err(detail::runtimeError(status, "ProgramInvocation.bindRenderPass"))};
        return Result<void, RuntimeError>{ok()};
    }

    [[nodiscard]] Result<void, RuntimeError>
    bindDrawCommand(uint32_t slot, const VernonProgramBindingToken &token, const VernonDrawCommand &draw,
                    const VernonProgramResourceLease *lease = nullptr) noexcept {
        if (!handle_)
            return Result<void, RuntimeError>{err(detail::invalidArgument("ProgramInvocation.bindDrawCommand"))};
        const VernonStatus status = vernonRuntimeProgramInvocationBindDrawCommand(handle_, slot, &token, &draw, lease);
        if (status != VERNON_STATUS_OK)
            return Result<void, RuntimeError>{err(detail::runtimeError(status, "ProgramInvocation.bindDrawCommand"))};
        return Result<void, RuntimeError>{ok()};
    }

    [[nodiscard]] Result<void, RuntimeError> bindDynamicState(uint32_t slot, const VernonProgramBindingToken &token,
                                                              const VernonDynamicState &dynamicState) noexcept {
        if (!handle_)
            return Result<void, RuntimeError>{err(detail::invalidArgument("ProgramInvocation.bindDynamicState"))};
        const VernonStatus status =
            vernonRuntimeProgramInvocationBindDynamicState(handle_, slot, &token, &dynamicState);
        if (status != VERNON_STATUS_OK)
            return Result<void, RuntimeError>{err(detail::runtimeError(status, "ProgramInvocation.bindDynamicState"))};
        return Result<void, RuntimeError>{ok()};
    }

    [[nodiscard]] Result<void, RuntimeError> setAutodiffOptions(const std::optional<uint64_t> &checkpointMemoryBudget,
                                                                std::string_view checkpointPolicy) noexcept {
        VernonProgramAutodiffInvocationOptions options{};
        options.struct_size = sizeof(options);
        options.abi_version = VERNON_PROGRAM_AUTODIFF_INVOCATION_OPTIONS_VERSION;
        options.has_checkpoint_memory_budget = checkpointMemoryBudget.has_value();
        options.checkpoint_memory_budget = checkpointMemoryBudget.value_or(0);
        options.checkpoint_policy = {checkpointPolicy.data(), checkpointPolicy.size()};
        if (!handle_)
            return Result<void, RuntimeError>{err(detail::invalidArgument("ProgramInvocation.setAutodiffOptions"))};
        const VernonStatus status = vernonRuntimeProgramInvocationSetAutodiffOptions(handle_, &options);
        if (status != VERNON_STATUS_OK)
            return Result<void, RuntimeError>{
                err(detail::runtimeError(status, "ProgramInvocation.setAutodiffOptions"))};
        return Result<void, RuntimeError>{ok()};
    }

    [[nodiscard]] Result<void, RuntimeError> execute(bool retainPullback = true) noexcept {
        if (!handle_)
            return Result<void, RuntimeError>{err(detail::invalidArgument("ProgramInvocation.execute"))};
        const VernonStatus status = vernonRuntimeProgramInvocationExecute(handle_, retainPullback, nullptr);
        if (status != VERNON_STATUS_OK)
            return Result<void, RuntimeError>{err(detail::runtimeError(status, "ProgramInvocation.execute"))};
        return Result<void, RuntimeError>{ok()};
    }

    [[nodiscard]] Result<Pullback, RuntimeError> commit() noexcept {
        if (!handle_)
            return Result<Pullback, RuntimeError>{err(detail::invalidArgument("ProgramInvocation.commit"))};
        VernonPullback *pullback = nullptr;
        const VernonStatus status = vernonRuntimeProgramInvocationCommit(handle_, &pullback);
        if (status != VERNON_STATUS_OK)
            return Result<Pullback, RuntimeError>{err(detail::runtimeError(status, "ProgramInvocation.commit"))};
        return Result<Pullback, RuntimeError>{ok(Pullback(pullback))};
    }

    [[nodiscard]] Result<Pullback, RuntimeError> pullback(const ProgramNodeHandle &node) noexcept {
        if (!handle_)
            return Result<Pullback, RuntimeError>{err(detail::invalidArgument("ProgramInvocation.pullback"))};
        if (!graphOwner_ || graphOwner_ != node.owner_)
            return Result<Pullback, RuntimeError>{err(detail::invalidArgument("ProgramInvocation.pullback"))};
        VernonPullback *pullback = nullptr;
        const VernonStatus status = vernonRuntimeProgramInvocationGetNodePullback(handle_, node.id(), &pullback);
        if (status != VERNON_STATUS_OK)
            return Result<Pullback, RuntimeError>{err(detail::runtimeError(status, "ProgramInvocation.pullback"))};
        return Result<Pullback, RuntimeError>{ok(Pullback(pullback))};
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
    [[nodiscard]] static Result<ProgramInstance, RuntimeError> create(const ProgramExecutable &executable) noexcept {
        if (!executable.handle_)
            return Result<ProgramInstance, RuntimeError>{err(detail::invalidArgument("ProgramInstance.create"))};
        VernonProgramInstance *handle = nullptr;
        const VernonRuntimeOperationStatus status =
            vernonRuntimeProgramInstanceCreateResult(executable.handle_.get(), &handle);
        if (status != VERNON_RUNTIME_OPERATION_OK)
            return Result<ProgramInstance, RuntimeError>{err(detail::runtimeError(status, "ProgramInstance.create"))};
        return Result<ProgramInstance, RuntimeError>{
            ok(ProgramInstance(executable.handle_, executable.graphOwner_, handle))};
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

    [[nodiscard]] Result<ProgramInvocation, RuntimeError> begin() noexcept {
        if (!handle_)
            return Result<ProgramInvocation, RuntimeError>{err(detail::invalidArgument("ProgramInstance.begin"))};
        VernonProgramInvocation *invocation = nullptr;
        const VernonRuntimeOperationStatus status =
            vernonRuntimeProgramInstanceBeginInvocationResult(handle_, &invocation);
        if (status != VERNON_RUNTIME_OPERATION_OK)
            return Result<ProgramInvocation, RuntimeError>{err(detail::runtimeError(status, "ProgramInstance.begin"))};
        return Result<ProgramInvocation, RuntimeError>{ok(ProgramInvocation(invocation, graphOwner_))};
    }

    [[nodiscard]] Result<VernonProgramBindingTelemetry, RuntimeError> telemetry() const noexcept {
        if (!handle_)
            return Result<VernonProgramBindingTelemetry, RuntimeError>{
                err(detail::invalidArgument("ProgramInstance.telemetry"))};
        VernonProgramBindingTelemetry result{};
        result.struct_size = sizeof(result);
        const VernonStatus status = vernonRuntimeProgramInstanceGetTelemetry(handle_, &result);
        if (status != VERNON_STATUS_OK)
            return Result<VernonProgramBindingTelemetry, RuntimeError>{
                err(detail::runtimeError(status, "ProgramInstance.telemetry"))};
        return Result<VernonProgramBindingTelemetry, RuntimeError>{ok(result)};
    }

    VernonProgramInstance *get() const noexcept { return handle_; }

private:
    ProgramInstance(std::shared_ptr<VernonProgramExecutable> executable,
                    std::shared_ptr<detail::ProgramGraphState> graphOwner, VernonProgramInstance *handle)
        : executable_(std::move(executable)), graphOwner_(std::move(graphOwner)), handle_(handle) {}

    std::shared_ptr<VernonProgramExecutable> executable_;
    std::shared_ptr<detail::ProgramGraphState> graphOwner_;
    VernonProgramInstance *handle_{};
};

} // namespace vernon::runtime

#endif
