#ifndef VERNON_RUNTIME_HPP
#define VERNON_RUNTIME_HPP

#include "VernonError.hpp"
#include "VernonRuntime.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <new>
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
    // `identity` is the canonical type identity the asset was cooked with, `<module>:<Class>` or a scalar name.
    [[nodiscard]] Result<void, RuntimeError> setType(std::string name, std::string identity) noexcept {
        if (identity.empty())
            return Result<void, RuntimeError>{err(detail::invalidArgument("ProgramVariant.setType"))};
        return set(std::move(name), VERNON_PROGRAM_SPECIALIZATION_TYPE, std::move(identity));
    }
    [[nodiscard]] Result<void, RuntimeError> setFormat(std::string name, VernonTextureFormat format) noexcept {
        return set(std::move(name), VERNON_PROGRAM_SPECIALIZATION_FORMAT, format);
    }

private:
    using Value = std::variant<bool, int32_t, uint32_t, float, double, std::string, VernonTextureFormat>;
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
                        else if constexpr (std::is_same_v<T, double>)
                            specialization.value.f64_value = value;
                        else if constexpr (std::is_same_v<T, VernonTextureFormat>)
                            specialization.value.format_value = value;
                        else
                            specialization.value.type_value = {value.data(), value.size()};
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

    // An empty `label` names the child's Nodes after its bundle id.
    [[nodiscard]] Result<ProgramNode, RuntimeError> add(const ProgramAsset &asset,
                                                        std::string_view label = {}) noexcept {
        return add(asset, nullptr, label);
    }

    [[nodiscard]] Result<ProgramNode, RuntimeError> add(const ProgramAsset &asset, const ProgramVariant &variant,
                                                        std::string_view label = {}) noexcept {
        auto native = variant.native("ProgramGraph.add");
        if (native.isErr())
            return Result<ProgramNode, RuntimeError>{err(std::move(native).error())};
        const std::vector<VernonProgramSpecialization> specializations = std::move(native).value();
        const VernonProgramVariantSelector selector{
            sizeof(VernonProgramVariantSelector), specializations.data(), specializations.size(), {}};
        return add(asset, &selector, label);
    }

private:
    explicit ProgramGraph(std::shared_ptr<detail::ProgramGraphState> state) : state_(std::move(state)) {}

    Result<ProgramNode, RuntimeError> add(const ProgramAsset &asset, const VernonProgramVariantSelector *selector,
                                          std::string_view label) noexcept {
        VernonProgramNodeId id{};
        if (!state_ || !state_->handle)
            return Result<ProgramNode, RuntimeError>{err(detail::invalidArgument("ProgramGraph.add"))};
        const VernonStatus status = vernonRuntimeProgramGraphAddProgram(state_->handle, asset.get(), selector,
                                                                        {label.data(), label.size()}, &id);
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

} // namespace vernon::runtime

#endif
