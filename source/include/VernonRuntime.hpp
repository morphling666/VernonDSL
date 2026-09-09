#ifndef VERNON_RUNTIME_HPP
#define VERNON_RUNTIME_HPP

#include "VernonRuntime.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>

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
        if (vernonProgramPullbackApplyWithOptions(handle_, arguments, argumentCount, &options) != VERNON_STATUS_OK)
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
    ProgramExecutable resolve(VernonFeatureSetView features = {nullptr, 0}) const;

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

struct ProgramNodeGraphics {
    VernonProgramNodeGraphicsToken token{sizeof(VernonProgramNodeGraphicsToken), 0, 0, 0};
};

class ProgramGraph;
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
    ProgramNodeGraphics graphics(std::string_view name) const;
    ProgramNodeGraphics graphics(size_t index) const;
    size_t graphicsCount() const noexcept {
        return owner_ ? vernonRuntimeProgramGraphGetGraphicsNodeCount(owner_->handle, id_) : 0;
    }
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
    explicit ProgramExecutable(VernonProgramExecutable *handle)
        : handle_(handle, vernonRuntimeProgramExecutableDestroy) {}

    std::shared_ptr<VernonProgramExecutable> handle_;
    friend class ProgramAsset;
    friend class ProgramInstance;
    friend class ProgramGraph;
};

inline ProgramExecutable ProgramAsset::resolve(VernonFeatureSetView features) const {
    VernonProgramExecutable *executable = vernonRuntimeResolveProgram(handle_.get(), features);
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

    ProgramNode add(const ProgramAsset &asset) {
        VernonProgramNodeId id{};
        if (!state_ || !state_->handle ||
            vernonRuntimeProgramGraphAddProgram(state_->handle, asset.get(), &id) != VERNON_STATUS_OK)
            throw std::runtime_error("failed to add ProgramGraph node");
        return ProgramNode(state_, id);
    }

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

    ProgramExecutable compile(VernonFeatureSetView features = {nullptr, 0}) {
        VernonProgramExecutable *executable =
            vernonRuntimeResolveProgramGraph(state_ ? state_->handle : nullptr, features);
        if (!executable)
            throw std::runtime_error("failed to compile ProgramGraph");
        return ProgramExecutable(executable);
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

inline ProgramNodeGraphics ProgramNode::graphics(std::string_view name) const {
    if (!owner_)
        throw std::logic_error("ProgramGraph node is empty");
    ProgramNodeGraphics result;
    const VernonStringView view{name.data(), name.size()};
    if (vernonRuntimeProgramGraphFindGraphicsNode(owner_->handle, id_, view, &result.token) != VERNON_STATUS_OK)
        throw std::runtime_error("ProgramGraph graphics node was not found");
    return result;
}

inline ProgramNodeGraphics ProgramNode::graphics(size_t index) const {
    if (!owner_)
        throw std::logic_error("ProgramGraph node is empty");
    ProgramNodeGraphics result;
    if (vernonRuntimeProgramGraphGetGraphicsNodeByIndex(owner_->handle, id_, index, &result.token) != VERNON_STATUS_OK)
        throw std::runtime_error("ProgramGraph graphics node index is out of range");
    return result;
}

class ProgramInvocation {
public:
    class NodeFrame {
    public:
        NodeFrame &bind(ProgramNodeBinding binding, const VernonProgramArgument &argument,
                        const VernonProgramResourceLease *lease = nullptr, uint64_t uploadBytes = 0,
                        uint64_t uploadRanges = 0) {
            if (binding.token.node != node_)
                throw std::invalid_argument("ProgramGraph binding belongs to another node");
            invocation_->bind(binding, argument, lease, uploadBytes, uploadRanges);
            return *this;
        }

        NodeFrame &bind(ProgramNodeGraphics graphics, const VernonRenderPass &renderPass, const VernonDrawCommand &draw,
                        const VernonDynamicState &dynamicState,
                        const VernonProgramResourceLease *renderPassLeases = nullptr, size_t renderPassLeaseCount = 0,
                        const VernonProgramResourceLease *drawLease = nullptr) {
            if (graphics.token.node != node_)
                throw std::invalid_argument("ProgramGraph graphics token belongs to another node");
            invocation_->bind(graphics, renderPass, draw, dynamicState, renderPassLeases, renderPassLeaseCount,
                              drawLease);
            return *this;
        }

    private:
        NodeFrame(ProgramInvocation &invocation, VernonProgramNodeId node) : invocation_(&invocation), node_(node) {}
        ProgramInvocation *invocation_{};
        VernonProgramNodeId node_{};
        friend class ProgramInvocation;
    };

    ProgramInvocation() = default;
    explicit ProgramInvocation(VernonProgramInvocation *handle) : handle_(handle) {}
    ProgramInvocation(const ProgramInvocation &) = delete;
    ProgramInvocation &operator=(const ProgramInvocation &) = delete;
    ProgramInvocation(ProgramInvocation &&other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
    ProgramInvocation &operator=(ProgramInvocation &&other) noexcept {
        if (this != &other) {
            vernonRuntimeProgramInvocationDestroy(handle_);
            handle_ = std::exchange(other.handle_, nullptr);
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

    ProgramInvocation &bind(ProgramNodeBinding node, const VernonProgramArgument &argument,
                            const VernonProgramResourceLease *lease = nullptr, uint64_t uploadBytes = 0,
                            uint64_t uploadRanges = 0) {
        if (!handle_ || vernonRuntimeProgramInvocationBindNode(handle_, &node.token, &argument, lease, uploadBytes,
                                                               uploadRanges) != VERNON_STATUS_OK)
            throw std::runtime_error("failed to bind ProgramGraph node");
        return *this;
    }

    ProgramInvocation &bind(ProgramGraphStorage storage, const VernonProgramArgument &argument,
                            const VernonProgramResourceLease *lease = nullptr, uint64_t uploadBytes = 0,
                            uint64_t uploadRanges = 0) {
        if (!handle_ || vernonRuntimeProgramInvocationBindGraphStorage(handle_, &storage.storage, &argument, lease,
                                                                       uploadBytes, uploadRanges) != VERNON_STATUS_OK)
            throw std::runtime_error("failed to bind ProgramGraph Storage");
        return *this;
    }

    ProgramInvocation &bind(ProgramNodeGraphics node, const VernonRenderPass &renderPass, const VernonDrawCommand &draw,
                            const VernonDynamicState &dynamicState,
                            const VernonProgramResourceLease *renderPassLeases = nullptr,
                            size_t renderPassLeaseCount = 0, const VernonProgramResourceLease *drawLease = nullptr) {
        if (!handle_ || vernonRuntimeProgramInvocationBindNodeGraphics(handle_, &node.token, &renderPass,
                                                                       renderPassLeases, renderPassLeaseCount, &draw,
                                                                       drawLease, &dynamicState) != VERNON_STATUS_OK)
            throw std::runtime_error("failed to bind ProgramGraph graphics controls");
        return *this;
    }

    NodeFrame node(const ProgramNodeHandle &node) { return NodeFrame(*this, node.id()); }

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

    Pullback forward(bool retainPullback = true) {
        if (!handle_)
            throw std::logic_error("Program invocation is empty");
        VernonPullback *pullback = nullptr;
        if (vernonRuntimeProgramInvocationForward(handle_, retainPullback ? &pullback : nullptr) != VERNON_STATUS_OK)
            throw std::runtime_error("Program invocation failed");
        return Pullback(pullback);
    }

    void rollback() {
        if (handle_)
            vernonRuntimeProgramInvocationRollback(handle_);
    }

    VernonProgramInvocation *get() const noexcept { return handle_; }

private:
    VernonProgramInvocation *handle_{};
};

class ProgramInstance {
public:
    explicit ProgramInstance(ProgramExecutable &executable)
        : executable_(executable.handle_), handle_(vernonRuntimeProgramInstanceCreate(executable_.get())) {
        if (!handle_)
            throw std::runtime_error("failed to create Program instance");
    }
    ProgramInstance(const ProgramInstance &) = delete;
    ProgramInstance &operator=(const ProgramInstance &) = delete;
    ProgramInstance(ProgramInstance &&other) noexcept
        : executable_(std::move(other.executable_)), handle_(std::exchange(other.handle_, nullptr)) {}
    ProgramInstance &operator=(ProgramInstance &&other) noexcept {
        if (this != &other) {
            vernonRuntimeProgramInstanceDestroy(handle_);
            executable_ = std::move(other.executable_);
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }
    ~ProgramInstance() { vernonRuntimeProgramInstanceDestroy(handle_); }

    ProgramInvocation begin() {
        VernonProgramInvocation *invocation = vernonRuntimeProgramInstanceBeginInvocation(handle_);
        if (!invocation)
            throw std::runtime_error("failed to begin Program invocation");
        return ProgramInvocation(invocation);
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
    VernonProgramInstance *handle_{};
};

} // namespace vernon::runtime

#endif
