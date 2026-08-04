#ifndef VERNON_AUTODIFF_GRAPH_H
#define VERNON_AUTODIFF_GRAPH_H

#include "VernonRuntime.h"

#include <cstdint>
#include <memory>
#include <string>

namespace vernon::runtime {

class CompiledAutodiffGraph;

struct AutodiffGraphNode {
    uint32_t index{UINT32_MAX};
    uint64_t graphIdentity{};
};

class AutodiffGraphPullback {
public:
    class Impl;

    ~AutodiffGraphPullback();
    AutodiffGraphPullback(AutodiffGraphPullback &&) noexcept;
    AutodiffGraphPullback &operator=(AutodiffGraphPullback &&) noexcept;
    AutodiffGraphPullback(const AutodiffGraphPullback &) = delete;
    AutodiffGraphPullback &operator=(const AutodiffGraphPullback &) = delete;

    VernonStatus apply(const VernonAdValueSet *cotangents, VernonAdValueSet &gradients);
    VernonRhiCommandEncoderStats forwardStats() const;
    VernonRhiCommandEncoderStats lastBackwardStats() const;

private:
    explicit AutodiffGraphPullback(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    friend class AutodiffGraph;
    friend class CompiledAutodiffGraph;
};

class AutodiffGraph {
public:
    explicit AutodiffGraph(VernonRuntimeContext *context);
    ~AutodiffGraph();
    AutodiffGraph(AutodiffGraph &&) noexcept;
    AutodiffGraph &operator=(AutodiffGraph &&) noexcept;
    AutodiffGraph(const AutodiffGraph &) = delete;
    AutodiffGraph &operator=(const AutodiffGraph &) = delete;

    VernonStatus addNode(std::string name, VernonLoadedPipeline *pipeline, AutodiffGraphNode &node);
    VernonStatus declareInput(AutodiffGraphNode node, std::string inputPath, std::string valuePath,
                              std::string gradientPath);
    VernonStatus connect(AutodiffGraphNode source, AutodiffGraphNode destination, std::string destinationInputPath);
    VernonStatus setOutput(AutodiffGraphNode node);
    VernonStatus compile(std::unique_ptr<CompiledAutodiffGraph> &compiled);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class CompiledAutodiffGraph {
public:
    ~CompiledAutodiffGraph();
    CompiledAutodiffGraph(CompiledAutodiffGraph &&) noexcept;
    CompiledAutodiffGraph &operator=(CompiledAutodiffGraph &&) noexcept;
    CompiledAutodiffGraph(const CompiledAutodiffGraph &) = delete;
    CompiledAutodiffGraph &operator=(const CompiledAutodiffGraph &) = delete;

    VernonStatus forward(VernonLaunchSize computeGrid, VernonAdValueSet &inputs, VernonAdValueSet &outputs,
                         std::unique_ptr<AutodiffGraphPullback> &pullback) const;

private:
    class Impl;
    explicit CompiledAutodiffGraph(std::unique_ptr<Impl> impl);
    static std::unique_ptr<AutodiffGraphPullback> makePullback(std::unique_ptr<AutodiffGraphPullback::Impl> impl);
    std::unique_ptr<Impl> impl_;
    friend class AutodiffGraph;
};

} // namespace vernon::runtime

#endif
