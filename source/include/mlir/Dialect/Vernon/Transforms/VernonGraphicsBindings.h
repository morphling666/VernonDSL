#ifndef MLIR_DIALECT_VERNON_TRANSFORMS_VERNONGRAPHICSBINDINGS_H
#define MLIR_DIALECT_VERNON_TRANSFORMS_VERNONGRAPHICSBINDINGS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mlir::vernon {

/// Module attribute holding the pipeline binding layout computed by
/// linkGraphicsBindings for this module. Absent for a module compiled on its
/// own, which is then planned as a one-module pipeline.
inline constexpr llvm::StringLiteral kGraphicsBindingLayoutAttrName = "vernon.graphics_binding_layout";

/// Marks a texture or sampler argument whose set and binding were written by
/// materializeGraphicsResourceBindings rather than authored.
inline constexpr llvm::StringLiteral kGeneratedBindingAttrName = "vernon.generated_binding";

/// Entry argument attribute naming the binding-group occurrence (its entry
/// parameter path) that a flattened leaf or its implicit sampler belongs to.
inline constexpr llvm::StringLiteral kBindingGroupAttrName = "vernon.binding_group";

/// One generated set-0 binding request of a graphics entry. `argument` is
/// empty for the entry's TensorView metadata carrier, which exists only after
/// target preparation. `group` is the binding-group occurrence, or empty.
struct GraphicsBindingRequest {
    std::string entry;
    std::optional<unsigned> argument;
    std::string sourceName;
    std::string kind;
    std::string typeKey;
    uint32_t count{};
    std::string group;
};

struct GraphicsAuthoredBinding {
    std::string entry;
    unsigned argument{};
    std::string sourceName;
    int64_t set{};
    int64_t binding{};
};

struct GraphicsBindingRequests {
    std::string stage;
    std::vector<GraphicsBindingRequest> requests;
    std::vector<GraphicsAuthoredBinding> authored;
};

/// Set-0 bindings generated for graphics entry arguments. Reflection and SPIR-V
/// lowering both read bindings from this plan; neither may assign its own.
struct GraphicsBindingPlan {
    /// (entry symbol, argument index) -> generated bindings: one for a uniform
    /// spilled to a buffer, a texture, or a sampler without an authored
    /// binding; one per storage leaf for a TensorView; one for the TensorView
    /// metadata carrier.
    std::map<std::pair<std::string, unsigned>, SmallVector<uint32_t>> arguments;

    const SmallVector<uint32_t> *find(StringRef entry, unsigned argument) const;
};

/// Per-entry generated bindings of one module of a linked pipeline, in the
/// order of that module's requests.
struct GraphicsBindingLayout {
    std::vector<std::pair<GraphicsBindingRequest, SmallVector<uint32_t>>> assignments;
};

FailureOr<GraphicsBindingRequests> collectGraphicsBindingRequests(ModuleOp module);

/// Assigns one set-0 layout to the modules of a pipeline, in module order.
/// Authored bindings stay fixed and may be shared across modules only by
/// endpoints with the same pipeline parameter name. A generated request reuses
/// the bindings of an earlier module's request with the same parameter name,
/// kind, type, and count; every other request takes the lowest free bindings.
/// Each binding-group occurrence takes one consecutive range covering the
/// union of its requests in every module, so both stages share it.
FailureOr<std::vector<GraphicsBindingLayout>> linkGraphicsBindings(ArrayRef<GraphicsBindingRequests> modules,
                                                                   std::string &error);

/// Stores a linked layout on the module it was computed for.
void attachGraphicsBindingLayout(ModuleOp module, const GraphicsBindingLayout &layout);

FailureOr<GraphicsBindingPlan> planGraphicsBindings(ModuleOp module);

/// Writes planned set-0 bindings onto graphics texture and sampler arguments
/// that have none, so code that reads authored bindings sees them. Run once
/// per module before reflection and lowering.
LogicalResult materializeGraphicsResourceBindings(ModuleOp module);

} // namespace mlir::vernon

#endif
