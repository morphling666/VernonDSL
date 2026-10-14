# Fragment discard

Status: proposed (Engine extension E12, masked raster). On acceptance it
amends the contracts listed in section 6 under one `COMPILER_CONTRACT_VERSION`
cut.

A masked material keeps or drops each fragment by an opacity test. The
language has no way to drop a fragment: `vd.discard` is an attachment load
operation. This document adds one fragment statement that does.

## 1. Rejected alternatives

- **Demote-to-helper semantics.** It keeps derivatives defined after the
  drop, but OpenGL GLSL cannot express it (SPIRV-Cross rejects
  `OpDemoteToHelperInvocation` outside Vulkan GLSL), and a portable emulation
  must predicate every later side effect. Masked materials need neither: they
  sample the opacity before testing it.
- **Alpha-to-coverage or alpha test as pipeline state.** Neither exists on
  every backend, and both move material logic out of the surface code.
- **Reusing the name `vd.discard`.** It already names the attachment load
  operation.

## 2. Language

```python
@vd.fragment
def masked_fragment(...) -> None:
    opacity = vd.texture_sample(mask, sampler, uv)[channel]
    if opacity < cutoff:
        vd.discard_fragment()
```

- `vd.discard_fragment()` is a statement intrinsic returning nothing. It is
  valid only in a `@vd.fragment` entry and in helpers reachable only from
  fragment entries; elsewhere it fails with the existing stage diagnostic
  `discard_fragment() is not available in this shader stage`.
- It may appear in non-uniform control flow and any number of times.
- It terminates the invocation: no later statement runs, and the fragment
  writes no color, depth or stencil output.
- Derivatives and implicit-LOD sampling after a `discard_fragment()` that is
  reached non-uniformly within a quad are undefined, as for any derivative in
  non-uniform control flow. Sample before discarding.
- Graphics stays primal-only; `discard_fragment` never enters a derivative
  path.

## 3. Lowering

`discard_fragment()` lowers to SPIR-V `OpKill` (valid in the SPIR-V 1.3
target environment). Vulkan consumes it directly; SPIRV-Cross emits `discard`
for GLSL and HLSL and `discard_fragment()` for MSL. `OpKill` is a block
terminator, so the frontend ends the current block after it and code that
follows it in the same block is unreachable: inference types and lowers no
statement after it.

In the Vernon dialect `vernon.discard_fragment` is an ordinary operation, not
an MLIR terminator, so `scf` regions keep their `scf.yield` and helpers inline
unchanged; its verifier rejects it inside an entry whose `vernon.stage` is not
`fragment`. SPIR-V translation emits it as a selection on constant `true`
whose taken block is `OpKill`. The kill ends that block, while the enclosing
structured block still reaches its merge or loop back edge, which a bare
`OpKill` in the middle of a loop body or selection arm would remove.

## 4. Engine use (non-normative)

The Engine's masked raster Program, one per geometry kind, samples the
standardized opacity mask (texture, channel, cutoff) and discards. The forward
pass of a masked material does not discard: it tests `EQUAL` against the
depth the masked prepass wrote, which already rejects the dropped fragments.

## 5. Acceptance

On every backend with graphics support:

1. A fragment that discards every other pixel leaves those pixels' color,
   depth and stencil at their cleared values.
2. Discard inside a loop and inside a helper function behaves as in 1.
3. Statements after `discard_fragment()` in the same block do not affect
   outputs.
4. `discard_fragment()` in a vertex or compute entry fails with its
   diagnostic.

## 6. Contract changes on acceptance

- [`../language/contract.md`](../language/contract.md): the intrinsic list
  gains `discard_fragment` with the stage and derivative rules of section 2.
- [`../testing/language_feature_inventory.md`](../testing/language_feature_inventory.md):
  test IDs for acceptance 1 to 4.
