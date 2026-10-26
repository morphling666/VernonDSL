# Example Design Notes

Status: non-normative example implementation notes.

## Terrain showcase

The terrain is a full-screen ray-marched graphics workload. A repeat-wrapped
512x512 noise texture drives a seven-octave ridged height field. Finite-
difference normals feed snow, rock, and grass material bands, while secondary
marches provide soft shadows and ambient occlusion. Atmospheric fog, sky and
sun shading, display mapping, and a host-driven camera complete the image.
Smoke and showoff presets vary primary, shadow, and AO iteration budgets.

The implementation is adapted from the MIT-licensed
[kevinroast/webglshaders](https://github.com/kevinroast/webglshaders),
Copyright (c) 2015 Kevin Roast. It preserves the original terrain, lighting,
fog, and post-processing structure while expressing texture sampling and
bounded control flow through VernonDSL.

## Mandelbulb showcase

The Mandelbulb is a full-screen ray marcher using the power-eight spherical
distance estimator. Surface shading combines finite-difference normals, soft
ray-marched shadows, an iteration-derived ambient-occlusion term, orbit-trap
coloring, specular highlights, and distance fog. A host-driven orbit gives
deterministic headless frames and a continuous interactive view. The smoke and
showoff presets vary ray, fractal, and shadow iteration budgets without
changing shader topology.

The implementation is adapted from the MIT-licensed
[matt-k-wong/WebGL-Mandelbulb](https://github.com/matt-k-wong/WebGL-Mandelbulb),
Copyright (c) 2026.

## Advanced example: dynamic terrain erosion

- Algorithm: Eulerian height-field hydraulic erosion with ping-pong height,
  water, and sediment grids.
- Reference: Musgrave, Kolb, and Mace, *The Synthesis and Rendering of Eroded
  Fractal Terrains*, SIGGRAPH 1989.
- Parameters: two half steps per frame, explicit four-direction outflow,
  sediment advection, capacity-limited erosion/deposition, and edge drainage.
- Rationale: a first dispatch writes one bounded outflow vector per cell; a
  second gathers the opposite components from four neighbors. Water and
  sediment are therefore conservative apart from explicit rain, evaporation,
  and boundary drainage, while retaining one writer per cell and requiring no
  atomics. Runoff appearance is derived only from simulated water. Terrain
  shape uses deterministic domain-warped FBM instead of directional sine
  layers. A CC0 rock normal map and diffuse luminance are packed into one RGBA
  texture and sampled triplanarly to fit the four-resource graphics layout.

## Qwen3 decoder

- Model: Qwen3 dense decoder (Qwen Team, *Qwen3 Technical Report*, 2025):
  RMSNorm pre-norm blocks, grouped-query attention with per-head RMSNorm on
  queries and keys, rotate-half RoPE (theta 1e6), and a SwiGLU MLP.
- Precision: bf16 checkpoints are widened to f32 on load, because GPU f16
  storage is not yet supported; 0.6B parameters need 3 GB of f32 weights.
- One `vd.Module` invocation decodes one token. Layer weights are stacked
  along a leading layer axis, and every kernel takes the layer index, because
  Module forwards accept only storages and scalars, not lists of storages.
  The unrolled Python layer loop makes each index a constant.
- The position and token live in a two-element `step` storage that the host
  rewrites before each invocation. Scalar forward arguments are captured at
  the first invocation, so they cannot carry per-token values.
- Write ownership: matrix-vector products use one workgroup per output row,
  with the lane-0 leader writing the reduced value. Grids that cover the whole
  context exit early beyond the current position, so dispatch shapes never
  change. The KV cache is appended with `atomic_add` into a cache zeroed per
  conversation, because a store indexed by the runtime position cannot be
  proven lane-exclusive.
- Attention values use a single-pass online softmax (Milakov and
  Gimelshein, *Online normalizer calculation for softmax*, 2018) so scores
  are read once. The prompt is fed one token at a time through the decode
  step. A batched prefill would need matrix-matrix kernels.
- `model.py` keeps a NumPy reference decoder that follows the Hugging Face
  implementation; `generate.py --check` compares logits step by step.

## Advanced example: dynamic ocean

The ocean showcase uses a damped finite-difference wave equation over a dense
height/velocity grid. Two half-step dispatches, A-to-B and B-to-A, run every
frame so the graph topology and final resource identities never change.
Procedural crossing and radial forcing keep the surface animated without host
uploads. Eight directional Gerstner components form a broad, hand-tuned
directional spectrum: two long swells establish silhouette, four crossing
mid-frequency waves break parallel bands, and two low-amplitude short waves
shape crests. The simulated field is deliberately lower amplitude so it
contributes irregular motion without turning the surface into soft blobs.

A separate compute pass gathers adjacent heights to generate positions,
analytic normals, and a curvature-gated crest mask. Simulation state remains
in TensorStorage so its typed, strided representation is shared by the
simulation and geometry passes; storage Textures remain available when
image-format access is the appropriate representation.

The water fragment stage samples one CC0 normal map twice in world space at
37-degree and 113-degree orientations, different scales, and different
velocities. This adds capillary detail without increasing the simulation grid.
The Poly Haven sun direction is
matched by the direct light, ACES performs display mapping, and distance fog
plus an ocean extent larger than the camera radius hides the finite grid edge.
