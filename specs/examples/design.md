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
