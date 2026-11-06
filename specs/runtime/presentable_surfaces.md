# Presentable surfaces

Status: implemented on Vulkan (Windows), DirectX 12, OpenGL and OpenGL ES.
Metal is implemented and unverified on hardware.

A presentable surface lets a linked schedule draw into an operating-system
window. The HAL owns the surface and its images; the host owns the window
and supplies its native handle; a schedule sees each acquired image as an
ordinary image argument. The language, the compiler and Program artifacts do
not change.

## 1. Why the HAL owns presentation

A host that renders with Programs needs its final image on screen. Every
production graphics abstraction puts presentation next to the device and
queue that rendered the image: wgpu `Surface`, Dawn `wgpu::Surface`, Unreal's
RHI viewport (`RHICreateViewport`, with the back buffer imported into RDG),
Frostbite and Unity's imported back buffer, and Vulkan itself, where the
swapchain is a device extension. Presentation is ordered and synchronized on
the same queue as the draws, and only the layer that submits those draws can
do that without exporting its synchronization primitives.

Rejected alternatives:

- **The host presents a Runtime image.** The host would copy each frame from
  a Runtime image into its own back buffer, an extra full-screen copy per
  window per frame, and it would have to wait for Runtime's work through a
  fence Runtime does not export. On OpenGL the host could share the context
  and order the copy implicitly; no other backend has that escape.
- **Presentation as a Program construct.** Presenting is a queue operation on
  a resource, not a computation. A `present` node would tie the Program's
  meaning to a window, and a Program could no longer run headless or in a
  test.
- **The HAL creates windows or links a window-system library.** It would
  duplicate SDL or the host's toolkit and fix one windowing policy. The
  OpenGL context is already host-owned (`design.md` section 10); surfaces
  follow the same rule.

## 2. Model

```text
host window  --native handle-->  Surface (HAL)  --configure-->  surface images
                                    |
                     acquire  ----> SurfaceImage {image, view}   valid until present
                                    |
   linked schedules write the view as a color attachment, as any image argument
                                    |
                     present  ----> ordered after every submission so far
```

- A surface belongs to one device and one window. A multi-window host
  creates one surface per window, all on the same device.
- A surface has at most one acquired image at a time. Acquire after acquire,
  present without acquire, and configure or destroy while an image is
  acquired fail with `VERNON_HAL_INVALID_ARGUMENT`.
- The acquired image is a 2D image with one mip level, one layer and one
  sample, in the configured format, extent and usage, with a view of all of
  it. The surface owns both, so destroying either fails with
  `VERNON_HAL_INVALID_ARGUMENT`. They are valid from acquire until present;
  configuring or destroying the surface replaces or destroys them after the
  submissions that use them, as for any destroyed image
  (`image_resources.md` section 7).
- A surface is a ring of `image_count` HAL images. Acquire hands out the
  next one; present copies it to the window (section 5). So an acquired
  image is a normal image whose descriptor, views, hazards and readback are
  those of any other, on every backend.
- Its contents are undefined at acquire, because the previous frame's
  contents are not guaranteed on any backend. Until a submitted schedule
  writes the image, that is a render scope that loads it with `CLEAR` or
  `DISCARD`, or a host write does, a submission that preserves it or reads
  it fails, and so does a host read (`design.md` section 10). Writes count at
  submission: a recording may read the image after its own earlier write,
  and another may read it only once the writing one is submitted.

## 3. HAL API

The declarations are in `VernonHal.h`: `VernonHalSurfaceDescriptor`,
`VernonHalSurfaceCapabilities`, `VernonHalSurfaceConfiguration`,
`VernonHalSurfaceImage`, `VernonHalPresentMode` (`FIFO`, which every surface
supports, `MAILBOX`, `IMMEDIATE`), and `vernonHalCreateSurface`,
`vernonHalSurfaceGetCapabilities`, `vernonHalSurfaceConfigure`,
`vernonHalSurfaceAcquire`, `vernonHalSurfacePresent` and
`vernonHalDestroySurface`. The descriptor's `native` member is the one the
device's backend reads: a window on Vulkan and D3D12, a layer on Metal,
callbacks on OpenGL and OpenGL ES (section 5), copied at creation.

- A device that cannot present fails surface creation with
  `VERNON_HAL_UNSUPPORTED`. There is no capability bit for it, because no
  host queries capabilities before creating the resource that needs them.
- Configure fails with `VERNON_HAL_INVALID_ARGUMENT` for a zero extent, a
  usage without `COLOR_ATTACHMENT` or an unknown present mode, and with
  `VERNON_HAL_UNSUPPORTED` for a format, usage, present mode or image count
  outside the capabilities. `hal/capi.cpp` checks both for every backend. A
  minimized window has a zero extent; the host skips the window's frame
  until it resizes. A failed configure keeps the previous images.
- Acquire may block until an image is available, which is how `FIFO` paces
  a frame. With `N` calls in flight, which the application paces on their
  timeline points (`architecture_v2.md` section 8), `image_count` of at
  least `N + 1` keeps acquire from waiting for the GPU as well.
- Acquire returns `VERNON_HAL_SURFACE_OUT_OF_DATE` once the window no
  longer matches the configuration; the host configures again and retries.
  `VERNON_HAL_SURFACE_LOST` means the window is gone. A failed acquire
  leaves no image acquired; a failed present still releases the image.
- `VernonHalSurfaceImage::suboptimal` reports an image that still presents
  but whose surface would match the window better configured again.

## 4. Synchronization

The HAL orders presentation the way it orders everything else, by queue
submission:

- Present runs after every submission made before it, so the schedules that
  wrote the image finish first, without a host wait.
- Neither present nor acquire waits on the host for completion; a host that
  wants to bound latency waits on its calls' timeline points.

## 5. Backend mappings

| Backend | Native input | Present |
|---|---|---|
| Vulkan | `HWND` | acquire a swapchain image, blit the surface image into it, `vkQueuePresentKHR` |
| OpenGL, OpenGL ES | `VernonOpenGLSurfaceCallbacks` | make the window current, blit with rows reversed, swap |
| DirectX 12 | `HWND` | flip-discard `IDXGISwapChain3` on the device's direct queue |
| Metal | `CAMetalLayer` | `nextDrawable`, draw the surface image into it, `presentDrawable:` after the last submission |

### Vulkan

The Vulkan device enables `VK_KHR_surface` and the platform surface
extension when the loader offers them, and creates the `VkSurfaceKHR` from
the window itself (`hal/vulkan/vulkan_platform_surface.cpp`, Win32 for now),
as wgpu and Dawn do. Exposing the instance for the host to create the
surface was rejected: the host would have to name instance extensions at
device creation and link the platform's Vulkan surface code.

The swapchain is the surface's private target. Present acquires a swapchain
image with a semaphore, transitions the surface image for transfer, blits it
into the swapchain image, scaling with a linear filter when the swapchain's
extent differs from the configured one, and presents after a semaphore the
copy signals. The swapchain format follows what the surface reports, which
need not be the configured format, because the blit converts. A frame whose
swapchain is out of date or whose window is minimized is dropped, and the
surface goes stale: the next acquire reports `SURFACE_OUT_OF_DATE`, and the
next present rebuilds the swapchain. Rendering into swapchain images
directly was rejected: their formats and counts are the driver's, so a
pipeline could not be cooked for them, and their acquire would block inside
a call.

### OpenGL

`VernonOpenGLSurfaceCallbacks` (`VernonOpenGLContext.h`) supplies
`make_current` (the device's context on this window), `swap_interval`
(`SDL_GL_SetSwapInterval`, with the window current), `swap_buffers`
(`SDL_GL_SwapWindow`) and `drawable_size` (`SDL_GetWindowSizeInPixels`).
Present makes the device's context current on the window, blits the
surface image into framebuffer 0 with its rows reversed, swaps, and makes
the device's own target current again.

The rows are reversed because of the vertex convention. Every GL vertex
stage negates y (`design.md` section 10), so a Program's row 0 is the
image's top row on every backend, while the default framebuffer counts rows
from the bottom of the window. Drawing into it directly would show every
frame upside down, and the negation is compiled into the cooked shaders, so
it cannot change per target.

Costs and limits of the fallback tier:

- one full-window blit per present;
- every surface's window must accept the device's own context
  (`SDL_GL_MakeCurrent(window, device_context)`), because framebuffer and
  vertex-array objects are not shared between contexts. A host that creates
  one context per window cannot present those windows through one device;
- `MAILBOX` and `IMMEDIATE` map to a swap interval of 0, `FIFO` to 1, and
  the driver alone paces the frame, so `suboptimal` is never reported;
- the window does not resize the images, so acquire reports
  `SURFACE_OUT_OF_DATE` whenever `drawable_size` differs from the configured
  extent.

## 6. Formats and color space

Surfaces present nonlinear sRGB. Every surface offers `rgba8_unorm`, with
`COLOR_ATTACHMENT` and `TRANSFER_SOURCE` usage. Sampled and storage usage
are withheld so that every read of an acquired image is a preserving render
scope or a read, which the undefined-contents rule of section 2 checks. A
host whose display pass writes display-encoded values configures the
`unorm` format; the `srgb` formats encode on store. HDR surfaces
(`rgba16_float` scRGB, `rgb10a2` PQ) are out of scope.

A graphics pipeline's color target formats are fixed when it is cooked, and
every backend requires the pipeline's format to match the attachment's
exactly. A Program that draws into a surface therefore needs one cooked
pipeline per format the host may configure. A pipeline's color target
formats are a static specialization argument (`language/contract.md`
section 6.6.1): one declaration cooks a specialization per format, and the
host selects the one matching the configured format. Cooking a separate
asset per format was rejected because it duplicates the declaration.

Metal's `CAMetalLayer` has no RGBA8, so the Metal surface keeps RGBA8 ring
images and present draws the presented image into the layer's BGRA8
drawable with a fullscreen-triangle pass, which also converts to the
drawable's encoding. `bgra8_unorm` and `bgra8_srgb` are ordinary image
formats, but a Metal surface configured as BGRA8 was rejected: every schedule
and cooked asset would then need a second variant per surface format for one
backend's swapchain.

## 7. Runtime

Runtime gains no surface concept. An acquired view is an ordinary image
argument of a linked schedule:

- the schedule describes it from the HAL view registry, as for any view;
- the check that a color attachment's format equals the Program's cooked
  target format applies unchanged;
- the undefined-contents rule of section 2 is checked by the HAL at
  submission, so Runtime does not learn what a surface is.

## 8. Engine consumer (non-normative)

The Vernon Engine uses surfaces for the editor and for ImGui with
multi-viewport:

```text
per frame
  for each window: acquire (skip minimized, reconfigure on OUT_OF_DATE)
  RenderGraph: the main window's image imported as the display target;
               the scene renders into Engine textures, the ImGui pass draws
               the UI into the window image, sampling Engine textures by
               ImTextureID index
  for each secondary viewport: the ImGui pass draws that viewport's draw data
                               into its own window image
  for each window: present
```

The RenderGraph imports a surface image as a frame resource whose contents
are undefined at the start of the frame, which is the rule the graph already
applies to transient textures.

## 9. Open questions

1. **Python.** Whether `RuntimeSession` exposes surfaces, or only the C++
   host path does.
2. **Vulkan surfaces off Windows.** `vulkan_platform_surface.cpp` creates
   Win32 surfaces only; Linux and macOS (MoltenVK) need their platform
   surface extensions and native handles in the descriptor.

## 10. Acceptance

`source/tests/runtime/hal_resources_test.cpp` covers, on every device that
creates surfaces:

- create, configure, acquire and present, with the acquired image and view
  in the configured format and extent; every other device fails creation
  with `VERNON_HAL_UNSUPPORTED`;
- a read of a freshly acquired image fails, and after a host write the image
  reads back exactly before present;
- two surfaces on one device present their own images (Vulkan);
- rejected before native work: acquire twice; present without acquire;
  configure, destroy, or destroying the image or view while an image is
  acquired; a zero extent, a usage without `COLOR_ATTACHMENT` or an unknown
  mode (`INVALID_ARGUMENT`); a format, usage or image count outside the
  capabilities (`UNSUPPORTED`);
- resize: acquire reports `SURFACE_OUT_OF_DATE`, and configure followed by
  acquire succeeds at the new extent.

The Engine's UI renderer draws into surface images through linked
schedules on Vulkan and both OpenGL tiers. No backend renders into the
OpenGL default framebuffer, and no host-side copy of a Runtime image to a
back buffer remains.
