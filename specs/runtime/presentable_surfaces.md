# Presentable surfaces

Status: proposed design, under review. Phase 1 (section 6) is implemented:
the RHI API and OpenGL surfaces.

A presentable surface lets a Program draw into an operating-system window.
VernonRHI owns the swapchain; the host owns the window and supplies its native
handle; Runtime sees each acquired swapchain image as an ordinary imported
color target. The language, the compiler and Program artifacts do not change.

## 1. Why the RHI owns presentation

A host that renders with Programs needs its final image on screen. Every
production graphics abstraction puts presentation next to the device and
queue that rendered the image: wgpu `Surface`, Dawn `wgpu::Surface`, Unreal's
RHI viewport (`RHICreateViewport`, with the back buffer imported into RDG),
Frostbite and Unity's imported back buffer, and Vulkan itself, where the
swapchain is a device extension. Presentation is ordered and synchronized on
the same queue as the draws, and only the layer that submits those draws can
do that without exporting its synchronization primitives.

Rejected alternatives:

- **The host presents a Runtime texture.** The host would copy each frame from
  a Runtime image into its own back buffer, an extra full-screen copy per
  window per frame, and it would have to wait for Runtime's work through a
  fence Runtime does not export. On OpenGL the host could share the context
  and order the copy implicitly; no other backend has that escape.
- **Presentation as a Program construct.** Presenting is a queue operation on
  a resource, not a computation. A `present` node would tie the Program's
  meaning to a window, and a Program could no longer run headless or in a
  test.
- **The RHI creates windows or links a window-system library.** It would duplicate
  SDL or the host's toolkit and fix one windowing policy. The OpenGL context
  is already host-owned (`design.md` section 10); surfaces follow the same
  rule.
- **Borrowed-device presentation.** A borrowed device records into the host's
  command list, which the host executes and presents. Such a host already has
  a swapchain, so a borrowed device does not report presentation.

## 2. Model

```text
host window  --native handle-->  Surface (RHI)  --configure-->  swapchain images
                                    |
                     acquire  ----> SurfaceImage {image, view}   valid until present
                                    |
   Runtime invocations bind the view as a color target, as any imported image
                                    |
                     present  ----> queue-ordered after every submission so far
```

- A surface belongs to one device and one window. A multi-window host
  creates one surface per window, all on the same device.
- A surface has at most one acquired image at a time. Acquire after acquire,
  present without acquire, and configure or destroy while an image is
  acquired fail with `INVALID_ARGUMENT`.
- The acquired image is a 2D image with one mip level, one layer and one
  sample, in the configured format, extent and usage. Its public handles are
  valid from acquire until present. The surface owns them, so destroying
  either through the image API fails with `INVALID_ARGUMENT`. Configuring
  or destroying the surface destroys its images; an encoder that retained
  one keeps it alive until its submission completes, as for any destroyed
  image (`image_resources.md` section 7).
- Its contents are undefined at acquire, because the previous frame's
  contents are not guaranteed on any backend. Until a submitted command
  writes the image, that is a rendering scope that loads it with `CLEAR` or
  `DISCARD` or a copy into it, the RHI command layer rejects a rendering
  scope that preserves it, a read of it in a rendering scope, a copy from
  it and a download of it. Writes count in submission order: an encoder
  may read the image after its own earlier write, and another encoder may
  read it only once the writing encoder is submitted. Runtime records
  through the same command layer, so the rule needs nothing in Runtime.
- Present transitions the image to `VERNON_RHI_STATE_PRESENT` itself. The
  RHI tracks the image's state, so neither Runtime nor the host declares
  that final transition.

## 3. RHI API

```c
VERNON_RHI_DECLARE_HANDLE(VernonRhiSurface);

VERNON_RHI_STATUS_SURFACE_OUT_OF_DATE = 5  /* configure again, then retry */
VERNON_RHI_STATUS_SURFACE_LOST = 6         /* destroy and recreate the surface */

typedef enum VernonRhiPresentMode {
    VERNON_RHI_PRESENT_FIFO = 0,       /* vertical sync; supported everywhere */
    VERNON_RHI_PRESENT_MAILBOX = 1,
    VERNON_RHI_PRESENT_IMMEDIATE = 2
} VernonRhiPresentMode;

typedef struct VernonRhiSurfaceDescriptor {
    uint32_t struct_size;
    VernonRhiBackend backend;
    union {
        void *win32_window;                             /* HWND for DirectX 12 */
        void *metal_layer;                              /* CAMetalLayer */
        const VernonOpenGLSurfaceCallbacks *opengl;     /* section 5 */
    } native;
    uint32_t reserved[4];
} VernonRhiSurfaceDescriptor;

typedef struct VernonRhiSurfaceCapabilities {
    uint32_t struct_size;
    VernonRhiFormat formats[8];
    uint32_t format_count;
    uint32_t present_modes;            /* 1u << VernonRhiPresentMode */
    uint32_t usage;                    /* VernonRhiImageUsageBits the images may have */
    uint32_t min_image_count;
    uint32_t max_image_count;          /* 0: no limit */
    uint32_t current_width;            /* UINT32_MAX: the configuration decides */
    uint32_t current_height;
    uint32_t reserved[4];
} VernonRhiSurfaceCapabilities;

typedef struct VernonRhiSurfaceConfiguration {
    uint32_t struct_size;
    VernonRhiFormat format;
    uint32_t usage;                    /* must include COLOR_ATTACHMENT */
    uint32_t width;
    uint32_t height;
    VernonRhiPresentMode present_mode;
    uint32_t image_count;
    uint32_t reserved[4];
} VernonRhiSurfaceConfiguration;

typedef struct VernonRhiSurfaceImage {
    uint32_t struct_size;
    VernonRhiImage image;
    VernonRhiImageView view;
    uint32_t suboptimal;               /* still presentable; reconfigure when convenient */
    uint32_t reserved[4];
} VernonRhiSurfaceImage;

VernonRhiStatus vernonRhiDeviceCreateSurface(VernonRhiDevice, const VernonRhiSurfaceDescriptor *, VernonRhiSurface *);
VernonRhiStatus vernonRhiSurfaceGetCapabilities(VernonRhiDevice, VernonRhiSurface, VernonRhiSurfaceCapabilities *);
VernonRhiStatus vernonRhiSurfaceConfigure(VernonRhiDevice, VernonRhiSurface, const VernonRhiSurfaceConfiguration *);
VernonRhiStatus vernonRhiSurfaceAcquireImage(VernonRhiDevice, VernonRhiSurface, VernonRhiSurfaceImage *);
VernonRhiStatus vernonRhiSurfacePresent(VernonRhiDevice, VernonRhiSurface);
VernonRhiStatus vernonRhiDeviceDestroySurface(VernonRhiDevice, VernonRhiSurface);
```

- `VernonRhiSurfaceImage` is an output with `struct_size` and `reserved`,
  so a backend can later return more from acquire, such as a wait handle;
  acquire fails with `INVALID_ARGUMENT` when `struct_size` is too small.
- A backend or device that cannot present fails surface creation with
  `UNSUPPORTED`. There is no capability bit for it, because no host queries
  RHI capabilities before creating the resource that needs them. The
  descriptor's `backend` must be the device's, OpenGL and OpenGL ES being
  one backend; otherwise creation fails with `INVALID_ARGUMENT`.
- Configure fails with `UNSUPPORTED` for a format, mode, usage or image
  count outside the capabilities, and with `INVALID_ARGUMENT` for a zero
  extent, a usage without `COLOR_ATTACHMENT` or an unknown present mode. A
  minimized window has a zero extent; the host skips the window's frame
  until it resizes. A failed configure keeps the previous images.
- Acquire may block until the swapchain returns an image, which is how
  `FIFO` paces a frame. With `N` submissions in flight
  (`pipelined_submission.md` section 14.3), `image_count` of at least
  `N + 1` keeps acquire from waiting for the GPU as well.
- Acquire and present return `SURFACE_OUT_OF_DATE` after a resize the
  swapchain can no longer serve. A failed acquire leaves no image acquired;
  a failed present still releases the image.
- Destroying a surface does not wait for the submissions that use its
  images; their encoders keep the images alive (section 2).

## 4. Synchronization

The RHI orders presentation the way it orders everything else, by queue
submission:

- The first submission that uses an acquired image waits for the
  swapchain's acquire signal. Submissions after it are ordered by the queue.
- Present runs after every submission made before it. It records the
  transition to `PRESENT` in the RHI's own final command buffer and then
  hands the image to the swapchain. Runtime's own barriers for color
  targets stay as they are (`image_resources.md` section 8).
- Neither present nor acquire waits on the host for completion; a host that
  wants to bound latency waits on its own completions as it does today.

## 5. Backend mappings

| Backend | Native input | Swapchain | Acquire | Present |
|---|---|---|---|---|
| Vulkan | open (section 9); the descriptor gains its member when that is decided | `VkSwapchainKHR` | `vkAcquireNextImageKHR` with a per-image semaphore | transition to `PRESENT_SRC_KHR`, signal a semaphore, `vkQueuePresentKHR` |
| DirectX 12 | `HWND` | `IDXGISwapChain3`, flip-discard, created on the RHI's direct queue | frame-latency waitable object, `GetCurrentBackBufferIndex` | transition to `PRESENT`, `Present` |
| Metal | `CAMetalLayer` | the layer | `nextDrawable` | `presentDrawable:` on a command buffer committed after the last submission |
| OpenGL, OpenGL ES | `VernonOpenGLSurfaceCallbacks` | RHI-owned textures, `image_count` of them | immediate | make the window current, flipped blit, swap |

The DXGI factory belongs to the Direct3D 12 stack and does not link a
window-system library. Who creates the Vulkan surface is open (section 9);
the native input in the table cannot work for an owned device as it stands.

### OpenGL

```c
typedef void (*VernonOpenGLSwapIntervalFn)(void *user_data, int32_t interval);
typedef void (*VernonOpenGLSwapBuffersFn)(void *user_data);
typedef void (*VernonOpenGLDrawableSizeFn)(void *user_data, uint32_t *width, uint32_t *height);

typedef struct VernonOpenGLSurfaceCallbacks {
    uint32_t struct_size;
    void *user_data;
    VernonOpenGLMakeCurrentFn make_current;       /* the device's context on this window */
    VernonOpenGLSwapIntervalFn swap_interval;     /* SDL_GL_SetSwapInterval, with the window current */
    VernonOpenGLSwapBuffersFn swap_buffers;       /* SDL_GL_SwapWindow */
    VernonOpenGLDrawableSizeFn drawable_size;     /* SDL_GetWindowSizeInPixels */
    uint32_t reserved[4];
} VernonOpenGLSurfaceCallbacks;
```

An OpenGL surface image is an ordinary RHI texture, not the window's default
framebuffer. Present makes the device's context current on the window, blits
the texture into framebuffer 0 with its rows reversed, swaps, and makes the
device's own current target current again.

The texture is required by the vertex convention. Every GL vertex stage
negates y (`design.md` section 10), so a Program's row 0 is the image's top
row on every backend. The default framebuffer counts rows from the bottom
of the window. Drawing into it directly would show every frame upside down,
and the negation is compiled into the cooked shaders, so it cannot change per
target. The texture also makes the GL surface image a normal image whose
descriptor, views, hazards and readback are those of any other image, and it
makes multiple windows independent of which window is current while a frame
records.

Costs and limits of the fallback tier, recorded as such:

- one full-window blit per present;
- every surface's window must accept the device's own context
  (`SDL_GL_MakeCurrent(window, device_context)`), because framebuffer and
  vertex-array objects are not shared between contexts. A host that creates
  one context per window cannot present those windows through one device;
- `MAILBOX` and `IMMEDIATE` map to a swap interval of 0, `FIFO` to 1, and
  the driver alone paces the frame, so `suboptimal` is never reported;
- the window does not resize the textures, so acquire reports
  `SURFACE_OUT_OF_DATE` whenever `drawable_size` differs from the configured
  extent;
- the surface offers `rgba8_unorm` only, with `COLOR_ATTACHMENT` and
  `TRANSFER_SOURCE` usage. Sampled and storage usage are withheld so that
  every read of an acquired image is one the command layer checks
  (section 2).

## 6. Formats and color space

Surfaces present nonlinear sRGB, and every surface offers at least one of
`rgba8_unorm` and `bgra8_unorm`:

| Backend | Formats offered |
|---|---|
| Vulkan | what the surface reports; some drivers report `B8G8R8A8` and not `R8G8B8A8` |
| DirectX 12 | `rgba8_unorm`, `bgra8_unorm` |
| Metal | `bgra8_unorm` (`CAMetalLayer` has no RGBA8) |
| OpenGL | `rgba8_unorm` |

So the format table (`VernonFormats.def`) gains `bgra8_unorm` and
`bgra8_srgb` rows: renderable, blendable, filterable and copyable, with host
data format `BGRA`. A host whose display pass writes display-encoded values
configures the `unorm` format; the `srgb` formats encode on store. HDR
surfaces (`rgba16_float` scRGB, `rgb10a2` PQ) are out of scope.

A graphics pipeline's color target formats are fixed when it is cooked, and
every backend requires the pipeline's format to match the attachment's
exactly. A Program that draws into a surface therefore needs one cooked
pipeline per format the host may configure. A pipeline's color target
formats are a static specialization argument (`language/contract.md`
section 6.6.1): one declaration cooks an `rgba8_unorm` and a `bgra8_unorm`
specialization, and the host selects the one matching the configured format.
Cooking a separate asset per format was rejected because it duplicates the
declaration; restricting surfaces to `rgba8_unorm` was rejected because Metal
and some Vulkan drivers do not offer it.

The work is phased. Phase 1 is format specialization and OpenGL surfaces,
which offer only `rgba8_unorm`. The `bgra8` rows land with the first backend
whose surfaces need them (Metal, or Vulkan on a driver without `R8G8B8A8`),
together with their view-compatibility classes and DirectX 12 typeless
mapping, so each row is exercised by a surface test when it is added.

## 7. Runtime

Runtime gains no new invocation fields. An acquired view is an imported image
view:

- the provider resolves its descriptor from the RHI record, as for any image;
- Runtime's existing check that a color target's format equals the Program's
  cooked target format applies unchanged;
- the "no read before a write" rule of section 2 is checked by the RHI
  command layer that Runtime's rendering scopes go through, so Runtime does
  not learn what a surface is.

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

1. **Python.** Whether `RuntimeSession` exposes surfaces now or after the
   C++ host path is accepted.
2. **Version.** The RHI ABI change falls under `PROGRAM_VERSION`
   (`README.md`); the release decision belongs to the owner.
3. **Vulkan surface creation.** An owned device creates its `VkInstance`
   without `VK_KHR_surface` or a platform surface extension and does not
   expose the instance, so a host cannot create a `VkSurfaceKHR` for it. The
   candidates are: the host passes native window handles and the RHI
   creates the surface, enabling the surface extensions the loader offers
   (wgpu, Dawn); the host names the instance extensions at device creation
   and the RHI exposes its instance; or the RHI always enables the surface
   extensions and exposes its instance. Vulkan and DirectX 12 surfaces are
   deferred until this is decided, and both backends fail surface creation
   with `UNSUPPORTED`.

## 10. Acceptance

- Every backend that creates surfaces: create, configure, acquire, clear,
  present, and on OpenGL read the surface image back exactly before
  present. Every other backend fails creation with `UNSUPPORTED`.
- A recorded Program draws into an acquired image and the result matches the
  same draw into an ordinary image of that format.
- Two surfaces on one device present in one frame, with one submission
  writing both images.
- Rejected before native encoding: `PRESERVE` load or sampled read of a freshly
  acquired image; acquire twice; present without acquire; configure or
  destroy while an image is acquired; a format, mode or usage outside the
  capabilities; a borrowed device creating a surface.
- Resize: acquire or present reports `SURFACE_OUT_OF_DATE`, and configure
  followed by acquire succeeds.
- No backend renders into the OpenGL default framebuffer, and no host-side
  copy of a Runtime image to a back buffer remains.

## 11. Contract changes on acceptance

- `VernonRHI.h` and `VernonOpenGLContext.h`: the API in sections 3 and 5.
- `VernonFormats.def`: the `bgra8` rows, with the backend that first needs
  them (section 6).
- [`image_resources.md`](image_resources.md): surface images and the
  undefined-at-acquire rule.
- [`design.md`](design.md) section 10: the backend mappings.
- `README.md`: this document moves from active plans to architecture.
