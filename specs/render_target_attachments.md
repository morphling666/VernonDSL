# RenderTarget Attachment Architecture

## Decision

The public Python Runtime uses a backend-neutral `RenderTarget` attachment
collection. It does not expose OpenGL `Framebuffer` or `Renderbuffer`
semantics, and it does not model depth as a standalone `DepthTexture` class.

```python
color = vd.Texture.zeros(shape=(size, size))
target = (
    vd.RenderTarget(shape=(size, size))
    .attach_color(0, color)
    .attach_depth(format=vd.depth32)
)

render(..., target=target)
rgba = color.to_numpy()
```

`RenderTarget` lowers to the native attachment model appropriate to each
backend:

- OpenGL/OpenGL ES: framebuffer object attachments.
- Vulkan: dynamic-rendering attachments or framebuffer attachments.
- DirectX 12: RTV and DSV bindings.

The backend implementation remains responsible for concrete framebuffer,
descriptor, view, layout-transition, and render-pass objects.

## Resource boundaries

- `Texture` represents an image that the application can upload, sample, or
  read back.
- A color attachment references an external `Texture` at one output location.
- A render-only depth attachment is an attachment image owned by
  `RenderTarget`; it is not independently exposed, sampled, or downloaded.
- Sampled depth is a future generic `Texture` format/usage extension. It must
  not reintroduce a dedicated `DepthTexture` resource type.
- All attachments in one `RenderTarget` have the same width and height.
- Color locations are unique and there is at most one depth attachment.

## Runtime boundary

`RenderTarget` is a Python resource-organization abstraction. At invocation it
expands into the existing native `VernonColorAttachment` array and optional
`VernonDepthAttachment`. These C ABI records describe attachment roles and
provider references; they do not create a second resource ownership path.

The Runtime continues to receive non-owning provider resource references.
VernonRHI owns images and backend-native attachment state.

## Migration

- Remove `DepthTexture` from public exports and Runtime resources.
- Replace `target`/`targets`/`depth` invocation combinations with
  `target=RenderTarget`.
- Replace the native `create_depth_image` convenience method with generic
  attachment-image creation by format and usage.
- Migrate pipeline tests and `examples/pbr.py`.
- Validate duplicate locations, mismatched dimensions, multiple depth
  attachments, missing color outputs, and backend/device ownership.
- Run complete CTest and Python test suites across available GPU backends.

This change is limited to VernonDSL. Vernon Engine migration remains paused.
