# Support policy

## Supported installations

VernonDSL 0.1.2 provides wheels for:

- Windows x64;
- Linux x64 with the wheel's declared manylinux baseline;
- Apple Silicon macOS 15 or newer;
- CPython 3.11 through 3.14.

Linux wheels are repaired with `auditwheel`; the platform tag in each wheel
filename defines its manylinux and minimum-glibc compatibility. VernonDSL does
not claim compatibility with an older baseline than that tag.

Python 3.11 and 3.12 installations require NumPy 1.26 or newer but below 2.0.
Python 3.13 and 3.14 installations require NumPy 2.1 or newer but below 3.0.
Intel macOS, 32-bit platforms, source distributions, PyPy, and unsupported
Python versions are outside the published 0.1.2 wheel matrix.

## Runtime backends

CPU compute requires no external GPU runtime. CUDA requires a compatible NVIDIA
driver. Vulkan requires a Vulkan loader, ICD, and usable device; macOS users
must install a Vulkan loader and MoltenVK separately. DirectX 12 requires
Windows and a compatible adapter. OpenGL/OpenGL ES require a compatible owned
or externally supplied context. Metal requires a supported Apple Silicon
device and the system Metal framework.

GPU availability is discovered at runtime. A compiled backend is not a promise
that the host has a usable device. Unsupported capabilities fail explicitly;
they do not silently select a different backend.

## Getting help

Use GitHub Issues for reproducible defects and support questions. Include the
VernonDSL version, Python version, operating system and architecture, selected
backend, device/driver information, complete diagnostic text, and a minimal
reproducer. Security reports must follow [`SECURITY.md`](SECURITY.md), not a
public issue.

The latest 0.1 patch release receives correctness and security fixes. Older
0.1 patch releases may be asked to upgrade before investigation.
