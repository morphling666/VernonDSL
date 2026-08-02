# Security policy

## Supported versions

Security fixes are provided for the latest published 0.1 patch release.
Development branches and older patch releases are not supported security
channels.

## Reporting a vulnerability

Report suspected vulnerabilities through GitHub's private vulnerability
reporting for the `morphling666/VernonDSL` repository. Do not open a public
issue for an unpatched vulnerability.

Include affected versions and platforms, impact, reproduction steps or a
proof-of-concept, and any known mitigation. Reports should receive an initial
acknowledgement within seven days. Disclosure timing is coordinated after the
issue is reproduced and a fix or mitigation is available.

VernonDSL compiles and loads executable CPU/GPU artifacts. Applications must
treat untrusted Python DSL source, MLIR, pipeline manifests, native objects,
shader artifacts, and caches as untrusted executable input and isolate them
accordingly. The 0.1.1 API does not provide a sandbox.
