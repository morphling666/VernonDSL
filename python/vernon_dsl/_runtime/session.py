from __future__ import annotations

import contextvars
import itertools
import threading
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

try:
    from .. import _native
except (ImportError, OSError):
    _native = None


@dataclass(frozen=True)
class Architecture:
    name: str


cpu = Architecture("cpu")
cuda = Architecture("cuda")
vulkan = Architecture("vulkan")
directx = Architecture("directx")
metal = Architecture("metal")
opengl = Architecture("opengl")
opengles = Architecture("opengles")

_SUPPORTED_ARCHITECTURES = frozenset({cpu, cuda, vulkan, directx, metal, opengl, opengles})
_OPENGL_ARCHITECTURES = frozenset({opengl, opengles})
_DEFAULT_API_VERSIONS = {opengl: (4, 3), opengles: (3, 1)}


@dataclass(frozen=True)
class _ExternalOpenGLContext:
    user_data: int
    make_current: int
    get_proc_address: int
    api_version: tuple[int, int]


@dataclass(frozen=True)
class RuntimeConfiguration:
    """Canonical immutable identity for constructing a Runtime session."""

    arch: Architecture
    api_version: tuple[int, int] | None
    _external_context: _ExternalOpenGLContext | None = field(default=None, repr=False)


@dataclass(frozen=True)
class _RuntimeSessionState:
    native_runtime: Any
    rhi_host: Any | None
    identity: int


class RuntimeUnavailableError(RuntimeError):
    """Raised when a requested Runtime session cannot be fully constructed."""


_CACHE_MISSING = object()


def _accept_cached(_: Any) -> bool:
    return True


@dataclass
class _ArtifactFlight:
    condition: threading.Condition
    epoch: int
    participants: int = 1
    complete: bool = False
    cancelled: bool = False
    value: Any = _CACHE_MISSING
    error: BaseException | None = None


class _SessionCachePartition:
    def __init__(self) -> None:
        self.snapshot: dict[Any, Any] = {}
        self.epoch = 0
        self.lock = threading.Lock()
        self.flights: dict[Any, _ArtifactFlight] = {}

    def _leave_flight(self, key: Any, flight: _ArtifactFlight) -> None:
        flight.participants -= 1
        if flight.participants == 0 and self.flights.get(key) is flight:
            self.flights.pop(key)

    def get_or_create(
        self,
        key: Any,
        factory: Callable[[], Any],
        validator: Callable[[Any], bool],
    ) -> Any:
        while True:
            snapshot = self.snapshot
            cached = snapshot.get(key, _CACHE_MISSING)
            if cached is not _CACHE_MISSING and validator(cached):
                return cached

            with self.lock:
                current = self.snapshot.get(key, _CACHE_MISSING)
                if current is not cached:
                    continue
                if current is not _CACHE_MISSING:
                    updated = dict(self.snapshot)
                    updated.pop(key, None)
                    self.snapshot = updated
                flight = self.flights.get(key)
                if flight is None:
                    flight = _ArtifactFlight(threading.Condition(self.lock), self.epoch)
                    self.flights[key] = flight
                    producer = True
                else:
                    producer = False
                if not producer:
                    flight.participants += 1
                    flight.condition.notify_all()
                    try:
                        while not flight.complete:
                            flight.condition.wait()
                        if flight.error is not None:
                            raise flight.error
                        if flight.cancelled:
                            continue
                        return flight.value
                    finally:
                        self._leave_flight(key, flight)

            try:
                value = factory()
            except BaseException as error:
                with self.lock:
                    flight.error = error
                    flight.complete = True
                    flight.condition.notify_all()
                    self._leave_flight(key, flight)
                raise

            with self.lock:
                if flight.epoch != self.epoch:
                    flight.cancelled = True
                else:
                    updated = dict(self.snapshot)
                    updated[key] = value
                    self.snapshot = updated
                    flight.value = value
                flight.complete = True
                flight.condition.notify_all()
                self._leave_flight(key, flight)
                if flight.cancelled:
                    continue
                return value

    def clear(self) -> None:
        with self.lock:
            self.epoch += 1
            self.snapshot = {}
            for flight in self.flights.values():
                flight.cancelled = True
                flight.complete = True
                flight.condition.notify_all()
            self.flights.clear()


class _SessionArtifactCache:
    """Object-local native artifacts partitioned by weak RuntimeSession identity."""

    def __init__(self) -> None:
        self._entries: weakref.WeakKeyDictionary[RuntimeSession, _SessionCachePartition] = weakref.WeakKeyDictionary()
        self._lock = threading.Lock()

    def _partition(self, session: RuntimeSession) -> _SessionCachePartition:
        partition = self._entries.get(session)
        if partition is not None:
            return partition
        with self._lock:
            partition = self._entries.get(session)
            if partition is None:
                partition = _SessionCachePartition()
                self._entries[session] = partition
            return partition

    def get_or_create(
        self,
        session: RuntimeSession,
        key: Any,
        factory: Callable[[], Any],
        validator: Callable[[Any], bool] | None = None,
    ) -> Any:
        return self._partition(session).get_or_create(key, factory, validator or _accept_cached)

    def clear(self) -> None:
        with self._lock:
            partitions = tuple(self._entries.values())
        for partition in partitions:
            partition.clear()


class _ArtifactCache:
    """Backend-independent artifact cache with the same publication protocol."""

    def __init__(self) -> None:
        self._partition = _SessionCachePartition()

    def get_or_create(
        self,
        key: Any,
        factory: Callable[[], Any],
        validator: Callable[[Any], bool] | None = None,
    ) -> Any:
        return self._partition.get_or_create(key, factory, validator or _accept_cached)

    def clear(self) -> None:
        self._partition.clear()


@dataclass(frozen=True)
class _InvocationContext:
    session: RuntimeSession

    @property
    def identity(self) -> int:
        return self.session.identity


@dataclass
class _ConstructionFlight:
    condition: threading.Condition
    waiters: int = 0
    complete: bool = False
    session: RuntimeSession | None = None
    error: BaseException | None = None


class _SessionRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._default: RuntimeSession | None = None
        self._external_opengl_contexts: dict[Architecture, _ExternalOpenGLContext] = {}
        self._construction_flights: dict[RuntimeConfiguration, _ConstructionFlight] = {}

    @property
    def default(self) -> RuntimeSession | None:
        return self._default

    def external_context(self, arch: Architecture) -> _ExternalOpenGLContext | None:
        with self._lock:
            return self._external_opengl_contexts.get(arch)

    def register_external_context(self, arch: Architecture, registration: _ExternalOpenGLContext) -> None:
        with self._lock:
            self._external_opengl_contexts[arch] = registration

    def select_default(self, configuration: RuntimeConfiguration) -> RuntimeSession:
        current = self._default
        if current is not None and current.configuration == configuration:
            return current

        with self._lock:
            current = self._default
            if current is not None and current.configuration == configuration:
                return current
            flight = self._construction_flights.get(configuration)
            if flight is None:
                flight = _ConstructionFlight(threading.Condition(self._lock))
                self._construction_flights[configuration] = flight
                producer = True
            else:
                producer = False
            if not producer:
                flight.waiters += 1
                flight.condition.notify_all()
                try:
                    while not flight.complete:
                        flight.condition.wait()
                    if flight.error is not None:
                        raise flight.error
                    assert flight.session is not None
                    return flight.session
                finally:
                    flight.waiters -= 1

        try:
            candidate = RuntimeSession._create(configuration)
        except BaseException as error:
            with self._lock:
                flight.error = error
                flight.complete = True
                self._construction_flights.pop(configuration, None)
                flight.condition.notify_all()
            raise

        with self._lock:
            current = self._default
            if current is not None and current.configuration == configuration:
                selected = current
                candidate._retired = True
            else:
                selected = candidate
                self._default = candidate
                if current is not None:
                    current._retired = True
            flight.session = selected
            flight.complete = True
            self._construction_flights.pop(configuration, None)
            flight.condition.notify_all()
            return selected


_registry = _SessionRegistry()
_session_identities = itertools.count(1)
_scoped_session: contextvars.ContextVar[RuntimeSession | None] = contextvars.ContextVar(
    "vernon_runtime_scoped_session",
    default=None,
)
_scope_stack: contextvars.ContextVar[tuple[tuple[RuntimeSession, contextvars.Token[RuntimeSession | None]], ...]] = (
    contextvars.ContextVar("vernon_runtime_scope_stack", default=())
)
_active_invocation: contextvars.ContextVar[_InvocationContext | None] = contextvars.ContextVar(
    "vernon_runtime_invocation_context",
    default=None,
)


def _validate_api_version(api_version: tuple[int, int]) -> None:
    if (
        not isinstance(api_version, tuple)
        or len(api_version) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in api_version)
    ):
        raise ValueError("api_version must be a non-negative (major, minor) pair")


def _canonical_configuration(
    arch: Architecture,
    api_version: tuple[int, int] | None,
) -> RuntimeConfiguration:
    if arch not in _SUPPORTED_ARCHITECTURES:
        raise ValueError("unsupported VernonDSL runtime architecture")
    if api_version is not None:
        if arch not in _OPENGL_ARCHITECTURES:
            raise ValueError("api_version is only valid for OpenGL runtimes")
        _validate_api_version(api_version)
    if arch not in _OPENGL_ARCHITECTURES:
        return RuntimeConfiguration(arch, None)
    external = _registry.external_context(arch)
    version = api_version or (external.api_version if external is not None else _DEFAULT_API_VERSIONS[arch])
    return RuntimeConfiguration(arch, version, external)


def _backend_name(arch: Architecture) -> str:
    return {
        cpu: "CPU",
        cuda: "CUDA",
        vulkan: "VULKAN",
        directx: "DIRECTX12",
        metal: "METAL",
        opengl: "OPENGL",
        opengles: "OPENGL_ES",
    }[arch]


def _construct_state(configuration: RuntimeConfiguration) -> _RuntimeSessionState:
    if _native is None:
        raise RuntimeUnavailableError(
            f"{configuration.arch.name} requires vernon_dsl._native; build the native "
            "targets or install a wheel containing the native module"
        )

    arch = configuration.arch
    backend_name = _backend_name(arch)
    backend = getattr(_native.RuntimeBackend, backend_name)
    uses_rhi = arch != cpu
    if uses_rhi and (not hasattr(_native, "RhiHost") or not hasattr(_native, "RhiBackend")):
        raise RuntimeUnavailableError(f"{arch.name} requires Vernon RHI support in vernon_dsl._native")

    rhi_host = None
    if arch in _OPENGL_ARCHITECTURES:
        version = configuration.api_version
        assert version is not None
        rhi_backend = getattr(_native.RhiBackend, backend_name)
        external = configuration._external_context
        if external is not None:
            rhi_host = _native.RhiHost.create_external_opengl(
                rhi_backend,
                external.user_data,
                external.make_current,
                external.get_proc_address,
                *version,
            )
        else:
            try:
                rhi_host = _native.RhiHost.create_owned_opengl(rhi_backend, *version)
            except RuntimeError as error:
                raise RuntimeUnavailableError(
                    f"{arch.name} {version[0]}.{version[1]} context is unavailable"
                ) from error
        native_runtime = rhi_host.create_runtime()
    elif arch == cpu:
        native_runtime = _native.Runtime(backend)
    else:
        if not _native.runtime_available(backend):
            raise RuntimeUnavailableError(f"{arch.name} loader or a usable device is unavailable")
        rhi_host = _native.RhiHost(getattr(_native.RhiBackend, backend_name))
        native_runtime = rhi_host.create_runtime()

    capabilities = dict(native_runtime.capabilities)
    if not capabilities.get("available", False):
        raise RuntimeUnavailableError(f"{arch.name} runtime probe reported unavailable capabilities")
    return _RuntimeSessionState(native_runtime, rhi_host, next(_session_identities))


class RuntimeSession:
    """A fully probed Runtime owner that can be selected context-locally."""

    def __init__(
        self,
        *,
        arch: Architecture = cpu,
        api_version: tuple[int, int] | None = None,
    ) -> None:
        self._initialize(_canonical_configuration(arch, api_version))

    @classmethod
    def _create(cls, configuration: RuntimeConfiguration) -> RuntimeSession:
        candidate = cls.__new__(cls)
        candidate._initialize(configuration)
        return candidate

    def _initialize(self, configuration: RuntimeConfiguration) -> None:
        try:
            state = _construct_state(configuration)
        except RuntimeUnavailableError:
            raise
        except (OSError, RuntimeError) as error:
            raise RuntimeUnavailableError(f"{configuration.arch.name} Runtime session probe failed: {error}") from error
        self._configuration = configuration
        self._state = state
        self._retired = False

    @property
    def configuration(self) -> RuntimeConfiguration:
        return self._configuration

    @property
    def arch(self) -> Architecture:
        return self._configuration.arch

    @property
    def native(self) -> Any:
        return _native

    @property
    def native_runtime(self) -> Any:
        return self._state.native_runtime

    @property
    def rhi_host(self) -> Any | None:
        return self._state.rhi_host

    @property
    def identity(self) -> int:
        return self._state.identity

    @property
    def retired(self) -> bool:
        return self._retired

    @property
    def interactive_glsl_version(self) -> int:
        version = self.configuration.api_version
        return 0 if version is None else version[0] * 100 + version[1] * 10

    def __enter__(self) -> RuntimeSession:
        token = _scoped_session.set(self)
        _scope_stack.set((*_scope_stack.get(), (self, token)))
        return self

    def __exit__(self, exception_type: Any, exception: Any, traceback: Any) -> None:
        stack = _scope_stack.get()
        if not stack or stack[-1][0] is not self:
            raise RuntimeError("RuntimeSession scopes must exit in nesting order")
        _scope_stack.set(stack[:-1])
        _scoped_session.reset(stack[-1][1])

    @contextmanager
    def scope(self) -> Iterator[RuntimeSession]:
        with self:
            yield self


def current_session() -> RuntimeSession | None:
    """Return the anchored or scoped session, falling back to the process default."""

    invocation = _active_invocation.get()
    if invocation is not None:
        return invocation.session
    scoped = _scoped_session.get()
    return scoped if scoped is not None else _registry.default


def _session_state() -> RuntimeSession:
    selected = current_session()
    if selected is None:
        raise RuntimeError("VernonDSL Runtime is not initialized; call vd.init() or enter a RuntimeSession scope")
    return selected


@contextmanager
def _invocation_context() -> Iterator[_InvocationContext]:
    existing = _active_invocation.get()
    if existing is not None:
        yield existing
        return
    context = _InvocationContext(_session_state())
    token = _active_invocation.set(context)
    try:
        yield context
    finally:
        _active_invocation.reset(token)


@contextmanager
def _use_invocation_context(context: _InvocationContext) -> Iterator[_InvocationContext]:
    token = _active_invocation.set(context)
    try:
        yield context
    finally:
        _active_invocation.reset(token)


def _execution_context() -> _InvocationContext:
    context = _active_invocation.get()
    if context is None:
        raise RuntimeError("VernonDSL operation has no anchored invocation context")
    return context


def init(*, arch: Architecture = cpu, api_version: tuple[int, int] | None = None) -> RuntimeSession:
    """Transactionally select the process-default Runtime session."""

    configuration = _canonical_configuration(arch, api_version)
    return _registry.select_default(configuration)


def register_external_opengl_context(
    *,
    arch: Architecture,
    user_data: int,
    make_current: int,
    get_proc_address: int,
    api_version: tuple[int, int],
) -> None:
    if arch not in _OPENGL_ARCHITECTURES:
        raise ValueError("external contexts are only valid for OpenGL backends")
    if (
        not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in (user_data,))
        or not isinstance(make_current, int)
        or isinstance(make_current, bool)
        or make_current <= 0
        or not isinstance(get_proc_address, int)
        or isinstance(get_proc_address, bool)
        or get_proc_address <= 0
    ):
        raise ValueError("external context callbacks must be non-zero addresses")
    _validate_api_version(api_version)
    registration = _ExternalOpenGLContext(user_data, make_current, get_proc_address, api_version)
    _registry.register_external_context(arch, registration)


__all__ = [
    "Architecture",
    "RuntimeConfiguration",
    "RuntimeSession",
    "RuntimeUnavailableError",
    "cpu",
    "cuda",
    "current_session",
    "directx",
    "init",
    "metal",
    "opengl",
    "opengles",
    "register_external_opengl_context",
    "vulkan",
]
