from __future__ import annotations

import gc
import threading
import time
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest import mock

import vernon_dsl as vd
import vernon_dsl._runtime.session as runtime_session


class RuntimeSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        registry = runtime_session._registry
        with registry._lock:
            self._default = registry._default
            self._external_contexts = dict(registry._external_opengl_contexts)
            registry._default = None
            registry._external_opengl_contexts.clear()
            registry._construction_flights.clear()

    def tearDown(self) -> None:
        registry = runtime_session._registry
        with registry._lock:
            registry._default = self._default
            registry._external_opengl_contexts.clear()
            registry._external_opengl_contexts.update(self._external_contexts)
            registry._construction_flights.clear()

    @staticmethod
    def _state(configuration: vd.RuntimeConfiguration) -> runtime_session._RuntimeSessionState:
        runtime = SimpleNamespace(capabilities={"available": True}, configuration=configuration)
        rhi_host = None if configuration.arch == vd.cpu else object()
        return runtime_session._RuntimeSessionState(
            runtime,
            rhi_host,
            next(runtime_session._session_identities),
        )

    def test_identical_init_returns_existing_default_without_reconstruction(self) -> None:
        with mock.patch.object(runtime_session, "_construct_state", side_effect=self._state) as construct:
            first = vd.init(arch=vd.cpu)
            second = vd.init(arch=vd.cpu)

        self.assertIs(second, first)
        self.assertIs(vd.current_session(), first)
        self.assertEqual(construct.call_count, 1)
        self.assertEqual(first.configuration, vd.RuntimeConfiguration(vd.cpu, None))

    def test_replacement_retires_old_default_after_candidate_probe(self) -> None:
        with mock.patch.object(runtime_session, "_construct_state", side_effect=self._state):
            old = vd.init(arch=vd.cpu)
            replacement = vd.init(arch=vd.metal)

        self.assertIs(vd.current_session(), replacement)
        self.assertTrue(old.retired)
        self.assertFalse(replacement.retired)

    def test_failed_candidate_leaves_default_unchanged(self) -> None:
        with mock.patch.object(runtime_session, "_construct_state", side_effect=self._state):
            old = vd.init(arch=vd.cpu)
        with mock.patch.object(
            runtime_session,
            "_construct_state",
            side_effect=runtime_session.RuntimeUnavailableError("probe failed"),
        ):
            with self.assertRaisesRegex(runtime_session.RuntimeUnavailableError, "probe failed"):
                vd.init(arch=vd.metal)

        self.assertIs(vd.current_session(), old)
        self.assertFalse(old.retired)

    def test_nested_scopes_are_context_local_and_do_not_replace_default(self) -> None:
        with mock.patch.object(runtime_session, "_construct_state", side_effect=self._state):
            default = vd.init(arch=vd.cpu)
            outer = vd.RuntimeSession(arch=vd.metal)
            inner = vd.RuntimeSession(arch=vd.vulkan)

        with outer:
            self.assertIs(vd.current_session(), outer)
            with inner:
                self.assertIs(vd.current_session(), inner)
            self.assertIs(vd.current_session(), outer)
        self.assertIs(vd.current_session(), default)
        self.assertFalse(outer.retired)
        self.assertFalse(inner.retired)

    def test_concurrent_scopes_do_not_cross_contexts(self) -> None:
        with mock.patch.object(runtime_session, "_construct_state", side_effect=self._state):
            default = vd.init(arch=vd.cpu)
            first = vd.RuntimeSession(arch=vd.metal)
            second = vd.RuntimeSession(arch=vd.vulkan)
        barrier = threading.Barrier(2)

        def selected(session: vd.RuntimeSession) -> tuple[vd.RuntimeSession, vd.RuntimeSession]:
            with session:
                barrier.wait()
                return vd.current_session(), runtime_session._session_state()

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = tuple(executor.map(selected, (first, second)))

        self.assertEqual(results, ((first, first), (second, second)))
        self.assertIs(vd.current_session(), default)

    def test_concurrent_replacements_publish_one_complete_default(self) -> None:
        barrier = threading.Barrier(2)

        def construct(configuration: vd.RuntimeConfiguration) -> runtime_session._RuntimeSessionState:
            barrier.wait()
            return self._state(configuration)

        with mock.patch.object(runtime_session, "_construct_state", side_effect=construct):
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = tuple(executor.map(lambda arch: vd.init(arch=arch), (vd.metal, vd.vulkan)))

        selected = vd.current_session()
        self.assertIn(selected, results)
        self.assertEqual(sum(not session.retired for session in results), 1)

    def test_concurrent_identical_init_shares_one_candidate_probe(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def construct(configuration: vd.RuntimeConfiguration) -> runtime_session._RuntimeSessionState:
            started.set()
            release.wait()
            return self._state(configuration)

        with mock.patch.object(runtime_session, "_construct_state", side_effect=construct) as probe:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = tuple(executor.submit(vd.init, arch=vd.cpu) for _ in range(4))
                self.assertTrue(started.wait(timeout=5))
                with runtime_session._registry._lock:
                    flight = runtime_session._registry._construction_flights[vd.RuntimeConfiguration(vd.cpu, None)]
                    self.assertTrue(flight.condition.wait_for(lambda: flight.waiters == 3, timeout=5))
                release.set()
                results = tuple(future.result(timeout=5) for future in futures)

        self.assertTrue(all(session is results[0] for session in results))
        self.assertEqual(probe.call_count, 1)

    def test_concurrent_identical_init_shares_probe_failure(self) -> None:
        barrier = threading.Barrier(4)
        started = threading.Event()
        release = threading.Event()

        def construct(configuration: vd.RuntimeConfiguration) -> runtime_session._RuntimeSessionState:
            started.set()
            release.wait()
            raise runtime_session.RuntimeUnavailableError(f"{configuration.arch.name} unavailable")

        def initialize(_: int) -> str:
            barrier.wait()
            try:
                vd.init(arch=vd.metal)
            except runtime_session.RuntimeUnavailableError as error:
                return str(error)
            raise AssertionError("failed RuntimeSession probe unexpectedly succeeded")

        with mock.patch.object(runtime_session, "_construct_state", side_effect=construct) as probe:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = tuple(executor.submit(initialize, index) for index in range(4))
                self.assertTrue(started.wait(timeout=5))
                with runtime_session._registry._lock:
                    flight = runtime_session._registry._construction_flights[vd.RuntimeConfiguration(vd.metal, None)]
                    self.assertTrue(flight.condition.wait_for(lambda: flight.waiters == 3, timeout=5))
                release.set()
                errors = tuple(future.result(timeout=5) for future in futures)

        self.assertEqual(errors, ("metal unavailable",) * 4)
        self.assertEqual(probe.call_count, 1)
        self.assertIsNone(runtime_session._registry.default)

    def test_invocation_anchor_survives_default_replacement_until_scope_exit(self) -> None:
        with mock.patch.object(runtime_session, "_construct_state", side_effect=self._state):
            old = vd.init(arch=vd.cpu)
            old_reference = weakref.ref(old)
            with runtime_session._invocation_context() as anchored:
                replacement = vd.init(arch=vd.metal)
                del old
                gc.collect()
                self.assertIs(runtime_session._session_state(), anchored.session)
                self.assertIs(vd.current_session(), anchored.session)
                self.assertIs(runtime_session._registry.default, replacement)
                self.assertIsNotNone(old_reference())
            del anchored

        gc.collect()
        self.assertIsNone(old_reference())

    def test_session_local_cache_does_not_keep_retired_session_alive(self) -> None:
        cache = runtime_session._SessionArtifactCache()
        with mock.patch.object(runtime_session, "_construct_state", side_effect=self._state):
            session = vd.RuntimeSession(arch=vd.cpu)
        cache.get_or_create(session, "executable", object)
        reference = weakref.ref(session)

        del session
        gc.collect()

        self.assertIsNone(reference())
        self.assertEqual(len(cache._entries), 0)

    def test_artifact_cache_single_flight_covers_factory_and_releases_flight(self) -> None:
        cache = runtime_session._ArtifactCache()
        calls = 0
        calls_lock = threading.Lock()

        def construct() -> object:
            nonlocal calls
            with calls_lock:
                calls += 1
            time.sleep(0.05)
            return object()

        with ThreadPoolExecutor(max_workers=4) as executor:
            values = tuple(executor.map(lambda _: cache.get_or_create("key", construct), range(4)))

        self.assertTrue(all(value is values[0] for value in values))
        self.assertEqual(calls, 1)
        self.assertEqual(cache._partition.flights, {})

    def test_cache_clear_cancels_stale_publication_without_detaching_waiters(self) -> None:
        cache = runtime_session._ArtifactCache()
        first_started = threading.Event()
        release_first = threading.Event()
        calls_lock = threading.Lock()
        calls = 0

        def construct() -> str:
            nonlocal calls
            with calls_lock:
                calls += 1
                call = calls
            if call == 1:
                first_started.set()
                release_first.wait()
                return "stale"
            return "current"

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(cache.get_or_create, "key", construct)
            self.assertTrue(first_started.wait(timeout=5))
            cache.clear()
            second = executor.submit(cache.get_or_create, "key", construct)
            self.assertEqual(second.result(timeout=5), "current")
            release_first.set()
            self.assertEqual(first.result(timeout=5), "current")

        self.assertEqual(calls, 2)
        self.assertEqual(cache._partition.snapshot, {"key": "current"})
        self.assertEqual(cache._partition.flights, {})

    def test_stale_cache_value_has_one_replacement_flight(self) -> None:
        cache = runtime_session._ArtifactCache()
        cache.get_or_create("key", lambda: "stale")
        calls = 0
        calls_lock = threading.Lock()
        barrier = threading.Barrier(4)
        started = threading.Event()
        release = threading.Event()

        def replace() -> str:
            nonlocal calls
            with calls_lock:
                calls += 1
            started.set()
            release.wait()
            return "current"

        def consume(_: int) -> str:
            barrier.wait()
            return cache.get_or_create("key", replace, lambda value: value == "current")

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = tuple(executor.submit(consume, index) for index in range(4))
            self.assertTrue(started.wait(timeout=5))
            with cache._partition.lock:
                flight = cache._partition.flights["key"]
                self.assertTrue(flight.condition.wait_for(lambda: flight.participants == 4, timeout=5))
            release.set()
            values = tuple(future.result(timeout=5) for future in futures)

        self.assertEqual(values, ("current",) * 4)
        self.assertEqual(calls, 1)

    def test_compile_failure_is_shared_by_existing_waiters_and_flight_is_removed(self) -> None:
        cache = runtime_session._ArtifactCache()
        barrier = threading.Barrier(4)
        started = threading.Event()
        release = threading.Event()
        calls = 0
        calls_lock = threading.Lock()

        def fail() -> object:
            nonlocal calls
            with calls_lock:
                calls += 1
            started.set()
            release.wait()
            raise ValueError("compile failed")

        def consume(_: int) -> str:
            barrier.wait()
            try:
                cache.get_or_create("key", fail)
            except ValueError as error:
                return str(error)
            raise AssertionError("cache failure was not propagated")

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = tuple(executor.submit(consume, index) for index in range(4))
            self.assertTrue(started.wait(timeout=5))
            with cache._partition.lock:
                flight = cache._partition.flights["key"]
                self.assertTrue(flight.condition.wait_for(lambda: flight.participants == 4, timeout=5))
            release.set()
            errors = tuple(future.result(timeout=5) for future in futures)

        self.assertEqual(errors, ("compile failed",) * 4)
        self.assertEqual(calls, 1)
        self.assertEqual(cache._partition.flights, {})

    def test_cross_session_resource_borrow_fails_before_materialization(self) -> None:
        from vernon_dsl._runtime.binding import _DispatchBorrowLease

        with mock.patch.object(runtime_session, "_construct_state", side_effect=self._state):
            first = vd.RuntimeSession(arch=vd.metal)
            second = vd.RuntimeSession(arch=vd.vulkan)
        storage = vd.storage.zeros(dtype=vd.f32, shape=(1,))

        first_context = runtime_session._InvocationContext(first)
        second_context = runtime_session._InvocationContext(second)
        lease = _DispatchBorrowLease([("storage", storage, "read")], first_context)
        try:
            with self.assertRaisesRegex(RuntimeError, "another RuntimeSession"):
                _DispatchBorrowLease([("storage", storage, "read")], second_context)
            self.assertIsNone(storage._residency.current)
        finally:
            lease.release()

    def test_same_session_materialization_is_single_flight_per_resource(self) -> None:
        class Buffer:
            def upload_ranges(self, ranges: object) -> None:
                pass

        class Host:
            def __init__(self) -> None:
                self.calls = 0

            def create_buffer(self, size: int) -> Buffer:
                self.calls += 1
                time.sleep(0.05)
                return Buffer()

        host = Host()
        configuration = vd.RuntimeConfiguration(vd.metal, None)
        state = runtime_session._RuntimeSessionState(
            SimpleNamespace(capabilities={"available": True}, configuration=configuration),
            host,
            next(runtime_session._session_identities),
        )
        with mock.patch.object(runtime_session, "_construct_state", return_value=state):
            session = vd.RuntimeSession(arch=vd.metal)
        storage = vd.storage.zeros(dtype=vd.f32, shape=(1,))

        def materialize(_: int) -> object:
            context = runtime_session._InvocationContext(session)
            return storage._resident_buffer(context)

        with ThreadPoolExecutor(max_workers=4) as executor:
            buffers = tuple(executor.map(materialize, range(4)))

        self.assertTrue(all(buffer is buffers[0] for buffer in buffers))
        self.assertEqual(host.calls, 1)


if __name__ == "__main__":
    unittest.main()
