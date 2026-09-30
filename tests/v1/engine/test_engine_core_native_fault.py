# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for EngineCore native-fault diagnostics (gfx906 ROCm 7.2.x).

These tests import the specific helper modules directly to avoid pulling in
the full vLLM package (which requires torch, transformers, pillow, etc.).

Background: a child EngineCore process can be killed by a native signal
(notably SIGSEGV from libamdhip64.so on gfx906 ROCm 7.2.x) and never
produce a Python traceback. We want to make sure:

* ``_decode_exit_code`` correctly translates the negative exitcode that
  ``multiprocessing`` exposes for signal-killed children back to the
  signal name (e.g. SIGSEGV).
* ``finished_procs_with_reasons`` returns the decoded reasons for every
  finished proc, in the format the new RuntimeError expects.
* ``_log_native_fault_if_present`` logs an ERROR line when a child was
  killed by a signal, and is a no-op when only normal exits are present.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import signal
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Load vllm/v1/engine/utils.py as a stand-alone module to avoid pulling in
# the full vLLM package (which transitively imports torch, transformers,
# pillow, etc.). We stub out the heavy dependencies with minimal stand-ins.
# ---------------------------------------------------------------------------


def _stub_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# Stub the imports that utils.py needs at module-load time.
_stub_module("vllm", PlatformEnum=None)
_stub_module("vllm.envs", VLLM_USE_V1=1)
_stub_module("vllm.logger", init_logger=lambda name: logging.getLogger(name))
_stub_module(
    "vllm.platforms",
    current_platform=types.SimpleNamespace(is_cuda_alike=lambda: False, is_xpu=lambda: False),
)
_stub_module(
    "vllm.ray.ray_env",
    get_env_vars_to_copy=lambda destination=None, exclude_vars=None: [],
)
_stub_module(
    "vllm.utils",
    numa_utils=types.SimpleNamespace(configure_subprocess=lambda *a, **kw: _stub_module(
        "vllm.utils.numa_utils"
    ) or __import__("contextlib").nullcontext()),
)
_stub_module("vllm.utils.network_utils", **{"get_open_port": lambda: 0, "get_tcp_uri": lambda *a, **kw: "", "get_open_zmq_ipc_path": lambda: "", "zmq_socket_ctx": lambda *a, **kw: __import__("contextlib").nullcontext()})
_stub_module("vllm.utils.system_utils", get_mp_context=lambda: __import__("multiprocessing").get_context())
_stub_module(
    "vllm.v1.engine.coordinator",
    DPCoordinator=type("DPCoordinator", (), {}),
)
_stub_module(
    "vllm.v1.executor",
    Executor=type("Executor", (), {}),
)
_stub_module(
    "vllm.v1.executor.ray_utils",
    WORKER_SPECIFIC_ENV_VARS=[],
)
_stub_module(
    "vllm.v1.utils",
    get_engine_client_zmq_addr=lambda *a, **kw: "",
    shutdown=lambda *a, **kw: None,
)
_stub_module("zmq", **{"Context": type("Context", (), {}), "ROUTER": 0, "DEALER": 0, "PAIR": 0, "POLLIN": 0, "Poller": type("Poller", (), {"register": lambda self, *a, **kw: None, "poll": lambda self, *a, **kw: []}), "Socket": type("Socket", (), {})})
_stub_module(
    "msgspec",
    **{"msgpack": types.SimpleNamespace(encode=lambda x: b"", decode=lambda b: {}), "Struct": type("Struct", (), {})},
)

# Stub config imports for the type annotations only.
_stub_module("vllm.config", **{"CacheConfig": type("CacheConfig", (), {}), "ParallelConfig": type("ParallelConfig", (), {}), "VllmConfig": type("VllmConfig", (), {})})


def _load_utils_module():
    src_path = Path(__file__).resolve().parents[3] / "vllm" / "v1" / "engine" / "utils.py"
    spec = importlib.util.spec_from_file_location("vllm.v1.engine.utils", src_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


utils_module = _load_utils_module()
_decode_exit_code = utils_module._decode_exit_code
_log_native_fault_if_present = utils_module._log_native_fault_if_present
_NATIVE_FAULT_HINT = utils_module._NATIVE_FAULT_HINT


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDecodeExitCode:
    def test_none_returns_not_finished(self):
        assert _decode_exit_code(None) == ("not_finished", -1)

    def test_zero_is_normal_exit(self):
        assert _decode_exit_code(0) == ("exit", 0)

    def test_positive_is_normal_exit(self):
        assert _decode_exit_code(1) == ("exit", 1)
        assert _decode_exit_code(42) == ("exit", 42)

    @pytest.mark.parametrize(
        "sig,expected_name",
        [
            (signal.SIGSEGV, "SIGSEGV"),
            (signal.SIGABRT, "SIGABRT"),
            (signal.SIGBUS, "SIGBUS"),
            (signal.SIGFPE, "SIGFPE"),
        ],
    )
    def test_negative_is_signal(self, sig, expected_name):
        reason, code = _decode_exit_code(-sig)
        assert reason == expected_name
        assert code == sig

    def test_unknown_signal_falls_back_to_signal_n(self):
        reason, code = _decode_exit_code(-73)
        assert reason == "signal_73"
        assert code == 73


class TestFinishedProcsWithReasons:
    """Test the CoreEngineProcManager.finished_procs_with_reasons method."""

    def test_returns_decoded_reasons(self):
        # Two fake procs: one normal exit, one SIGSEGV.
        proc_ok = MagicMock()
        proc_ok.name = "EngineCore_OK"
        proc_ok.exitcode = 0

        proc_segfault = MagicMock()
        proc_segfault.name = "EngineCore_DEAD"
        proc_segfault.exitcode = -signal.SIGSEGV

        proc_running = MagicMock()
        proc_running.name = "EngineCore_RUNNING"
        proc_running.exitcode = None

        manager = utils_module.CoreEngineProcManager.__new__(utils_module.CoreEngineProcManager)
        manager.processes = [proc_ok, proc_segfault, proc_running]

        result = manager.finished_procs_with_reasons()
        assert result == {
            "EngineCore_OK": ("exit", 0),
            "EngineCore_DEAD": ("SIGSEGV", signal.SIGSEGV),
        }


class TestLogNativeFaultIfPresent:
    def test_no_signal_is_noop(self, caplog):
        caplog.set_level(logging.ERROR)
        _log_native_fault_if_present({"a": ("exit", 0), "b": ("exit", 1)})
        assert not any(r.levelno == logging.ERROR for r in caplog.records)

    def test_sigsegv_is_logged(self, caplog):
        caplog.set_level(logging.ERROR)
        _log_native_fault_if_present({"x": ("SIGSEGV", signal.SIGSEGV)})
        msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert any("SIGSEGV" in m for m in msgs)
        assert any("libamdhip64.so" in m for m in msgs)

    def test_sigabrt_is_logged(self, caplog):
        caplog.set_level(logging.ERROR)
        _log_native_fault_if_present({"x": ("SIGABRT", signal.SIGABRT)})
        msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert any("SIGABRT" in m for m in msgs)

    def test_mixed_reasons_logs_only_signals(self, caplog):
        caplog.set_level(logging.ERROR)
        _log_native_fault_if_present(
            {"ok": ("exit", 0), "dead": ("SIGSEGV", signal.SIGSEGV)}
        )
        msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert any("SIGSEGV" in m for m in msgs)
        assert not any(m.strip().startswith("ok") for m in msgs)


def test_native_fault_hint_mentions_gfx906():
    """The RuntimeError hint string should help users find the right knob."""
    assert "VLLM_GFX906_SAFE_MODE" in _NATIVE_FAULT_HINT
    assert "6.3" in _NATIVE_FAULT_HINT
    assert "dmesg" in _NATIVE_FAULT_HINT
    assert "SIGSEGV" in _NATIVE_FAULT_HINT


class TestGfx906SafeModeEnvVar:
    """Verify the env var round-trips through envs."""

    def test_default_is_auto(self, monkeypatch):
        # Re-import envs with the env var unset.
        monkeypatch.delenv("VLLM_GFX906_SAFE_MODE", raising=False)
        # The default in envs.py is "auto" - we don't actually need to
        # import the real vllm.envs, just check the literal.
        import importlib

        # The simplest check: the default value declared in envs.py.
        import re

        src = Path(__file__).resolve().parents[3] / "vllm" / "envs.py"
        text = src.read_text(encoding="utf-8")
        m = re.search(
            r'VLLM_GFX906_SAFE_MODE:\s*str\s*=\s*"([^"]+)"', text
        )
        assert m is not None
        assert m.group(1) == "auto"

    def test_force_eager_default_is_true(self):
        import re

        src = Path(__file__).resolve().parents[3] / "vllm" / "envs.py"
        text = src.read_text(encoding="utf-8")
        m = re.search(
            r'VLLM_GFX906_FORCE_EAGER:\s*bool\s*=\s*(True|False)', text
        )
        assert m is not None
        assert m.group(1) == "True"

    def test_force_eager_can_be_disabled(self, monkeypatch):
        # We don't import vllm.envs because it pulls in everything; instead
        # we verify that the lambda default is parseable as a bool.
        import re

        src = Path(__file__).resolve().parents[3] / "vllm" / "envs.py"
        text = src.read_text(encoding="utf-8")
        m = re.search(
            r'"VLLM_GFX906_FORCE_EAGER":\s*lambda:\s*bool\(\s*int\('
            r'os\.getenv\("VLLM_GFX906_FORCE_EAGER",\s*"(\d)"\)\)\)',
            text,
        )
        assert m is not None
        assert m.group(1) == "1"


class TestResolveGfx906SafeMode:
    """Verify the rocm.py helper correctly parses VLLM_GFX906_SAFE_MODE."""

    def _load_rocm_safe_mode(self):
        """Load only the safe-mode helper from vllm/platforms/rocm.py."""
        src_path = (
            Path(__file__).resolve().parents[3] / "vllm" / "platforms" / "rocm.py"
        )
        text = src_path.read_text(encoding="utf-8")

        # Extract the _resolve_gfx906_safe_mode function body.
        # We exec the function in a controlled namespace with on_gfx906
        # replaced by a mock.
        import re as _re

        m = _re.search(
            r"def _resolve_gfx906_safe_mode\(\).*?(?=\n\n_GFX906_SAFE_MODE_KEYS|\nclass |\ndef )",
            text,
            _re.DOTALL,
        )
        assert m is not None
        return m.group(0)

    @pytest.mark.parametrize(
        "raw,on_gfx906,expected",
        [
            ("1", False, True),
            ("true", False, True),
            ("0", False, False),
            ("false", False, False),
            ("auto", True, True),
            ("auto", False, False),
            ("", True, True),  # any unknown on gfx906 -> apply
            ("garbage", False, False),  # unknown on non-gfx906 -> don't apply
        ],
    )
    def test_resolution(self, monkeypatch, raw, on_gfx906, expected):
        body = self._load_rocm_safe_mode()
        monkeypatch.setenv("VLLM_GFX906_SAFE_MODE", raw)
        namespace: dict = {
            "envs": types.SimpleNamespace(
                VLLM_GFX906_SAFE_MODE=os.environ["VLLM_GFX906_SAFE_MODE"]
            ),
            "on_gfx906": lambda: on_gfx906,
        }
        exec(body, namespace)
        assert namespace["_resolve_gfx906_safe_mode"]() is expected
