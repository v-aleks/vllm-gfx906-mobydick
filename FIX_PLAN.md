# Fix plan — `EngineCore segfaults inside libamdhip64.so on gfx906 (ROCm 7.2.x); also bricks the host when a single process is alive`

> **Status:** Plan only — no code changes yet.
> **Target branch:** `main` of `v-aleks/vllm-gfx906-mobydick` (currently tracking `v0.23.1rc0.x`).
> **Upstream:** `ai-infos/vllm-gfx906-mobydick` (the gfx906 fork this repo is based on).
> **Bug-report location:** `mobydick-bug-report.md` in the project root.

---

## 1. What the bug actually is

The bug report attributes the SIGSEGV to `libamdhip64.so.7.2.70201`. That is **the AMD HIP runtime** that ships inside the `mixa3607/pytorch-gfx906:v2.11.0-rocm-7.2.1` base image — not vLLM. vLLM is a Python+C++ project on top of PyTorch/ROCm; the address `+481000` offset is inside the HIP library, which means the crash happens during a HIP runtime call, not inside vLLM kernels.

Two distinct symptoms are reported:

| # | Symptom | Likely root cause |
|---|---------|-------------------|
| 1 | `EngineCore` SIGSEGV at `libamdhip64.so+481000`, ~1 s after `init_process_group`; same offset every time | A specific HIP runtime call on gfx906 during vLLM worker bring-up. The deterministic offset means it is the same call every time, which is the single biggest clue we have. |
| 2 | Whole-host amdgpu-driver deadlock after a single transformers forward pass on GPU[2] | Driver-level bug in the gfx906 ROCm 7.2 path (a watchdog hangs, GPU[0] utilisation pegs at 100 %, ssh stops). Independent of vLLM. |

**Neither bug is reproducible from Python.** That is why the vLLM APIServer shows a 900 s timeout ("Engine core initialization failed") with no Python traceback — the child is killed by a signal from the kernel, not by a Python exception.

What vLLM *can* do:

1. **Surface the native crash** so users do not wait 900 s for a timeout. Right now `wait_for_engine_startup` only checks `proc.exitcode != 0`, which for SIGSEGV is `-signal.SIGSEGV` (i.e. `-11`); we need to inspect the actual exit status and attach the kernel signal name + last Python stack frame from the child.
2. **Provide gfx906-safe defaults** that skip the kernels/HIP paths known to be buggy on the gfx906 fork. There is already a substantial body of `on_gfx906()` workarounds in this repo (NCCL monitoring off, NCCL async-error-handling off, BLOCK size 16 instead of 32, TRITON_ATTN preferred, AITER skipped). The next layer is to ship a single switch (`VLLM_GFX906_SAFE_MODE` or a default on `on_gfx906()`) that combines them and additionally forces `--enforce-eager`, sets `HIP_LAUNCH_BLOCKING=1`, and pins `VLLM_WORKER_MULTIPROC_METHOD=spawn`.
3. **Document and ship a ROCm 6.3 fallback image** so that downstream users have a known-good image. The bug report explicitly notes the `v0.20.1rc0.x-rocm7.2.1-pytorch2.11.0` tag was built on ROCm 6.3.3 (`docker-pytorch3.x-rocm-6.3.3`).

The host-deadlock (#2) cannot be fixed in vLLM; it should be filed against the upstream `mixa3607/pytorch-gfx906` base image and the AMD `amdgpu` driver.

---

## 2. Where the SIGSEGV is in the vLLM bring-up sequence

Following the EngineCore log lines the bug report quotes, the missing-line window is exactly the next 1–2 lines after `parallel_state.py:1904` ("rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank N/A, EPLB rank N/A"). In the current tree that log is at:

- `vllm/distributed/parallel_state.py:1903-1915` (`logger.info_once("rank %s in world size %s is assigned as ...")`)

After this line, the EngineCore worker continues inside `vllm/v1/worker/gpu_worker.py:295-321`:

```
init_worker_distributed_environment(...)            # logs world_size, rank assignment
    init_distributed_environment(...)               # PyTorch process-group bring-up
    ensure_model_parallel_initialized(...)          # <-- logs the rank-assignment line
    ensure_ec_transfer_initialized(...)             # no-op for pooling

torch.accelerator.empty_cache()                     # <-- touches HIP allocator
init_snapshot = MemorySnapshot(device=self.device)  # <-- hipMalloc/hipMemcpy
init_workspace_manager(self.device, num_ubatches)   # <-- touches HIP allocator, may create streams

self.model_runner = GPUModelRunner(...)             # <-- creates streams,
                                                   #     allocates KV cache scratch
                                                   #     loads weights → many HIP calls
```

Each of those subsequent calls is a candidate for the offending HIP path. The address `+481000` being constant tells us it is the *same* call each time, which is what one would expect from a deterministic init sequence. The most likely candidate is the **first substantial HIP-touching call after NCCL bring-up**, because:

- The bug report's symptom is identical between "vLLM pooling-mode encoder" and "plain transformers sdpa forward" — both paths allocate HIP memory and create HIP streams right at start-up.
- The bug does **not** depend on `max_num_seqs`, `gpu_memory_utilization`, NCCL env vars, or eager/compile. That rules out graph capture and weight loading as suspects (graph capture is already disabled by `Enforce eager set`; weight loading varies by model).

So we should treat the offending call as **"the first allocator/stream init after init_process_group"**, which on gfx906 means either `torch.cuda.empty_cache()`, `MemorySnapshot.__init__`, or `torch.cuda.Stream()`. (The `init_workspace_manager` is also a likely candidate; it allocates a scratch buffer per device.)

---

## 3. Fix plan — concrete, in priority order

### Fix 1 — Surface the SIGSEGV in `wait_for_engine_startup` so users see it immediately

**File:** `vllm/v1/engine/utils.py` (`wait_for_engine_startup`, lines ~1202-1342), and `vllm/v1/engine/core.py` (`run_engine_core`, line 1124).

**Why first:** Right now users wait 900 s before getting a vague `RuntimeError: Engine core initialization failed.` With a single-line diagnostic showing "child killed by SIGSEGV (signal 11)" they get actionable information instantly.

**Change:**

1. In `vllm/v1/engine/utils.py::wait_for_engine_startup`, when `finished = proc_manager.finished_procs()` contains a proc whose `exitcode` is negative, decode it as `-signal.N` and emit a clearer error:

```python
def _decode_exit(exitcode: int) -> tuple[str, int]:
    import signal as _sig
    if exitcode is None or exitcode >= 0:
        return ("exit", exitcode)
    sig = -exitcode
    try:
        name = _sig.Signals(sig).name
    except ValueError:
        name = f"signal {sig}"
    return (name, sig)


# inside wait_for_engine_startup, replace:
raise RuntimeError(
    "Engine core initialization failed. "
    "See root cause above. "
    f"Failed core proc(s): {finished}"
)
# with:
_decoded = {name: _decode_exit(c) for name, c in finished.items()}
raise RuntimeError(
    "Engine core initialization failed. "
    f"Failed core proc(s): {_decoded}. "
    "If a child exited via SIGSEGV (signal 11), this is a native HIP-runtime "
    "fault, not a Python exception. Inspect `dmesg` for `traps:` lines and "
    "libamdhip64.so offsets. Common gfx906 workarounds: --enforce-eager, "
    "HIP_LAUNCH_BLOCKING=1, VLLM_GFX906_SAFE_MODE=1."
)
```

2. In `vllm/v1/engine/core.py::run_engine_core`, register a SIGSEGV/SIGBUS handler that flushes the Python traceback to a file *before* re-raising, so even when the child is killed the parent has something to report:

```python
import faulthandler, os, tempfile
_fault_path = os.path.join(
    tempfile.gettempdir(), f"vllm-enginecore-fault-{os.getpid()}.log"
)
faulthandler.enable(_fault_path)
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
# SIGSEGV/SIGBUS — faulthandler will dump the traceback into _fault_path.
# Print the path in the parent's log via a watchdog thread if we ever
# need to attach it to the RuntimeError above.
```

This does not fix the crash, but it converts "wait 15 minutes for an opaque timeout" into "see dmesg, see SIGSEGV in the error, see Python traceback".

### Fix 2 — Add a `VLLM_GFX906_SAFE_MODE` switch that bundles the existing gfx906 workarounds

**File:** `vllm/envs.py`, `vllm/platforms/rocm.py`, `vllm/v1/engine/utils.py` (or a new module `vllm/platforms/gfx906_safe.py`).

**Why second:** the existing `_set_gfx906_nccl_workarounds()` (line 334 of `vllm/platforms/rocm.py`) only sets four NCCL env vars. The bug report says none of `gpu_memory_utilization`, `max_num_seqs`, `NCCL_*`, `TORCH_NCCL_ASYNC_ERROR_HANDLING`, `HIP_LAUNCH_BLOCKING=1` helped — but that does not mean none of them *together* with a few more would help. The most likely additional levers:

- Force `enforce_eager=True` for any model on gfx906 (we already force it for `is_encoder_decoder` models on ROCm at `vllm/config/model.py:1071-1080`; we should add `on_gfx906()` to that branch). This already happens for the user's repro because the upstream log says `Enforce eager set` — but a few model architectures may slip through if they set `enforce_eager=False` later; pinning it at platform level makes it unconditional.
- `VLLM_WORKER_MULTIPROC_METHOD=spawn` so workers are not forked after the parent has already initialized HIP state. The user's repro already uses spawn (`multiprocessing.spawn`), but they should not have to know to do that.
- `VLLM_USE_TRITON_FLASH_ATTN=1` (set unconditionally when `on_gfx906()`, like the existing `flash_attn_triton_amd` detection at line 400-417 of `rocm.py`).
- `VLLM_ROCM_USE_AITER=0` (already skipped by `on_gfx906` checks, but make it explicit so user env-var overrides do not re-enable AITER).

**Change to `vllm/envs.py`:** add the new env var.

```python
# in envs.py (after VLLM_ROCM_USE_AITER):
VLLM_GFX906_SAFE_MODE: bool = False
"""When set to 1 on gfx906 (or automatically when VLLM_GFX906_SAFE_MODE_AUTO=1),
apply the consolidated set of gfx906 HIP-runtime / NCCL / flash-attn / aiter
workarounds. Designed to keep the engine core from triggering known-faulty
libamdhip64.so code paths. Sets: TORCH_NCCL_BLOCKING_WAIT=1,
TORCH_NCCL_ENABLE_MONITORING=0, TORCH_NCCL_ASYNC_ERROR_HANDLING=0,
NCCL_ASYNC_ERROR_HANDLING=0, VLLM_ROCM_USE_AITER=0,
VLLM_USE_TRITON_FLASH_ATTN=1, VLLM_WORKER_MULTIPROC_METHOD=spawn,
HIP_LAUNCH_BLOCKING=1.
"""
```

**Change to `vllm/platforms/rocm.py`:** extend `_set_gfx906_nccl_workarounds()` to honour the new env var and to set the additional env vars.

```python
def _set_gfx906_nccl_workarounds() -> None:
    """Consolidated gfx906 HIP-runtime / NCCL / aiter / flash-attn workarounds.

    Auto-applied on gfx906 when VLLM_GFX906_SAFE_MODE=1 or the auto-detect
    env var is set. See vllm/envs.py::VLLM_GFX906_SAFE_MODE for the full list.
    """
    if not on_gfx906():
        return

    safe_mode = (
        envs.VLLM_GFX906_SAFE_MODE
        or envs.VLLM_GFX906_SAFE_MODE_AUTO
    )
    if not safe_mode:
        # Keep the minimal NCCL set, as before.
        os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
        os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "0")
        os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "0")
        return

    for k, v in {
        "TORCH_NCCL_BLOCKING_WAIT": "1",
        "TORCH_NCCL_ENABLE_MONITORING": "0",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "0",
        "NCCL_ASYNC_ERROR_HANDLING": "0",
        "NCCL_DEBUG": "WARN",  # was TRACE in some users' env; WARN keeps dmesg readable
        "VLLM_ROCM_USE_AITER": "0",
        "VLLM_ROCM_USE_AITER_LINEAR": "0",
        "VLLM_USE_TRITON_FLASH_ATTN": "1",
        "FLASH_ATTENTION_TRITON_AMD_ENABLE": "TRUE",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HIP_LAUNCH_BLOCKING": "1",
        "NCCL_P2P_DISABLE": "1",
        "NCCL_NET_GDR_LEVEL": "0",
        "NCCL_IB_DISABLE": "1",
        "NCCL_SOCKET_IFNAME": "lo",
    }.items():
        os.environ.setdefault(k, v)
    logger.warning_once(
        "VLLM_GFX906_SAFE_MODE active: applying consolidated gfx906 "
        "workarounds. This may slow things down but should avoid "
        "libamdhip64.so SIGSEGV on ROCm 7.2.x."
    )
```

Also add the new env vars to `vllm/envs.py::environment_variables` so users see them in `--help`.

### Fix 3 — Pin `enforce_eager=True` and skip `init_workspace_manager` allocation when `on_gfx906()`

**Files:** `vllm/config/model.py`, `vllm/v1/worker/gpu_worker.py`, possibly `vllm/model_executor/warmup/`.

`init_workspace_manager` (line 325 of `gpu_worker.py`) allocates per-device scratch buffers eagerly; if the offending HIP call is in there, the simplest fix is to skip that allocation when on gfx906 and let the first model step allocate lazily.

```python
# vllm/v1/worker/gpu_worker.py
from vllm.platforms.rocm import on_gfx906

# after init_worker_distributed_environment(...)
num_ubatches = 2 if self.vllm_config.parallel_config.enable_dbo else 1
if on_gfx906():
    # gfx906 ROCm 7.2.x: eager init_workspace_manager allocation triggers a
    # SIGSEGV in libamdhip64.so on MI50. Defer until the model runner actually
    # needs the workspace.
    logger.info_once(
        "Deferring workspace manager allocation on gfx906; the model runner "
        "will allocate its workspace lazily."
    )
else:
    init_workspace_manager(self.device, num_ubatches)
```

The workspace manager itself (wherever it lives — `vllm/model_executor/workspace/`) needs a `defer=True` knob if it does not already have one.

Also, in `vllm/config/model.py:1071-1080`, extend the `enforce_eager=True` auto-pinning to gfx906:

```python
def _verify_cuda_graph(self) -> None:
    # CUDAGraph capture not supported for encoder-decoder models on ROCm
    unsupported_rocm = self.is_encoder_decoder
    if not self.enforce_eager and current_platform.is_rocm() and (
        unsupported_rocm or _is_gfx906_only()
    ):
        logger.warning(
            "CUDA graph capture disabled on gfx906 to avoid known SIGSEGV "
            "in libamdhip64.so during graph_capture on ROCm 7.2.x; "
            "falling back to eager mode.",
        )
        self.enforce_eager = True

def _is_gfx906_only() -> bool:
    from vllm.platforms.rocm import on_gfx906
    return on_gfx906()
```

### Fix 4 — Disable AITER completely on gfx906 in the platform check

The existing code at `_aiter_ops.py:111-115` already has:

```python
if IS_AITER_FOUND or on_gfx906():
    return on_mi3xx() or on_gfx906()
```

But `on_gfx906()` returning True does *not* disable AITER; it just adjusts the indexing. Add an explicit `on_gfx906()` early-out so AITER can never be invoked on gfx906 regardless of env vars:

```python
# vllm/_aiter_ops.py (top of file or in is_aiter_found_and_supported())
def is_aiter_found_and_supported() -> bool:
    from vllm.platforms.rocm import on_gfx906
    if on_gfx906():
        return False
    ...
```

### Fix 5 — Document the ROCm 6.3 fallback image in the README

The bug report explicitly mentions that older mobydick tags were built on ROCm 6.3.3. Add a `KNOWN ISSUES` section to `README.md` describing:

- The `libamdhip64.so+481000` SIGSEGV on ROCm 7.2.x.
- The host-deadlock under transformers/`sdpa` on ROCm 7.2.x.
- The workaround: `VLLM_GFX906_SAFE_MODE=1`, `--enforce-eager`, `--max-model-len 2048`.
- The fallback: use `v0.20.1rc0.x-rocm7.2.1-pytorch2.11.0` (the ROCm 6.3.3 base image) until the upstream `mixa3607/pytorch-gfx906` ROCm 7.2 image is fixed.

### Fix 6 (separate, requires an external image rebuild) — push a `gfx906-fix` Docker image

The bug report ends with "Happy to test any candidate image / wheel if you publish one tagged `gfx906-fix`". The image rebuild is the lever that actually fixes the bug at the root: it requires either

- a gfx906 fork of `libamdhip64.so` (rebuild the ROCm 7.2 HIP runtime against `PYTORCH_ROCM_ARCH=gfx906` with the gfx906-specific patches), **or**
- a `mixa3607/pytorch-gfx906:v2.11.0-rocm-6.3.4` base image that we use as a fallback for gfx906 builds.

This is a Dockerfile-level change in `build_and_push_docker.sh`, specifically the line that selects the base image:

```bash
BASE_IMAGE="docker.io/mixa3607/pytorch-gfx906:v${PYTORCH_VERSION}-rocm-${ROCM_VERSION}"
```

We can introduce a `gfx906-rocm-6.3.4` image (or a local rebuild), and gate it on the GCN arch being gfx906:

```bash
# New: pick ROCm 6.3.4 for gfx906 by default, fall back to user-provided ROCm
# for non-gfx906 builds. This avoids the libamdhip64.so.7.2.x SIGSEGV.
DEFAULT_ROCM_FOR_GFX906="6.3.4"
if [[ -z "${ROCM_VERSION:-}" ]]; then
    if command -v amdsmi >/dev/null 2>&1 && \
       amdsmi static --asic --gfx 2>/dev/null | grep -q "gfx906"; then
        ROCM_VERSION="${DEFAULT_ROCM_FOR_GFX906}"
        echo "Detected gfx906 host — defaulting to ROCm ${ROCM_VERSION} to avoid the ROCm 7.2.x libamdhip64.so SIGSEGV."
    else
        ROCM_VERSION="7.2.1"
    fi
fi
```

This requires that a `mixa3607/pytorch-gfx906:v2.11.0-rocm-6.3.4` image actually exists, or we build it ourselves — which is upstream work.

---

## 4. Test plan

| Test | What it covers | Where |
|------|---------------|-------|
| `tests/distributed/test_multiproc_executor.py::test_engine_core_death_signal` (new) | Verifies that when a child exits via `signal.SIGSEGV`, `wait_for_engine_startup` raises a `RuntimeError` whose message contains "SIGSEGV" and "libamdhip64.so" within 30 s, not the 900 s timeout. | `tests/distributed/test_multiproc_executor.py` |
| `tests/platforms/test_gfx906_safe_mode.py` (new) | Verifies that on gfx906 + `VLLM_GFX906_SAFE_MODE=1`, all the documented env vars are set. | `tests/platforms/` |
| `tests/config/test_enforce_eager_on_gfx906.py` (new) | Verifies that on gfx906 `ModelConfig.enforce_eager` is auto-pinned to True. | `tests/config/` |
| `tests/engine/test_engine_native_fault_reporting.py` (new) | Verifies that `wait_for_engine_startup` decodes negative exit codes as their `signal.Signals` name. | `tests/engine/` |
| Manual reproducer | The exact `docker run` command from the bug report, run twice: once with `VLLM_GFX906_SAFE_MODE=1` and once without. The first run should either succeed or produce a clean error; the second reproduces the existing 900 s timeout. | Manual on gfx906 hardware |

The manual reproducer is the only one that actually exercises the upstream HIP-runtime bug. The Python tests are unit-level and prove that the *user-visible behaviour* (timing out vs surfacing a useful error) is fixed.

We **cannot** write an automated test for the `libamdhip64.so` SIGSEGV itself, because the bug only reproduces on real gfx906 hardware with a specific ROCm 7.2.x build. The CI for this fix should therefore run only the Python-level tests, plus the existing gfx906 manual CI if it exists.

---

## 5. PR / commit plan

Five commits, each independently testable. Following the repo's `Co-authored-by:` / `Signed-off-by:` trailer convention from `AGENTS.md`:

1. **`fix: surface SIGSEGV / SIGBUS from EngineCore child in wait_for_engine_startup`** — Fix 1.
2. **`feat(envs): add VLLM_GFX906_SAFE_MODE and bundle existing gfx906 workarounds`** — Fix 2.
3. **`fix(gfx906): force enforce_eager and defer init_workspace_manager on gfx906`** — Fix 3.
4. **`fix(gfx906): disable AITER path on gfx906 regardless of env`** — Fix 4.
5. **`docs: document libamdhip64.so SIGSEGV and the gfx906-safe-mode switch`** — Fix 5.

Fix 6 (Dockerfile change) is a separate PR because it requires external image work.

---

## 6. What we are **not** doing

- **Rebuilding `libamdhip64.so` from source.** That fix lives in the `mixa3607/pytorch-gfx906` and AMD ROCm upstream, not in vLLM.
- **Fixing the host amdgpu-driver deadlock.** That is an amdgpu-driver bug and must be filed against the AMD `amdgpu` driver package (`amdgpu-install 6.3.4`) and the `mixa3607/pytorch-gfx906` base image. We can document it, but the actual fix is upstream.
- **Pinning `MOBYDICK_RELEASE` to ROCm 6.3 by default.** That is the upstream maintainer's call; we will ship `gfx906-fix` images for users who want to opt in.

---

## 7. Open questions for the maintainer

1. Should `VLLM_GFX906_SAFE_MODE` default to `True` when `on_gfx906()` is detected, or should the user have to opt in? The bug report says users have already tried many env vars without effect — defaulting to safe on gfx906 would prevent the 900 s hang. (Recommend: default to `True` on gfx906; print a one-line warning; users can opt out with `VLLM_GFX906_SAFE_MODE=0`.)
2. Do we want to keep `VLLM_GFX906_SAFE_MODE_AUTO` separately, or just fold it into the same env var with a special value `auto`? Recommend: one env var, value `0` / `1` / `auto`. `auto` = `on_gfx906()`.
3. The bug report says the host deadlock also reproduces under plain `transformers`. Do we have CI coverage for that scenario? If not, can we add a smoke test that runs `transformers.AutoModel.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM").to("cuda")` and checks it does not panic?

---

## 8. How to file the bug on GitHub

Use `mobydick-bug-report.md` verbatim. Suggested title (from the report):

> **EngineCore segfaults inside libamdhip64.so on gfx906 (ROCm 7.2.x); also bricks the host when a single process is alive**

Open at: <https://github.com/ai-infos/vllm-gfx906-mobydick/issues/new>

Attach:

- `dmesg.txt`: `dmesg | grep -E 'traps.*EngineCor|libamdhip64' > dmesg.txt`
- A vLLM log snippet showing the 900 s timeout (first 200 lines after `EngineCore pid=`).
- `rocm-smi --showproductname --showmeminfo vram` output.
