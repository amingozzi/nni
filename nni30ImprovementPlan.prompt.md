# NNI 3.0 Audit — Improvement Plan

## Status: Draft — pending user approval

---

## Overview
Full audit of NNI 3.0 covering:
- 3 confirmed critical logic bugs (NAS engine)
- 6 confirmed security vulnerabilities (injection, deserialization, CORS)
- 5 numerical/validation bugs (HPO, quantization, config)
- 6 vision-model capability gaps (compression, NAS search spaces)
- 4 dependency hygiene issues

---

## Phase 1 — Critical Bug Fixes (P0)

### 1.1 Fix `==` vs `in` logic error in MixedConv2d
- File: `nni/nas/oneshot/pytorch/supermodule/operation.py:335`
- `if name == ['kernel_size', 'padding']:` → `if name in ['kernel_size', 'padding']:`
- Impact: mutable kernel_size/padding never handled, silently ignored

### 1.2 Fix division-by-zero in LatencyPenalty
- File: `nni/nas/oneshot/pytorch/profiler.py:151`
- Guard `self.baseline` at construction: raise `ValueError` if `baseline == 0`

### 1.3 Fix GPU index shell injection in Trial
- File: `nni/tools/trial_tool/trial.py:95-100`
- Validate `gpuIndices` with regex `^[0-9,]+$` before embedding in shell command
- Fail with `ValueError` if validation fails

---

## Phase 2 — Security Fixes (P0)

### 2.1 Replace `shell=True` subprocess calls
- Files: `trial.py:100`, `trial_keeper.py:96/98`, `nnictl_utils.py:781`, `command_utils.py:22`
- Use `shlex.split()` + list-form `Popen()` where command is user-controlled (trial commands are inherently user code — document this boundary clearly)
- nnictl internal commands: switch to list args, no shell=True

### 2.2 Restrict pickle/cloudpickle deserialization
- File: `nni/contrib/distillation/utils.py:41-45` — wrap with try/except and document file-trust boundary
- File: `nni/algorithms/feature_engineering/gradient_selector/fginitialize.py:273` — same
- File: `nni/common/serializer.py:945-947` — add comment + type validation after load

### 2.3 Fix command injection in TypeScript backend
- File: `ts/nni_manager/training_service/common/util.ts:87-156`
- Replace string template interpolation in exec calls with properly escaped paths (use `shellEscape` or pass as array to `execFile`)

### 2.4 Fix TOCTOU in temp directory generation
- File: `nni/tools/nnictl/common_utils.py:87`
- Replace with `tempfile.mkdtemp()` for atomic creation

### 2.5 Enable/fix input validation in REST API
- File: `ts/nni_manager/rest_server/restHandler.ts:21-22`
- Re-enable expressJoi or add explicit per-route validation middleware

### 2.6 Restrict CORS in REST server
- File: `ts/nni_manager/rest_server/restHandler.ts:38-41`
- Add origin whitelist (localhost only) unless explicitly configured

---

## Phase 3 — Numerical & Validation Bugs (P1)

### 3.1 Replace `assert` with exceptions in experiment config
- File: `nni/experiment/config/experiment_config.py:104,139,178-182`
- Change all validation `assert` → `if not ...: raise ValueError(...)` to prevent silent bypass with `-O`

### 3.2 Add iteration limit in BOHB NaN replacement loop
- File: `nni/algorithms/hpo/bohb_advisor/config_generator.py:242-257`
- Add `max_iter` counter; break with fallback value if exhausted

### 3.3 Add baseline zero-guard in profiler (see 1.2 above)

### 3.4 InputChoice empty-list `None` return guard
- File: `nni/nas/nn/pytorch/choice.py:366-380`
- For `reduction='none'` on empty list: return `[]` consistently, not `None`

---

## Phase 4 — Vision Model Support (P1)

### 4.1 Extend compression module registry for modern vision ops
- File: `nni/compression/base/setting.py:147-218`
- Add entries for: `MultiheadAttention`, `LayerNorm`, `GroupNorm`, `GELU` activation
- These are required for ViT/BERT/ConvNeXt pruning and quantization

### 4.2 Add MixedMultiheadAttention supermodule
- File: `nni/nas/oneshot/pytorch/supermodule/operation.py`
- New class alongside existing MixedLinear / MixedConv2d for mutable num_heads and embed_dim (ViT head search)

### 4.3 Add ViT-style patch embedding search space helper
- File: `nni/nas/nn/pytorch/` (new module, e.g. `layers_vision.py`)
- Class `MutablePatchEmbedding` wrapping `Conv2d` with mutable `patch_size` and `embed_dim`

### 4.4 MixedDepthwiseConv2d
- File: `nni/nas/oneshot/pytorch/supermodule/operation.py`
- Variant of MixedConv2d where `groups == in_channels`; needed for MobileNet/EfficientNet NAS

### 4.5 Vision NAS examples
- Add example: `examples/nas/vision_transformer/` — ViT search on CIFAR-10
- Add example: `examples/nas/mobilenet_like/` — SPOS/ProxylessNAS with depthwise convs

---

## Phase 5 — Dependency Hygiene (P2)

### 5.1 Update Python dependencies
- File: `dependencies/required.txt`
  - Remove `numpy < 1.22 ; python_version < "3.8"` (EOL Python)
  - Update `filelock < 3.12` → `filelock >= 3.12`
  - Pin `cloudpickle >= 2.0, < 4.0`

### 5.2 Update Node.js dependencies
- File: `ts/nni_manager/package.json`
  - Replace `child-process-promise` (unmaintained since 2018) with native `util.promisify(exec)`
  - Update `azure-storage ^2.10.7` → `@azure/storage-blob` (modern SDK)
  - Update `ssh2` to latest stable

---

## Relevant Files

| File | Phase | Operation |
|---|---|---|
| `nni/nas/oneshot/pytorch/supermodule/operation.py` | 1.1 / 4.2 / 4.4 | Fix bug; add new classes |
| `nni/nas/oneshot/pytorch/profiler.py` | 1.2 | Fix guard |
| `nni/tools/trial_tool/trial.py` | 1.3 / 2.1 | Fix injection |
| `nni/tools/trial_tool/trial_keeper.py` | 2.1 | Fix shell=True |
| `nni/tools/nnictl/nnictl_utils.py` | 2.1 | Fix shell=True |
| `nni/tools/nnictl/command_utils.py` | 2.1 | Fix shell=True |
| `nni/tools/nnictl/common_utils.py` | 2.4 | Fix TOCTOU |
| `nni/contrib/distillation/utils.py` | 2.2 | Document trust boundary |
| `nni/algorithms/feature_engineering/gradient_selector/fginitialize.py` | 2.2 | Document trust boundary |
| `nni/common/serializer.py` | 2.2 | Add type guard |
| `ts/nni_manager/training_service/common/util.ts` | 2.3 | Fix injection |
| `ts/nni_manager/rest_server/restHandler.ts` | 2.5 / 2.6 | Validation + CORS |
| `nni/experiment/config/experiment_config.py` | 3.1 | assert → raise |
| `nni/algorithms/hpo/bohb_advisor/config_generator.py` | 3.2 | Loop guard |
| `nni/nas/nn/pytorch/choice.py` | 3.4 | Empty list guard |
| `nni/compression/base/setting.py` | 4.1 | Registry extension |
| `nni/nas/nn/pytorch/layers_vision.py` | 4.3 | New file |
| `examples/nas/vision_transformer/` | 4.5 | New example |
| `dependencies/required.txt` | 5.1 | Update constraints |
| `ts/nni_manager/package.json` | 5.2 | Update deps |

---

## Verification

1. Run existing test suite: `pytest test/ut/ -x` — verify no regressions
2. Manually test `MixedConv2d` with mutable kernel_size — confirm both paths (tuple and scalar) execute
3. Test `LatencyPenalty(baseline=0)` — confirm `ValueError` raised at construction, not at `forward()`
4. Test trial command with `gpuIndices = "0\" && echo INJECTED"` — confirm rejection
5. Run TypeScript tests in `ts/nni_manager/` — verify no regressions after util.ts changes
6. Construct a small ViT-like model with `MixedMultiheadAttention` and run one-shot search — confirm forward pass works
7. Run compression on a small ViT with LayerNorm — confirm new registry entries are triggered

---

## Decisions / Scope Boundaries

- `shell=True` for trial_command is inherently user-controlled code — it cannot be fully locked down. The fix focuses on the `gpuIndices` injection and nnictl-internal uses.
- DARTS 2nd-order is excluded from scope (significant algorithmic addition, not a bug fix).
- Vision examples target CIFAR-10 scale only (not full ImageNet training).
- `MixedLayerNorm` generator fix (operation.py:570) deferred — requires deeper testing of shape contract.
