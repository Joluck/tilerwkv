# FlashRWKV2

FlashRWKV2 is a TileLang operator library for RWKV-7 with a shared BF16
training and inference forward, FP32 accumulation, and FP32 recurrent state. Its public
inference API exposes the highest semantic fusion islands used by the model;
kernel composition and shape-dependent dispatch remain internal. Complete
models, schedulers, and training frameworks remain the responsibility of
downstream projects.

An optional FP16-state inference mode halves state storage while keeping BF16
token IO and FP32 arithmetic. It does not promise bitwise equality with FP32
training. The default FP32-state path retains training precision and parity.

## Requirements

- Python 3.10 or newer
- Linux, a writable FlashRWKV2 cache, and a CUDA GPU with Compute
  Capability 8.0 or newer
- A system CUDA Toolkit compatible with PyTorch, a C++20 host compiler,
  and Ninja
- `uv`, using the repository-local `./.venv`

The runtime supports PyTorch 2.11 through 2.14 and pins TileLang 0.1.15.
Importing the package does not compile kernels. TileLang specializes model
operators for the active GPU and caches compiled kernels. Packed metadata
validation and sampling use a small ATen/pybind extension, compiled from the
packaged native sources for the Compute Capability of the visible GPUs.
All GPUs visible to that extension must have the same Compute Capability.

## Installation

The project publishes a universal `py3-none-any` source wheel containing only
Python and CUDA/C++ sources. It contains no precompiled extension or architecture
fallback. Building the wheel does not import Torch or run NVCC:

```bash
python -m pip install --pre FlashRWKV2
```

To prewarm the metadata and sampling extension cache:

```bash
python -m flashrwkv2.compile
```

The command emits JSON containing `status` (`compiled` or `cached`), `target`,
`cache_key`, and `library`. Build failures preserve the original compiler
diagnostics; FlashRWKV2 does not download a compiler or select another backend.

For development from a checkout:

```bash
git clone https://github.com/rwkv-rs/FlashRWKV2.git
cd FlashRWKV2
uv sync
./.venv/bin/python -m pip install -v --no-build-isolation -e .
```

## Kernel API

See the [Kernel API reference](https://github.com/rwkv-rs/FlashRWKV2/blob/main/docs/kernel_api.md)
for the complete public operator surface and tensor contracts.

Training operators support `torch.compile(fullgraph=True)`, including backward,
LoRA/NoRA, and CUDA Graph capture. Preserve the BF16 intermediate rounding
contract with Inductor's precision option:

```python
compiled_model = torch.compile(
    model, fullgraph=True, options={"emulate_precision_casts": True},
)
```

Prepare tuning with eager forward-and-backward calls before compiling or
capturing. StateTune chunk metadata is prepared once with
`prepare_tmix_wkv7_statetune_metadata` and supplied through `prepared_metadata`;
see the [training contract](docs/kernel_api.md#training).

## Local changes in this fork

This tree carries four additions on top of `0.1.0a18`. All of them are additive:
default calls keep the released behaviour.

### Recurrent-state checkpoints (speculative decoding)

The WKV-7 inference kernels can write the per-token recurrent state while they
run, so a speculative decoder can rewind to any accepted prefix without
replaying it:

```python
out, boundaries = infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
    r, w, k, v, a, b,
    state=state,
    cu_seqlens=cu,
    state_indices=indices,
    save=True,              # new: write per-token state checkpoints
    checkpoint_interval=1,  # new: token stride, kernel default is 16
)
```

- `boundaries` has shape `(n // checkpoint_interval + batch, heads, head_dim,
  head_dim)` and is indexed as `CU[seq] + seq + offset`; entry `i` is the state
  **before** token `i`, i.e. the state after `i` consumed tokens.
- `save=True` returns `(output, boundaries)` instead of `output`.
- `checkpoint_interval` is now a kernel constexpr. The training path keeps the
  module default of 16, so its forward and the backward checkpoint replay are
  unchanged.

### Post-norm shift-state export

`infer_tmix_postnorm_tokenshift_forward_varlen` and `infer_cmix_forward_varlen`
accept `save_normalized=` with a `(rows, channels)` BF16 buffer:

```python
residual, xr, xw, xk, xv, xa, xg = infer_tmix_postnorm_tokenshift_forward_varlen(
    x, res, weight, bias, x_r, x_w, x_k, x_v, x_a, x_g,
    shift_state_pool=pool,
    cu_seqlens=cu,
    state_indices=indices,
    save_normalized=buffer,   # new: per-token post-norm input
)
```

`buffer[row]` is exactly the value the kernel leaves in the shift-state pool for
a sequence that ends at `row`, which makes the token-shift state rewindable to
the same granularity as the recurrent state.

### H100 compile target

`_internal/linear.py` and `tmix/wkv_prepare/tilelang.py` select `sm_90a` for
`arch == 90`. H100 exposes `sm_90`, but the warpgroup kernels require the `a`
suffix, so the previous target failed to compile on H100.

### `flashrwkv2.mtp.DraftLoop`

`flashrwkv2/mtp` adds a static-buffer light-MTP draft loop. Every intermediate
lives in a preallocated buffer, the loop never synchronizes to the host, and the
whole K-step loop can be captured once and replayed as a single graph launch:

```python
from flashrwkv2.mtp import DraftLoop

loop = DraftLoop.from_state_dict(block_state, lm_head_weight, depth=2)
loop.capture(embed)
ids = loop.replay(hidden, token_ids)   # [1, depth], device tensor
```

Draft ids match the allocating torch reference and the graph replay bitwise. The
loop is bandwidth bound on the shared LM head, so graph capture only removes the
launch and allocation overhead (H100 SXM, BF16, batch 1):

| depth | eager | graph | speed-up | head reads |
|------:|------:|------:|---------:|-----------:|
| 2 | 1.746 ms | 1.681 ms | 1.04x | 2.49 GB |
| 3 | 2.610 ms | 2.517 ms | 1.04x | 3.73 GB |
| 5 | 4.315 ms | 4.189 ms | 1.03x | 6.22 GB |

Reproduce with:

```bash
./.venv/bin/python benchmarks/mtp/draft/bench.py --depth 2
```

Cutting the draft below this floor needs fewer weight bytes, for example an FP8
copy of the block and head weights; the kernels in this tree are BF16 only.

## Tests

```bash
./.venv/bin/python -m pytest -q
```

CUDA tests compile TileLang kernels and the metadata/sampling extension lazily.
The full block suite checks outputs, FP32 state, and selected logprobs bitwise
across training/inference, prefill/decode, and batch composition.

## CI and release acceptance

`Quality Gate` is the required merge check. Ordinary PRs run CPU contracts,
package checks when needed, and the affected owners' complete CUDA test matrices
on PRO6000 with Torch 2.14.1. Provider changes expand to their consumers and the
block integration suite. Kernel changes also run the selected memcheck/racecheck
cases. Changes to dependencies, native interfaces, or runtime loading add the
Torch 2.11–2.14 compatibility matrix. Tests load the installed source wheel;
pytest's checkout import path is excluded from GPU validation.

Daily validation runs at 08:17 UTC on `main`, with full correctness, CUDA Graph,
memcheck, and racecheck coverage. Sunday adds all supported Torch versions, a
cold native build, and the complete benchmark/tuning suite. Daily performance
reports measure owners changed since their latest compatible successful
measurement. Performance failures and regressions appear in an independent
report; they do not block the PR quality check. The Actions summary records
actual selected/executed tests and tracks five completed PR timings against an
initial 20-minute target for a single-owner change, excluding queue time.

Successful results and original distributions share one 90-day artifact bundle.
Identical source trees can reuse CPU results and individual GPU version results
when the policy, environment, and complete collected test scope match. A full run can add
missing compatibility or cold-build checks to an existing bundle. Validation
never combines module coverage from different source revisions.

Before releasing, run the `Quality Gate` workflow on the exact release tag with
`mode=full` and `extended=true`. This produces the original wheel/sdist, full
PRO6000 correctness and compatibility results, and a separate complete
performance report. The `PyPI release` workflow verifies these results and
publishes those original files. Its manual `tag` input verifies acceptance
without publishing.

RTX4090 acceptance is independent. On its checkout of the same tag, download
the PRO6000 `flashrwkv2-quality-v7-<source-tree>` artifact to
`artifacts/acceptance`. Use the project `.venv`, with the default Torch version,
and the wheel from that bundle. With read access to Actions available through
`GH_TOKEN`, the existing CI entrypoints produce the independent reports:

```bash
timeout 600s uv sync --frozen --no-install-project --group dev
timeout 60s .venv/bin/python ci/quality_gate.py plan \
  --base HEAD^ --head HEAD --mode full --extended --target sm89 \
  --output artifacts/sm89/plan.json
timeout 60s .venv/bin/python ci/quality_gate.py package \
  --directory artifacts/acceptance --output artifacts/sm89/package.json
CUDA_VISIBLE_DEVICES="" timeout 600s .venv/bin/python ci/quality_gate.py test \
  --plan artifacts/sm89/plan.json --suite cpu --output artifacts/sm89/cpu.json \
  --reuse artifacts/acceptance/quality-manifest.json
CUDA_VISIBLE_DEVICES=0,1 PATH="$PWD/.venv/bin:$PATH" timeout 6600s .venv/bin/python ci/quality_gate.py gpu \
  --plan artifacts/sm89/plan.json --package artifacts/acceptance \
  --python .venv/bin/python --output artifacts/sm89
# Compatibility testing ends on an older Torch; restore the primary version.
timeout 600s uv sync --frozen --no-install-project --group dev
timeout 180s uv pip install --python .venv/bin/python --no-deps --reinstall artifacts/acceptance/dist/*.whl
timeout 120s .venv/bin/python ci/quality_gate.py finalize \
  --plan artifacts/sm89/plan.json --package artifacts/acceptance \
  --validation artifacts/sm89 --output artifacts/sm89-quality-manifest.json
CUDA_VISIBLE_DEVICES=0,1 PATH="$PWD/.venv/bin:$PATH" timeout 13800s .venv/bin/python ci/benchmark_gate.py report \
  --quality artifacts/sm89-quality-manifest.json --package artifacts/acceptance \
  --profile full --repository rwkv-rs/FlashRWKV2 --python .venv/bin/python \
  --output artifacts/sm89/performance --summary artifacts/sm89/performance.md
timeout 30s cp artifacts/sm89/performance/summary.json artifacts/sm89-performance-summary.json
```

Attach `sm89-quality-manifest.json` and `sm89-performance-summary.json` to the
tag's draft GitHub release. Both identify the exact source tree and wheel
SHA256; the release verifier checks RTX4090 identity, full test completion,
compatibility, sanitizer coverage, and complete performance results on both
architectures. Publish the release after its manual verification succeeds.

## Benchmarks

Operator benchmarks live in [`benchmarks/`](benchmarks/). The WKV matrix covers
batch size, sequence length, and head dimensions:

```bash
PYTHONPATH=benchmarks ./.venv/bin/python benchmarks/tmix/wkv7/bench.py \
  --shapes h12d64 h16d64 h32d64 h40d64 h64d64 h80d64 h16d128 h8d256 \
  --cases 1x1 16x1 1x16 64x16 1x128 --tune --timing-mode graph \
  --output artifacts/wkv7-recurrent-bf16.json
```

Deterministic strategies on RTX 4090 (SM89/128 SM) and PRO 6000 (SM120/188 SM)
prepare bounded kernel variants for dynamic batch/token workloads. Supply real
first/later-layer, decode/prefill and training-backward calls with their actual
layouts and gradient requirements:

```python
from flashrwkv2 import prepare_tuning

report = prepare_tuning(warmup_calls, mode="tune")
assert report["complete"]
```

Prepare before torch.compile specialization and CUDA Graph capture. Other paths
retain verified profiles and bounded tuning. After upgrading, prepare again;
compatible TileLang binaries can be reused. See the
[preparation and cache contract](docs/kernel_api.md#tuning-and-acceptance) for
coverage, state ownership, validation and performance results.

## License

FlashRWKV2 is distributed under the
[MIT License](https://github.com/rwkv-rs/FlashRWKV2/blob/main/LICENSE).
