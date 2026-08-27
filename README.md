# Qwen3.8 Flash-Next on one DGX Spark or ASUS Ascent GX10

A pinned, checksum-verified recipe for serving `unsloth/Qwen3.8-Flash-Next-GGUF` on one NVIDIA GB10 system with llama.cpp. Two exercised profiles are included: `UD-IQ1_S` and `UD-Q3_K_XL`.

This repository contains scripts and measured results. It does not contain model weights.

## Verified stack

- Hardware: one NVIDIA GB10 system with 128 GB unified memory
- Model: `unsloth/Qwen3.8-Flash-Next-GGUF`
- Quantization: `UD-IQ1_S`, 3 GGUF shards, 72,546,461,344 bytes
- Model revision: `d3bc75ee6ccef3efc1e228ec00a6cc2cdb1e2249`
- Q3 profile: `UD-Q3_K_XL`, 3 GGUF shards, 89,986,353,824 bytes
- Q3 model revision: `8bdc666649440e9bdc97e16f3f75782c98478ff5`
- Native-vision projector: `mmproj-F16.gguf`, revision `824f539b2710e5a9e47af4952cf6578cf5ee8932`, 904,004,000 bytes
- Runtime: llama.cpp PR [#27742](https://github.com/ggml-org/llama.cpp/pull/27742), commit `b8bdf73bb9baf044caadd33be2a51be70156ec57`
- CUDA target: SM121 (`121` in this pinned llama.cpp build)
- API: OpenAI-compatible llama.cpp server on `127.0.0.1:8001`

The model is licensed separately under the [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/f5d08274bafd880402bd16f5e3e6c514136ec06c/LICENSE). Review it before downloading or deploying the weights. The scripts in this repository are MIT licensed.

## 1. Preflight the host

Run this on the GB10 host, not inside a management container:

```bash
uname -m
nvidia-smi
docker run --rm --gpus all nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 nvidia-smi
free -h
df -h "$HOME"
```

`uname -m` must report `aarch64`. Both GPU checks must pass. Keep at least 10 GiB free beyond the model download and at least 6 GiB `MemAvailable` while serving.

Install build dependencies:

```bash
sudo apt-get update
sudo apt-get install -y git clang cmake ninja-build libcurl4-openssl-dev libssl-dev python3
```

## 2. Download and verify the exact model

The downloader is resumable, pins the Hub revision, verifies every size and SHA-256, and requires a 10 GiB disk reserve.

```bash
python3 scripts/download_model.py \
  --destination "$HOME/models/Qwen3.8-Flash-Next-UD-IQ1_S-d3bc75ee6cce"
```

Verify an existing download without network writes:

```bash
python3 scripts/download_model.py \
  --destination "$HOME/models/Qwen3.8-Flash-Next-UD-IQ1_S-d3bc75ee6cce" \
  --verify-only
```

For authenticated Hub access, export `HF_TOKEN` in the shell. Do not put tokens in command arguments or unit files.

To use the separately verified Q3 profile instead:

```bash
python3 scripts/download_model.py \
  --manifest manifests/q3-q3kxl.json \
  --destination "$HOME/models/Qwen3.8-Flash-Next-UD-Q3_K_XL-8bdc66664944"
```

Native vision requires the separate F16 multimodal projector. It was added to the Unsloth repository after the pinned Q3 weight revision, so it has its own immutable manifest. Download it into the same destination:

```bash
python3 scripts/download_model.py \
  --manifest manifests/q3-mmproj-f16.json \
  --destination "$HOME/models/Qwen3.8-Flash-Next-UD-Q3_K_XL-8bdc66664944"
```

The installer verifies and enables `mmproj-F16.gguf` automatically when it is present. If you skip this 904 MB file, Q3 still serves text but rejects native image input.

## 3. Build the pinned llama.cpp revision

```bash
bash scripts/build_llama.sh "$HOME/src/llama.cpp-qwen38-flash-next"
```

The script fetches the exact pinned PR commit, builds `llama-server` for SM121, and runs `--list-devices`.

## 4. Install the user service

```bash
bash scripts/install_service.sh \
  "$HOME/models/Qwen3.8-Flash-Next-UD-IQ1_S-d3bc75ee6cce" \
  "$HOME/src/llama.cpp-qwen38-flash-next"

systemctl --user enable --now qwen38-flash-next-llama.service
```

For Q3, select its profile during installation:

```bash
bash scripts/install_service.sh \
  "$HOME/models/Qwen3.8-Flash-Next-UD-Q3_K_XL-8bdc66664944" \
  "$HOME/src/llama.cpp-qwen38-flash-next" \
  q3-q3kxl
```

The service binds localhost only, disables service swap, and caps its cgroup at 110 GiB. The installer writes profile-specific controls to `~/.config/qwen38-flash-next/server.env`.

The Q1 profile retains the original measured concurrency configuration:

```text
-c 262144
-np 8
-b 2048
-ub 64
NGRAM_MOD=0
```

The Q3 profile uses the single-slot operating point exercised by the current n-gram comparison:

```text
-c 65536
-np 1
-b 512
-ub 64
NGRAM_MOD=1
MMPROJ_PATH=<model-root>/mmproj-F16.gguf
--mmproj <model-root>/mmproj-F16.gguf
--spec-type ngram-mod
--spec-ngram-mod-n-match 24
--spec-ngram-mod-n-min 48
--spec-ngram-mod-n-max 64
```

Both profiles keep these controls:

```text
-ngl 99
-ot per_layer_token_embd=CPU
--no-kv-unified
--cache-ram 0
--no-cache-idle-slots
--no-context-shift
--load-mode mmap
```

With eight Q1 slots, the 262,144-token total context provides 32,768 tokens per slot. Q3 is installed with one 65,536-token slot because that is the measured n-gram operating point. Revalidate memory and output after changing context, slots, cache type, or speculative settings.

Follow startup without exposing the server publicly:

```bash
journalctl --user -u qwen38-flash-next-llama.service -f
```

## 5. Health and generation smoke test

A running process is not a ready model. Wait for model loading, then run both checks:

```bash
python3 scripts/smoke.py --base-url http://127.0.0.1:8001
```

The smoke script requires HTTP health and a nonempty real completion.

## Measured Q1 sweep

The Q1 sweep and the original Q3 validation below were produced with the earlier pinned runtime commit `bea3b12daee45876b0129a3602dc8f534ce30bf0`. Their machine-readable result files retain that runtime identity. The current Q3 n-gram comparison uses the newer commit listed in the verified stack.

The measured sweep used exact input denominators of 512, 4,000, 16,000, and 32,000 tokens, concurrency 1, 2, 4, and 8, and exactly 256 generated tokens per request. There were 32 batches and 120 request rows. Cold requests disabled prompt caching. Warm requests reused the same prompt with server cache evidence.

Selected endpoints from that sweep:

| Input | Concurrency | Cold end-to-end completion throughput | Warm end-to-end completion throughput |
|---:|---:|---:|---:|
| 512 | 1 | 23.07 tok/s | 26.60 tok/s |
| 512 | 8 | 65.01 tok/s | 93.93 tok/s |
| 4,000 | 1 | 13.09 tok/s | 25.36 tok/s |
| 4,000 | 8 | 20.10 tok/s | 82.80 tok/s |
| 16,000 | 1 | 5.04 tok/s | 22.74 tok/s |
| 16,000 | 8 | 5.80 tok/s | 52.41 tok/s |
| 32,000 | 1 | 2.60 tok/s | 20.46 tok/s |
| 32,000 | 8 | 2.78 tok/s | 35.26 tok/s |

At concurrency 1, observed cold TTFT was 1.63, 9.62, 39.69, and 86.06 seconds at 512, 4,000, 16,000, and 32,000 input tokens. Warm TTFT was 0.11, 0.11, 0.12, and 0.13 seconds.

“End-to-end completion throughput” is total completion tokens divided by batch wall time. It includes prompt processing and is not per-request decode speed. Values shown are observed measurements from eligible batches. During the full sweep, minimum host `MemAvailable` was 64,950,276,096 bytes, service swap stayed at 0 bytes, and host swap grew by 29,167,616 bytes.

Machine-readable qualifiers and these selected rows are in [`results/q1-iq1s.json`](results/q1-iq1s.json).

## Measured Q3 validation

The Q3 profile passed a short fit, safety, and paired-quality gate under the same runtime controls. This was not a full performance sweep. All 10 performance requests used exact input denominators and generated exactly 256 tokens with `finish_reason: length`.

| Input | Concurrency | End-to-end completion throughput | TTFT | Server decode throughput |
|---:|---:|---:|---:|---:|
| 512 | 1 | 23.16 tok/s | 1.84 s | 27.67 tok/s |
| 32,000 | 1 | 2.42 tok/s | 93.28 s | 20.30 tok/s |

The 32K × 8 fit gate completed all eight requests in 787.41 seconds at 2.60 aggregate completion tok/s. The four-case deterministic quality corpus covered coding, structured tool arguments, instruction following, and reasoning. Q3 passed 4/4 cases and matched the recorded Q1 outputs exactly in 4/4 cases.

During this validation, minimum host `MemAvailable` was 17,894,166,528 bytes, service swap stayed at 0 bytes, and host swap grew by 74,678,272 bytes. Machine-readable qualifiers are in [`results/q3-q3kxl.json`](results/q3-q3kxl.json).

## Measured Q3 `ngram-mod` comparison

`ngram-mod` is a draftless speculative decoder. It hashes recent token sequences, proposes continuations seen earlier in the prompt or generated history, and lets Qwen verify the whole proposal. It does not add a second model and does not change accepted output tokens.

A single paired sweep compared the latest pinned runtime with speculation disabled against the same runtime with the four flags shown above. Both arms used identical request bytes, temperature 0, seed 42, thinking disabled, one 65,536-token slot, and 3,496 completion tokens across four cases.

| Case | Completion tokens | Baseline wall time | `ngram-mod` wall time | Speedup | Accepted draft tokens |
|---|---:|---:|---:|---:|---:|
| Copy Python | 1,189 | 48.00 s | 12.64 s | 3.80× | 1,069 / 1,088 |
| Copy JSON | 1,552 | 62.62 s | 23.74 s | 2.64× | 1,453 / 2,432 |
| Structured transform | 520 | 24.42 s | 12.56 s | 1.94× | 470 / 896 |
| Novel code | 235 | 8.70 s | 8.76 s | 0.99× | 0 / 0 |

All four treatment outputs matched their paired baseline outputs exactly. Aggregate wall time fell from 143.74 to 57.71 seconds, a 2.49× speedup. Server-reported decode throughput increased from 27.26 to 82.61 tok/s, and 2,992 of 4,416 drafted tokens were accepted. The novel-code control received no usable drafts and showed no meaningful improvement.

This is one operational sweep, not a broad model benchmark. It supports enabling `ngram-mod` for the measured single-slot Q3 profile, especially for copy-heavy code, JSON, and transformations. It does not establish a 2.49× gain for unrelated prompts. Full machine-readable qualifiers are in [`results/q3-q3kxl-ngram-mod.json`](results/q3-q3kxl-ngram-mod.json).

To reproduce the paired requests, restart once with `NGRAM_MOD=0` and once with `NGRAM_MOD=1`, then run:

```bash
python3 scripts/benchmark_ngram_mod.py \
  --base-url http://127.0.0.1:8001 \
  --arm baseline \
  --output baseline.json

python3 scripts/benchmark_ngram_mod.py \
  --base-url http://127.0.0.1:8001 \
  --arm ngram-mod \
  --output ngram-mod.json
```

If the server uses bearer authentication, export `QWEN38_GX10_API_KEY` before running the benchmark script.

## Measured native vision

The pinned F16 projector was hash-verified (`1f7b7f0b984cf065c604360c29c8098362ed61b290db0ff12c6f360bb1a8a980`), loaded with `--mmproj`, and exercised while the Q3 text model and `ngram-mod` remained unchanged. The server advertised both `completion` and `multimodal` capabilities.

Two direct OpenAI-compatible API checks passed:

- A generated 64×64 red square returned exactly `red`.
- A 2,108×972 Hermes failure screenshot returned the exact visible error banner: `The model provider failed after retries. I kept raw provider details out of chat; check gateway logs for diagnostics.`

A Hermes `vision_analyze` tool round trip against the same screenshot returned the same sentence through Qwen without the previous HTTP 500. Vision-enabled startup reached readiness in 107.49 seconds, minimum host `MemAvailable` during startup was 27,929,059,328 bytes, service swap stayed at 0, and host swap did not grow. Machine-readable evidence is in [`results/q3-q3kxl-vision.json`](results/q3-q3kxl-vision.json).

For a named Hermes provider, mark the served model vision-capable only after `/v1/models` advertises `multimodal` and a real image request passes:

```bash
hermes config unset model.supports_vision
hermes config set \
  providers.qwen38-flash-next-gx10.models.qwen38-flash-next-q3-k-xl.supports_vision \
  true
hermes config check
```

The `unset` prevents the persisted default model's top-level capability shortcut from shadowing a session-only `/model` switch on affected Hermes builds. If the projector is not installed, keep the Qwen model override `false` so Hermes routes screenshots through an auxiliary vision model instead of sending unsupported image input to llama.cpp.

## Safety and troubleshooting

Before changing runtime flags, check:

```bash
systemctl --user status qwen38-flash-next-llama.service --no-pager
systemctl --user show qwen38-flash-next-llama.service \
  -p MemoryCurrent -p MemorySwapCurrent -p MemoryMax -p MemorySwapMax
awk '/MemAvailable|SwapTotal|SwapFree/ {print}' /proc/meminfo
nvidia-smi
```

Stop the owned service if `MemAvailable` drops below 6 GiB, service swap becomes nonzero, or host swap grows by more than 512 MiB from the pre-launch baseline. Do not kill unrelated workloads to make this model fit.

If startup fails, diagnose in this order: GPU visibility, exact shard verification, disk/cache, pinned binary and CUDA device list, service logs, health endpoint, then generation.

## Cleanup

```bash
systemctl --user disable --now qwen38-flash-next-llama.service
rm -f "$HOME/.config/systemd/user/qwen38-flash-next-llama.service"
systemctl --user daemon-reload
```

The cleanup command intentionally does not delete model weights or source trees.
