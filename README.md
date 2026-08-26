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
- Runtime: llama.cpp PR [#27742](https://github.com/ggml-org/llama.cpp/pull/27742), commit `bea3b12daee45876b0129a3602dc8f534ce30bf0`
- CUDA target: SM121 (`121a-real` in this pinned llama.cpp build)
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

## 3. Build the pinned llama.cpp revision

```bash
bash scripts/build_llama.sh "$HOME/src/llama.cpp-qwen38-flash-next"
```

The script fetches PR #27742, rejects any head other than the pinned commit, builds `llama-server` for SM121, and runs `--list-devices`.

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

The service binds localhost only, disables host swap for its cgroup, caps memory at 110 GiB, and uses these measured controls:

```text
-ngl 99
-ot per_layer_token_embd=CPU
-b 2048
-ub 64
-c 262144
-np 8
--no-kv-unified
--cache-ram 0
--no-cache-idle-slots
--no-context-shift
```

With eight slots, the 262,144-token total context provides 32,768 tokens per slot. Reduce `-np` in `scripts/run_server.sh` if one request needs more than 32K context. Revalidate memory and output after changing any control.

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
