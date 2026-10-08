# Qwen3.8 Flash-Next on one DGX Spark or ASUS Ascent GX10

A pinned, checksum-verified llama.cpp recipe for the `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` IQ3_XXS target on one NVIDIA GB10 system, plus a matched TensorFold versus native ExLlamaV3 comparison on identical EXL3 4.05 language weights.

## Existing llama.cpp recipe architecture

```text
Target:        GSQ-RCO IQ3_XXS
Runtime:       llama.cpp 797da982 + sparse FA
QSA selection: block-granular top-k
MTP sidecar:   shared Q4_K_M
MTP policy:    depth 4 below 49,152 active tokens
               depth 6 at or above 49,152 active tokens
p-min:         0.75
Deployment:    one 262,144-token slot
Batch/ubatch:  2,048 / 512
Vision:        F16 projector
```

This repository contains the build patches, artifact manifests, service scripts, and public-safe measured results. It does not contain model weights.

## Latest measured result: TensorFold vs native ExLlamaV3

![Qwen3.8 Flash-Next on one GB10: TensorFold 75.68 versus native ExLlamaV3 67.54 tok/s, +12.0% short-prompt decode, with native ahead on approximately 32K prefill](results/tensorfold-exl3-405-card.png)

**TensorFold wins short-prompt decode and TTFT. Native ExLlamaV3 wins the approximately 32K prefill proxies.** All 70 selected A/B quality items have identical pass/fail outcomes.

| Metric | TensorFold EXL3 4.05 | Native ExLlamaV3 EXL3 4.05 | Earlier GSQ-RCO IQ3_XXS |
|---|---:|---:|---:|
| Short-prompt decode | 75.68 tok/s | 67.54 tok/s | 45.69 tok/s |
| TTFT | 0.209 s | 0.532 s | 0.280 s |
| 8K code prefill proxy | 700.13 tok/s | 600.94 tok/s | 639.69 tok/s |
| ~32K code prefill proxy | 485.56 tok/s | 667.69 tok/s | 532.59 tok/s |
| ~32K prose prefill proxy | 448.49 tok/s | 667.14 tok/s | 458.70 tok/s |
| GSM8K subset | 49/50 | 49/50 | 49/50 |
| HumanEval subset | 14/20 | 14/20 | 9/20 |
| Peak whole-host unavailable memory | 77.37 GiB | 80.29 GiB | 68.36 GiB |

A and B used the same `turboderp/Qwen3.8-Flash-Next-exl3` language artifact, branch `4.05bpw_h6_ng6`, revision `55a732e0c4c3d4614bc42b68493bb930d9b02c0a`. This is a **recipe-deployment comparison**, not an isolated runtime toggle:

- TensorFold 0.6.5, commit `609ca419abecebdc5a059498a613680bd3aa847f`: MTP6, confidence 0.70, BF16 KV, lookup off, demand-paged n-gram table. No language-loader patch.
- Native ExLlamaV3 1.5.1 fork, commit `94ba01d50a13fa9ff672473f2d0eef8b51a71e99`: dynamic MTP5, confidence 0.60, 8-bit KV, disk-backed n-gram table. Direct ExLlama API collection, not Tabby API throughput.
- Both measured configurations: **40,960-token context, text-only, one stream**. The GSQ column is an **earlier same-prompts sweep**, with different weights and HTTP collection, not a contemporaneous third arm.

Speed uses four fixed prompts, three repeats per prompt, and 400 generated tokens per request. The aggregate is the equal-weight mean of the four per-prompt medians. Decode is a post-first-emission proxy; TTFT includes prefill and the first sample. Prefill is a single-token request-wall proxy including EOS and overhead, not isolated kernel throughput. The ~32K fixtures contain 32,020 code and 32,196 prose tokens; A/B emitted EOS first in those requests. Whole-host unavailable memory is `MemTotal - min(MemAvailable)` across loading and collection, not GPU allocated memory.

The quality panel contains 50 GSM8K and 20 HumanEval selected items, not full benchmark scores. TensorFold's drafted-versus-serial exactness check matched **400/400 token IDs on each of four prompts** within the same TensorFold load; this is not A/B output equivalence. All **259/259 rows** are accounted for: A89, B85, earlier C85.

The subsequent serving configuration has **262,144-token context, two streams and native BF16 vision**. Image and text functional checks passed separately. The displayed throughput is **not** a 262K or vision-enabled benchmark, and no full-window stress result is claimed.

Machine-readable values, runtime pins, card checksum and evidence qualifiers: [`results/tensorfold-exl3-405.json`](results/tensorfold-exl3-405.json). The launcher below remains the existing llama.cpp recipe; it does not install TensorFold.

### Earlier GSQ-RCO target-quant comparison

The following results compare two GGUF targets in the same llama.cpp deployment. Their synthetic 32K/65K workloads are different from the four-prompt TensorFold panel above. In particular, 71.11 tok/s on the synthetic 65K fixture and 45.69 tok/s in the earlier same-prompts column are not a before/after regression comparison.

The measured swap changed **only the target GGUF**: Unsloth UD-Q3_K_XL to ISTA-DASLab GSQ-RCO IQ3_XXS. Both arms used the same pinned, patched llama.cpp binary, shared Q4_K_M MTP sidecar, context gate, flags, and requests.

| Metric | UD-Q3_K_XL | GSQ-RCO IQ3_XXS | Change |
|---|---:|---:|---:|
| 32K decode | 62.18 tok/s | 68.40 tok/s | +10.0% |
| 65K decode | 61.66 tok/s | 71.11 tok/s | +15.3% |
| Six-row wall time | 204.63 s | 193.05 s | -5.7% |
| Target GGUF size | 89.99 GB | 75.84 GB | -15.7% |
| Wikitext-2 raw-test perplexity | 4.0486 | 4.0824 | +0.8% |
| Source-code perplexity | 1.3748 | 1.3861 | +0.8% |
| 30K real-text prefill | 539.42 tok/s | 529.74 tok/s | -1.8% |
| Four local task checks | 4/4 | 4/4 | unchanged |

This is **one local pass per row**, not a repeated-run performance claim. The 32K and 65K decode rows use deterministic synthetic prompts with a forced 256-token output; the 30K prefill row uses real source text. Perplexity is measured at 4,096 context on two fixed corpora and rises slightly, but stays within the predeclared 2% limit. Smoke, tool-call, JSON, vision, and bounded long-generation checks passed with no detected loops. The previous outputs are not claimed to match the new quant's outputs.

Full machine-readable values, artifact hashes, and qualifiers: [`results/gsq-iq3xxs.json`](results/gsq-iq3xxs.json). Historical results remain in `results/` for comparison, not as current recommendations.

## Verified llama.cpp stack

- Hardware: one NVIDIA GB10 system with 128 GB unified memory
- Target: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`
- Target quantization: `IQ3_XXS`, 2 GGUF shards, 75,839,998,528 bytes
- Target revision: `c67535ccaa71f61547bb323a2828d5298221d83e`
- Target shard SHA-256 values: [`manifests/gsq-iq3xxs.json`](manifests/gsq-iq3xxs.json)
- MTP sidecar: shared Q4_K_M, 1,907,151,936 bytes
- MTP revision: `38bb39ee97821de2c9009abb7e93950eec396e66`
- MTP SHA-256: `f521868a9e143718bef513772f6e04d9642551e362cf2439636d2abdbd149dfc`
- Projector: `mmproj-F16.gguf`, revision `824f539b2710e5a9e47af4952cf6578cf5ee8932`
- Projector SHA-256: `1f7b7f0b984cf065c604360c29c8098362ed61b290db0ff12c6f360bb1a8a980`
- Runtime base: `ggml-org/llama.cpp` commit `797da982b488254b11f844718c30e0c41f34d718`
- Measured `llama-server` SHA-256: `3eba0da6c1b3f1f28e77901245432f20f505b3a10a98bd096480b62e87049793`
- Block-top-k patch SHA-256: `1c8c953bae42a9b9765330b802a3d8ce38d0f15dd09b0f12242103010c152814`
- Context-gated MTP patch SHA-256: `68f6e9e3ea7999aabeabcc253dd78faa985a63f91299256a95bdda01945dada7`
- CUDA target: SM121
- API: OpenAI-compatible llama.cpp server bound to localhost

The model is licensed separately under the [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/f5d08274bafd880402bd16f5e3e6c514136ec06c/LICENSE). Review it before downloading or deploying the weights. The scripts and recipe patches in this repository are MIT licensed.

## 1. Preflight the host

Run this on the GB10 host, not inside a management container:

```bash
uname -m
nvidia-smi
docker run --rm --gpus all nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 nvidia-smi
free -h
df -h "$HOME"
```

`uname -m` must report `aarch64`. Keep at least 10 GiB free beyond the downloads and at least 6 GiB `MemAvailable` while serving.

Install build dependencies:

```bash
sudo apt-get update
sudo apt-get install -y git clang cmake ninja-build libcurl4-openssl-dev libssl-dev python3
```

## 2. Download and verify the artifacts

The downloader is resumable, pins every Hub revision, verifies size and SHA-256, and requires a disk reserve.

```bash
export MODEL_ROOT="$HOME/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-c67535ccaa71"
export MTP_ROOT="$HOME/models/Qwen3.8-Flash-Next-MTP-38bb39ee9782"

python3 scripts/download_model.py \
  --manifest manifests/gsq-iq3xxs.json \
  --destination "$MODEL_ROOT"

python3 scripts/download_model.py \
  --manifest manifests/q3-mmproj-f16.json \
  --destination "$MODEL_ROOT"

python3 scripts/download_model.py \
  --manifest manifests/q3-mtp-shared-q4.json \
  --destination "$MTP_ROOT"
```

For authenticated Hub access, export `HF_TOKEN` in the shell. Do not put tokens in command arguments or unit files. Use `--verify-only` to verify existing artifacts without network writes.

## 3. Build the pinned runtime and patch set

```bash
export LLAMA_ROOT="$HOME/src/llama.cpp-qwen38-flash-next-next"
JOBS=2 bash scripts/build_llama_mtp.sh "$LLAMA_ROOT"
```

The build script:

1. fetches the exact base commit;
2. verifies both recipe-patch hashes;
3. applies block-granular QSA top-k;
4. applies the context-gated MTP-4/MTP-6 controller;
5. builds `llama-server` and `test-arg-parser` for SM121;
6. executes the parser/controller tests and CUDA device probe.

## 4. Install the GSQ-RCO profile

```bash
bash scripts/install_service.sh \
  "$MODEL_ROOT" \
  "$LLAMA_ROOT" \
  gsq-iq3xxs \
  "$MTP_ROOT/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf" \
  "$MODEL_ROOT/mmproj-F16.gguf"

systemctl --user enable --now qwen38-flash-next-llama.service
```

The installer verifies the target, Q4_K_M sidecar, and projector before writing the service environment. This is a reproducible **localhost** service recipe, not an installer for external access proxies. It writes the generic `qwen38-flash-next-llama.service` unit; inspect that unit before installing alongside an existing deployment. The service disables service swap and caps its cgroup at 110 GiB.

The promoted deployment flags are:

```text
-c 262144
-np 1
-b 2048
-ub 512
-ngl 99
-ot per_layer_token_embd=CPU
--no-kv-unified
--cache-prompt
--cache-reuse 0
--slot-prompt-similarity 0.10
--cache-ram 0
--no-cache-idle-slots
--no-context-shift
--load-mode mmap
--fit off
--spec-type draft-mtp
--spec-draft-n-max 6
--spec-draft-n-min 0
--spec-draft-p-min 0.75
--spec-draft-mtp-context-threshold 49152
```

## 5. Verify readiness and generation

```bash
journalctl --user -u qwen38-flash-next-llama.service -f
python3 scripts/smoke.py --base-url http://127.0.0.1:8001
```

A running process is not a ready model. Require model identity, real generation, advertised multimodal capability when the projector is installed, and zero service swap.

## llama.cpp context and vision validation

The configured context is 262,144 tokens, but the controlled throughput comparison stopped at 65K and is **not** a full 256K stress test. Both quants passed the same vision check with the unchanged F16 projector and a fixed red image, plus tool-call, JSON, smoke, and bounded long-generation checks. These are local functional checks, not general quality certification.

Long client conversations can prefill many thousands of history tokens before the first streamed response; a client-side first-chunk watchdog can fire even while the server is actively processing. Check the actual `/slots` and `/metrics` endpoints before treating a delayed first chunk as a downed service.

## Earlier llama.cpp comparison evidence boundaries

- One measured sweep per condition. No variance estimate.
- The 32K and 65K rows use deterministic synthetic long prompts and forced 256-token outputs.
- Absolute tok/s values are workload-scoped.
- The task suite and its copy-heavy fixtures favor speculative acceptance.
- Long-form creative prose is not represented by the synthetic headline rates.
- Raw prompts, outputs, SSE, host paths, credentials, and private deployment logs are intentionally excluded.

Historical MTP-3/MTP-4, n-gram, vision, and Q1 receipts remain under [`results/`](results/) for reproduction. Their earlier comparisons involved other runtime or policy changes and must not be conflated with this controlled quant swap. The old result-card image is no longer the README artifact.

## Prior art

[`MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark`](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark) established a single-Spark vLLM/NVFP4/PLE/MTP lane. [`Weschera/Qwen3.8-Flash-Next-1x-DGX-Spark`](https://github.com/Weschera/Qwen3.8-Flash-Next-1x-DGX-Spark) established a one-Spark llama.cpp MTP recipe with UD-Q4_K_XL and the shared Q8_0 sidecar. Unsloth publishes the target artifacts and shared MTP sidecars.

## Safety

Inspect the live service before and during long-context requests:

```bash
systemctl --user show qwen38-flash-next-llama.service \
  -p ActiveState -p MemoryCurrent -p MemorySwapCurrent -p MemoryMax -p MemorySwapMax
awk '/MemAvailable|SwapTotal|SwapFree/ {print}' /proc/meminfo
```

Stop the owned service if `MemAvailable` drops below 6 GiB, service swap becomes nonzero, or host swap grows by more than 512 MiB from the pre-launch baseline. Do not kill unrelated workloads to make this model fit.

## Cleanup

```bash
systemctl --user disable --now qwen38-flash-next-llama.service
rm -f "$HOME/.config/systemd/user/qwen38-flash-next-llama.service"
systemctl --user daemon-reload
```

Cleanup intentionally leaves model weights and source trees untouched.
