# Qwen3.8 Flash-Next on one DGX Spark or ASUS Ascent GX10

A **TensorFold EXL3 4.05** serving recipe for one NVIDIA GB10 with 128 GB unified memory. The default is **thinking enabled, 262,144-token context, native vision and MTP6**.

![Qwen3.8 Flash-Next on one GB10: TensorFold 75.68 versus native ExLlamaV3 67.54 tok/s, +12.0% short-prompt decode, with native ahead on approximately 32K prefill](results/tensorfold-exl3-405-card.png)

## The recipe

```text
Language weights: turboderp/Qwen3.8-Flash-Next-exl3
Quantization:     4.05bpw_h6_ng6
Runtime:          TensorFold 0.6.5, pinned source
Hardware:         one GB10, 128 GB unified memory
Context:          262,144 tokens
MTP:              depth 6, confidence 0.70
KV cache:         BF16
Streams:          2, required for native vision
N-gram table:     demand-paged, prefetch=False
Lookup:           off
CUDA graphs:      on
Vision:           verified BF16 tower
API:              OpenAI-compatible, localhost:8001
Model ID:         qwen38-flash-next-tf405
Output default:   32,768 tokens, request-overridable
Thinking default: on, request-overridable
```

The language checkpoint includes its MTP head. No separate language drafter or language-loader patch is needed. The vision tower is a separate verified download; language weights remain unchanged.

## 1. Prepare the host

Use a GB10 host with working Docker GPU access, Python 3 and Git. The container includes its own Python 3.12, CUDA 13 development tools and pinned runtime dependencies.

```bash
git clone https://github.com/sxuff/qwen38-flash-next-dgx-spark.git
cd qwen38-flash-next-dgx-spark
uname -m
nvidia-smi
docker info
free -h
df -h "$HOME"
```

`uname -m` must report `aarch64`. Only one model-scale deployment should occupy the box. The serving launcher checks the preload headroom plan before starting and **does not stop other services** to make room. It keeps at least **7 GiB MemAvailable**, **20 GiB free disk**, **zero container swap**, and at most **1 GiB net host-swap growth** from launch. Stable pre-existing host swap is allowed; system swap settings are not changed.

## 2. Download and verify the weights

```bash
export MODEL_ROOT="$HOME/models/Qwen3.8-Flash-Next-EXL3-4.05"
export VISION_ROOT="$HOME/models/Qwen3.8-Flash-Next-BF16-vision"

python3 scripts/download_model.py \
  --manifest manifests/exl3-405.json \
  --destination "$MODEL_ROOT"

python3 scripts/download_model.py \
  --manifest manifests/vision-bf16.json \
  --destination "$VISION_ROOT"

export VISION_WEIGHTS="$VISION_ROOT/vision_tower_bf16.safetensors"
```

Both manifests pin immutable Hub revisions and file sizes. Large payloads use SHA-256; Git-stored metadata files use their Git-blob SHA-1. Downloads resume through temporary files, verify before promotion, and retain a 20 GiB disk reserve. `--verify-only` checks an existing download without writing it. Both artifacts are public; no Hub token is required.

- Language revision: `55a732e0c4c3d4614bc42b68493bb930d9b02c0a`
- TensorFold revision: `609ca419abecebdc5a059498a613680bd3aa847f`
- BF16 tower revision: `b5a234522364b10cdc4869844597bf77eeb942a7`
- BF16 tower SHA-256: `cdd69998e52e34badced49ef8ec09824b8b9a52e488094f0cadeadc7ebfc641e`

## 3. Build TensorFold

```bash
bash scripts/build_tensorfold.sh
```

The build uses an immutable ARM64 CUDA base, PyTorch **2.13.0+cu130**, Triton **3.7.1**, and the pinned TensorFold source. It installs no other inference engine. The command finishes with CPU-side runtime and image-processor imports. CUDA extensions compile on the first model load, with `MAX_JOBS=1`; their cache is retained under `runtime-cache/`.

## 4. Start the server

Foreground:

```bash
bash scripts/run_server.sh "$MODEL_ROOT" "$VISION_WEIGHTS"
```

Or install the user service:

```bash
bash scripts/install_service.sh "$MODEL_ROOT" "$VISION_WEIGHTS"
systemctl --user enable --now qwen38-flash-next-tensorfold.service
journalctl --user -u qwen38-flash-next-tensorfold.service -f
```

The installer verifies both artifact manifests. It refuses to overwrite an existing unit and does not start a deployment implicitly. The Docker container is capped at **112 GiB**, with service swap disabled. The independent host supervisor checks the reserve and swap-growth guards throughout startup and serving. A healthy idle server has no idle timeout.

The endpoint is bound to **127.0.0.1:8001**. This recipe does not install an external proxy or publish an unauthenticated API to the network.

## 5. Verify thinking, text and native vision

```bash
python3 scripts/smoke.py --base-url http://127.0.0.1:8001
```

The smoke checks model identity, the advertised 262,144-token context, a completed response with nonempty `reasoning_content` and no thinking override, a real text response and a generated two-color image. The latter two checks explicitly disable thinking to isolate text and vision behavior. The image prompt does not name the colors. Passing this check establishes functional thinking, text and native image input, not full-window quality.

Example request:

```bash
curl http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-flash-next-tf405","messages":[{"role":"user","content":"Write a Python function that validates a UUID."}],"max_tokens":2048,"temperature":0,"stream":false}'
```

The output default is 32,768 tokens when a request omits its own cap. `max_tokens` or `max_completion_tokens` can override it; input plus generated tokens must fit the configured context. A client must also send images as native image content, not just a local file path.

**Thinking is enabled by default.** Chat responses return it in `reasoning_content`, separately from the answer in `content`; streamed responses use the corresponding delta fields. Reasoning and the answer share the output token budget. A client must support that field to display thinking traces. To disable thinking for an individual request, send `"chat_template_kwargs":{"enable_thinking":false}`. Requests that omit the switch retain thinking.

After pulling this update, rebuild with `bash scripts/build_tensorfold.sh` and restart an existing deployment so the container uses the updated launcher.

## Measured comparison

**TensorFold wins short-prompt decode and TTFT. Native ExLlamaV3 wins the approximately 32K prefill proxies.** All 70 selected A/B quality items have identical pass/fail outcomes.

| Metric | TensorFold EXL3 4.05 | Native ExLlamaV3 EXL3 4.05 |
|---|---:|---:|
| Short-prompt decode | 75.68 tok/s | 67.54 tok/s |
| TTFT | 0.209 s | 0.532 s |
| 8K code prefill proxy | 700.13 tok/s | 600.94 tok/s |
| ~32K code prefill proxy | 485.56 tok/s | 667.69 tok/s |
| ~32K prose prefill proxy | 448.49 tok/s | 667.14 tok/s |
| GSM8K subset | 49/50 | 49/50 |
| HumanEval subset | 14/20 | 14/20 |
| Peak whole-host unavailable memory | 77.37 GiB | 80.29 GiB |

The benchmark used **40,960-token context, text-only, one stream**. Both A/B arms used the same EXL3 language weights, but this is a **recipe-deployment comparison**, not an isolated runtime toggle: TensorFold used MTP6, confidence 0.70 and BF16 KV; the tested native ExLlamaV3 fork used dynamic MTP5, confidence 0.60 and 8-bit KV.

Speed is the equal-weight mean of four per-prompt medians, with three repetitions and 400 generated tokens per request. Decode is a post-first-emission proxy; TTFT includes prefill and the first sample. Prefill is a single-token request-wall proxy including EOS and overhead, **not isolated kernel throughput**. The approximately 32K fixtures contain 32,020 code and 32,196 prose tokens. Whole-host unavailable memory is `MemTotal - min(MemAvailable)` across loading and collection, not GPU allocated memory.

The quality scores cover 50 GSM8K and 20 HumanEval selected items, not full benchmark scores. TensorFold's same-load drafted-versus-serial check matched **400/400 token IDs on each of four prompts**, not A/B output equivalence. The supplied card's additional reference column is preserved with its provenance in the result receipt; it is not another install profile.

**The default 262K/native-vision deployment is separate from the 40K/text-only benchmark.** Image and text functional checks passed, but the displayed throughput is **not** a 262K or vision-enabled benchmark. No full-window stress result is claimed.

Exact values, runtime pins, retained evidence hashes and card checksum: [`results/tensorfold-exl3-405.json`](results/tensorfold-exl3-405.json).

## Cleanup

```bash
systemctl --user disable --now qwen38-flash-next-tensorfold.service
rm -f "$HOME/.config/systemd/user/qwen38-flash-next-tensorfold.service"
systemctl --user daemon-reload
```

This stops only the recipe's owned container. Model files, the Docker image and compilation caches are retained.

## Licenses

The recipe scripts are MIT licensed. TensorFold is Apache-2.0 licensed. The model weights are separately governed by the [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/f5d08274bafd880402bd16f5e3e6c514136ec06c/LICENSE). Review the model license before downloading or deploying them.
