# Matched three-repeat prompt-kernel results

**Before = legacy; after = prompt kernels.** Both use EXL3 4.05/NG6, 262,144-token context, BF16 KV, MTP6/confidence 0.70, two streams and native vision. Each arm completed **33/33** requests: 12 short-decode requests, 3 long-context decode requests and 18 one-token prefill requests. This comparison measures timing and output identity, not a new task-quality panel.

### Prefix-uncached, page state uncontrolled: medians of 3 repetitions

| Prompt | TTFT before | TTFT after | Native prefill before | Native prefill after | Rate gain |
|---|---:|---:|---:|---:|---:|
| 8K code | 11.669 s | 4.979 s | 726.69 tok/s | 1711.77 tok/s | +135.6% |
| 8K prose | 11.516 s | 4.921 s | 739.13 tok/s | 1733.94 tok/s | +134.6% |
| ~32K code | 46.260 s | 19.990 s | 697.64 tok/s | 1616.17 tok/s | +131.7% |
| ~32K prose | 45.273 s | 19.844 s | 717.24 tok/s | 1638.42 tok/s | +128.4% |
| 64K code | 98.544 s | 42.015 s | 668.15 tok/s | 1568.19 tok/s | +134.7% |
| 64K prose | 97.361 s | 41.664 s | 676.29 tok/s | 1581.57 tok/s | +133.9% |

TTFT is request start to the first emitted sample. Native prefill rates include first-token sampling and are not isolated GPU kernel throughput or prompt-tokens/TTFT proxies. Each column is its own per-cell median over 3 repetitions. Prefix reuse was cleared between requests, but page residency was uncontrolled: **no cold-performance claim**. The retained ~32K fixtures contain 32,020 code and 32,196 prose prompt tokens.

| Post-first-emission decode cell | Before median | After median | Change |
|---|---:|---:|---:|
| Short code | 86.03 tok/s | 85.59 tok/s | -0.52% |
| Short math | 79.69 tok/s | 79.54 tok/s | -0.19% |
| Short prose | 56.85 tok/s | 56.84 tok/s | -0.02% |
| Short structured | 80.71 tok/s | 80.81 tok/s | +0.11% |
| 64K prose context | 90.54 tok/s | 90.91 tok/s | +0.40% |

These are also medians of 3 repetitions, with 400 generated tokens per request. All five changes meet the within-3% gate, not statistical equivalence. Output token-ID sequences matched on **33/33** paired requests, including the 18 one-token probes. This is bounded same-runtime identity evidence, not universal losslessness, full-window correctness or a new complete-answer quality result.

Exact values, counts, row-file hashes, paired token IDs and the default qualification: [`prefill-repeat.json`](prefill-repeat.json). Only the prompt-kernel default changes. The NG6 checkpoint, pinned runtime, and safety policy are retained.

## Scope and provenance

Each arm completed 33 frozen requests. Prefix reuse was cleared, while OS page residency was uncontrolled. The 64K fixtures extend retained 32K text by repetition, then native-encode, truncate to 65,536 IDs, decode and verify exact re-encoding. They are not newly recovered historical 64K originals.

Safety floors remained unchanged: 7 GiB available memory, 20 GiB disk free, zero owned-container swap and at most 1 GiB net host-swap growth. Both arms passed. Exact sampled extrema and fixed baseline are in the receipt.

The receipt preserves all paired emitted token IDs, request clocks, fixture hashes, source pins and patch hashes. Original fixture text, private raw events and the collection harness are not shipped; hashes cannot recreate those inputs. No new inference was run for this publication.
