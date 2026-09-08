# Q3 MTP analysis correction

The sealed primary analyzer calculated `host_swap_growth_bytes` and `minimum_mem_available_bytes` from request collection only. Each arm’s runtime receipt separately retained startup-through-collection telemetry, and the controller enforced the frozen 512 MiB swap-growth ceiling and 6 GiB memory floor continuously.

The published analysis therefore:

- preserves the request-only value as `request_collection_host_swap_growth_bytes`;
- reports `host_swap_growth_bytes` as the maximum of request and complete-runtime growth;
- reports `minimum_mem_available_bytes` as the minimum across both scopes;
- applies the unchanged safety gates to those complete-scope values.

No request, output, timing, proposal counter, task score, or runtime receipt was changed or recollected. The correction did not change any arm’s eligibility, the aggregate winner, or the operator recommendation.
