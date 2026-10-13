# Reading the OCR resource lab

The [results](../experiments/ocr-resource-profile.md) and
[04e notebook](../../notebooks/04e_ocr_resource_lab.ipynb) are a small systems
experiment, not a production deployment or the official benchmark.

## Follow one request

The input is an already canonical RGB document in RAM. We crop seven fields,
resize/normalize pixels on the CPU, copy float32 tensors to CUDA, run the vision
encoder and autoregressive text decoder, bring token IDs back, decode strings,
and apply the existing normalizers and validators.

One processed field is `1 × 3 × 384 × 384 × 4` bytes: **1.6875 MiB**.
Seven fields total **11.8125 MiB**, despite the original crops being small.
The model sees resized tensors, not compressed JPEG files. Moving weights is a
separate startup cost; we do not reload the model for every request.

CUDA operations are asynchronous. Timing only a Python call can measure how
quickly work was queued rather than when it finished. The lab synchronizes at
stage boundaries and uses CUDA events in the transfer microbenchmark. These
barriers make completed costs visible, but deliberately prevent overlap and add
overhead. They are a diagnostic decomposition, not an optimized serving loop.

Generation wall time includes CPU dispatch, GPU execution, synchronization, and
the encoder plus decoder. It is **not GPU busy time**. Process CPU milliseconds
count work across threads, so they may exceed wall milliseconds. Actual kernel
intervals are visible in the separate instrumented traces. Never add inclusive
operator timings together as if they were a nonoverlapping timeline.

## Four meanings of cache

| Mechanism | What gets reused | What still runs |
|---|---|---|
| Decoder KV cache | Attention keys/values within one field's generation | Encoder, current decoder steps, preprocessing |
| Prepared CPU input | Resized/normalized pixel tensor for an identical crop | H2D, model, decoding |
| Prepared GPU input | That tensor already on CUDA | Model, decoding |
| Result cache | Entire response for an identical request and pipeline version | Key hashing, lookup, response reconstruction |

KV cache is reset between independent generations. It is not a cross-document
answer lookup. Short outputs leave less prefix work to reuse; its benefit must
be measured. Prepared tensors trade RAM/VRAM for avoiding work and only help
when those exact inputs recur. Their experiment excludes cache-fill cost and
uses three repeated observations, so it is a diagnostic upper bound, not a
production traffic forecast.

The result cache has eight slots and an LRU eviction rule: accessing an entry
makes it most recently used; inserting beyond capacity evicts the oldest.
Its key hashes canonical pixels, crop geometry, source scale, and a fingerprint
of weights, generation policy, and pipeline definitions. Identity alone would
be unsafe: another photo of the same person is a different input. An unchanged
image with new weights or preprocessing must also miss. Hashing and response
reconstruction are inside hit timings. Three initial misses and 60 intentional
hits do **not** estimate a real hit rate. A deployed identity-document cache
would additionally need tenant isolation, expiry, and privacy controls.

## Why pinning did not transform performance

Ordinary CPU memory is pageable; pinned memory is page-locked. Already-pinned
buffers can support efficient transfers. But pinning a buffer on the request
path itself costs time. `non_blocking=True` lets the host continue without the
usual immediate synchronization; it does not by itself demonstrate overlap.
Our test waits for completion and uses a single stream. It measures transfer
cost, not a multistream copy/compute pipeline. The seven-field microbenchmark
uses one packed tensor, not seven individual copies. See the
[PyTorch transfer tutorial](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html).

## Quota is not thread count

Linux cgroups enforce CPU time and memory budgets. `cpu.max=100000 100000`
permits one CPU-second per second over quota periods, not ownership of one
particular core. A pool can briefly run on several cores, consume its allowance,
and then be throttled. Four CPU threads under a one-CPU quota can therefore
behave very differently from one thread under that same quota. The follow-up
control changes PyTorch/OpenMP/BLAS thread policy together; it does not isolate
one library or prove one particular spin-wait implementation caused the effect.

`memory.max` is a real host-memory ceiling; `memory.swap.max=0` disables swap for
the worker. Read `memory.events` to distinguish normal operation from limit/OOM
events. Cgroup memory includes charges beyond process RSS, such as file cache;
they are different measurements. Shared mapped pages can also make RSS larger
than this cgroup's own charge. Throttled time can accumulate across threads
and exceed elapsed wall time—it is not a wall-time percentage. See the
[kernel's cgroup v2 documentation](https://docs.kernel.org/admin-guide/cgroup-v2.html).

The 1 GiB GPU cap is **only a PyTorch allocator limit**. It is not a cgroup GPU
limit, hardware partition, or compute-share guarantee. CUDA context/driver and
other processes' memory are outside it. PyTorch allocated memory is live tensor
storage; reserved memory includes allocator-held reusable blocks. Releasing a
tensor need not reduce reserved memory. We leave desktop/other kernels alone and
do not mislabel whole-device utilization or power as this pipeline's consumption.

## Why a queue changes the story

One worker handles requests sequentially. A producer schedules arrivals without
waiting for completions: an **open-loop** load, avoiding the illusion of low
latency caused by clients slowing down with the server. There are four pending
slots, excluding the active request. Full queues reject immediately.

```text
Response latency = queue wait + service time
```

When arrivals outpace completions, waiting grows until the queue fills. A bounded
queue trades rejection for bounded backlog; it creates no additional capacity.
We report offered/completed/rejected counts together, and p95 response time only
for completed requests. Ignoring rejections would make an overloaded system
look deceptively healthy. Actual arrival lateness is retained because the
producer shares the worker's CPU quota. A 12-request burst is different from
evenly paced traffic, even with the same total request count.

Batching here means the **seven fields of one document**, not seven users or an
adaptive cross-request batch. It reduces generation calls but increases peak
VRAM and pads generation to a batch's longest sequence. All resulting strings
are checked against the retained sequential predictions. Faster but different
answers would be a separate accuracy tradeoff, not a free optimization.

## Boundaries worth remembering

The short runs are demonstrations, not SLA estimates. Variants run in order;
there is no repeated counterbalanced machine-session study. Traces add overhead
and are excluded from latency samples. No HTTP, network, image upload/decode,
rectification, autoscaler, distributed cache, or independent GPU isolation is
simulated. The first request is cold for that process, **not** a cold disk/page
cache experiment. We never flush system caches or deliberately trigger OOM.

For trace mechanics, see [PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html);
for decoder reuse, see [Transformers caching](https://huggingface.co/docs/transformers/kv_cache).
The experiment changes no official model/pipeline settings and closes no phase.

Try explaining: Why can a GPU service be CPU-throttled? Why can a 200 ms service
have a 900 ms response? Why does a 1 ms cache hit say nothing about new images?
Which measured costs would return if a cached tensor or answer were evicted?
