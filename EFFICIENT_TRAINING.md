# FitLLM — Efficient Training Architecture (Memory / CPU / GPU)

> Design note. **No code is changed by this document.** It specifies an
> efficient memory + compute architecture for training, answers the question
> *"do we keep weights in CPU RAM and move them to GPU, and how do we cut
> latency?"*, and gives the algorithm and the latency math.

---

## 0. TL;DR

1. **Pick the regime by where the model fits** (Section 2). On the current box
   (A100‑80 GB, 216 GB RAM, 32B‑in‑4bit ≈ 16 GB) the model fits *entirely* in
   VRAM — so the fastest correct path is **load it fully on the GPU** (ordinary
   QLoRA), no streaming at all. Streaming only earns its keep when the model is
   larger than VRAM.

2. **When the model is larger than VRAM but fits in CPU RAM** (the real FitLLM
   case), the efficient design is:
   - **Keep the full quantized model resident in *pinned* (page‑locked) CPU
     RAM, permanently.** Read from NVMe exactly once, at startup.
   - **Stream layer weights CPU→GPU just‑in‑time on a dedicated CUDA copy
     stream, double/triple‑buffered**, so the transfer of layer *i+1* overlaps
     the compute of layer *i*.
   - **Use a fixed, pre‑allocated GPU "slot ring"** and let the caching
     allocator reuse it. **Never** call `gc.collect()`, `empty_cache()`, or
     `malloc_trim()` inside the layer loop.
   - **Verify checksums once at load, not per pass.**

3. **The current 19 h/step is not a bandwidth problem.** Moving 16 GB over
   PCIe Gen4 costs ≈ 0.5–0.6 s; a full forward is dominated by compute, not
   transfer. The time is being burned by (a) per‑layer Python GC + CUDA
   `empty_cache` + `malloc_trim(0)` (called 64 layers × ~16 passes/step), and
   (b) re‑hashing 16 GB of shards on *every* forward and backward. Remove those
   two and the step drops to **tens of seconds** (Section 6).

---

## 1. Why the current pipeline is slow (bottleneck diagnosis)

Per logged optimizer step with `grad_accum=8`, the engine does ~8 forward +
~8 backward full traversals of 64 layers = ~1024 layer "loads". For **each**
of those layer loads the current code path does:

| Cost per layer load | Where | Order of magnitude |
|---|---|---|
| `load_shard_with_checksum(..., verify=True)` → SHA‑256 of a 242 MB file | `scheduler._read_to_cpu` | ~0.3–0.8 s (CPU‑bound hashing) |
| `pin_memory()` copy | `scheduler._read_to_cpu` | tens of ms |
| `gc.collect()` + `torch.cuda.empty_cache()` + `malloc_trim(0)` | `scheduler.evict` | **0.1–1 s each, synchronous, frees pages back to OS** |
| H2D transfer 242 MB | `_transfer_to_gpu` | ~10 ms |

The two highlighted rows are the killers. `empty_cache()` forces a full CUDA
allocator sync + free; `malloc_trim(0)` returns heap pages to the OS so the
**next** allocation faults them back in. Doing this 1000×/step, plus hashing
~16 GB×16/step, is what produces ~0.1 tok/s. **PCIe is not the bottleneck.**

(There is also a *correctness* bug — the backward re‑runs each layer with no
attention mask and no autocast, and 4‑bit quant state may not round‑trip
through sharding — but that is orthogonal to throughput and is tracked
separately.)

---

## 2. Choose the regime: where does the model fit?

Let `W` = quantized weight bytes, `V` = usable VRAM, `R` = usable CPU RAM.

| Regime | Condition | Strategy | Relative speed |
|---|---|---|---|
| **A. Fits in VRAM** | `W + activations + LoRA ≲ V` | Load fully on GPU. No streaming. | Fastest |
| **B. Fits in RAM, not VRAM** | `W > V` but `W ≲ R` | **CPU‑pinned residency + overlapped CPU→GPU streaming** (this doc) | ~compute‑bound; near A for large batch |
| **C. Bigger than RAM** | `W > R` | NVMe→CPU→GPU streaming with prefetch + RAM cache for the hot set | Slowest; disk‑bound |

The current A100 box with a 32B‑4bit model is squarely in **Regime A**
(16 GB ≪ 80 GB). The artificial 4 GB cap forces it into Regime B/C for
demonstration purposes. Everything below is the efficient design for
**Regime B**, with C as a graceful fallback.

---

## 3. Core architecture (Regime B): CPU‑pinned residency + streaming

```
        ┌─────────────────────────── HOST (CPU) ───────────────────────────┐
        │  Pinned weight residency (ALL layers, loaded once)                │
        │   layer0_w … layerN_w   [NF4, page-locked]   ~16 GB               │
        │  Activation checkpoints (forward) h0..hL      [pinned]            │
        │  LoRA adapters A,B + Adam moments             [small, ~0.3 GB]    │
        └───────────────┬───────────────────────────────────────────────────┘
                        │  async H2D (non_blocking) on COPY STREAM, pinned→VRAM
                        ▼
        ┌─────────────────────── GPU (accelerator) ─────────────────────────┐
        │  Weight slot ring:  slot[0], slot[1], (slot[2])   ← K=2 or 3       │
        │   - dequant NF4→bf16 here, compute on COMPUTE STREAM              │
        │  Current activation h (bf16)                                      │
        │  LoRA adapters (resident, tiny)                                   │
        └───────────────────────────────────────────────────────────────────┘
```

**Principles**

1. **Residency, not re‑reading.** The 16 GB of quantized weights live in
   *pinned* CPU RAM for the entire run. Pinned memory is the precondition for
   (a) true asynchronous `cudaMemcpyAsync` and (b) full PCIe bandwidth. Disk is
   touched once.

2. **Two CUDA streams.** A *copy stream* moves the next layer's weights
   host→device; a *compute stream* runs the current layer. They are
   synchronized with CUDA events so compute waits only for *its own* layer's
   copy, and the copy of layer *i+1* proceeds during compute of layer *i*.

3. **Fixed GPU slot ring (K slots).** Pre‑allocate `K` device buffers big
   enough for one layer. Layer *i* uses `slot[i % K]`. No per‑layer
   `malloc`/`free`, no `empty_cache`. The PyTorch caching allocator keeps the
   memory; we just overwrite slots. `K=2` is enough to hide transfer behind
   compute; `K=3` adds slack for jitter.

4. **Dequantize on the GPU.** Transfer the *compact* NF4 bytes (4× smaller than
   fp16) over PCIe, then dequantize to bf16 in the slot on the GPU. This both
   minimizes PCIe volume and keeps compute in a fast dtype.

5. **No safety‑margin churn.** VRAM peak is deterministic:
   `K · slot_bytes + activation + LoRA + workspace`. Size it once; never probe
   mid‑loop, never trim.

---

## 4. The training‑step algorithm

### 4.1 One‑time setup
```python
# Load every layer's NF4 weights into pinned host tensors, once.
host_w[i] = load_shard(i, verify=True).pin_memory()      # i = 0..L-1
copy_stream    = torch.cuda.Stream()
compute_stream = torch.cuda.current_stream()
slot   = [alloc_layer_buffer() for _ in range(K)]        # K device buffers
ready  = [torch.cuda.Event() for _ in range(K)]          # "copy done" events
```

### 4.2 Forward (overlapped, double‑buffered)
```python
def prefetch(i):
    s = slot[i % K]
    with torch.cuda.stream(copy_stream):
        s.copy_(host_w[i], non_blocking=True)             # pinned → VRAM, async
        ready[i % K].record(copy_stream)

h = embed(input_ids)                                      # on GPU
save_ckpt(0, h.to('cpu', non_blocking=True))              # activation → pinned host
for j in range(min(K, L)): prefetch(j)                    # fill the pipeline

for i in range(L):
    compute_stream.wait_event(ready[i % K])               # wait ONLY this layer's copy
    w = dequant(slot[i % K])                              # NF4 → bf16 on GPU
    h = layer_i(h, w, mask, rope)                         # compute (autocast bf16)
    save_ckpt(i+1, h.to('cpu', non_blocking=True))        # checkpoint for backward
    if i + K < L: prefetch(i + K)                         # refill the slot we just used
logits = lm_head(h)
```
Key point: the only synchronization is `wait_event` on the *current* layer.
Layer *i+1*..*i+K‑1* are already in flight on the copy stream. No GC, no
`empty_cache`, no per‑layer host allocation.

### 4.3 Backward (correct + streamed)
Use **per‑layer activation‑checkpoint recompute** with the *same* mask / RoPE /
dtype as forward (this is what fixes the correctness bug), streaming weights the
same way in reverse:
```python
g = grad_of_loss_wrt(h_L)                                 # via lm_head re-run
for j in range(min(K, L)): prefetch(L-1-j)                # reverse-fill pipeline
for i in reversed(range(L)):
    compute_stream.wait_event(ready[i % K])
    w   = dequant(slot[i % K])
    h_in = load_ckpt(i).to('cuda', non_blocking=True).requires_grad_()
    with torch.enable_grad(), autocast(bf16):             # MATCH forward exactly
        h_out = layer_i(h_in, w, mask, rope)              # same mask & rope as fwd
    h_out.backward(g)                                     # populates LoRA .grad + h_in.grad
    g = h_in.grad                                         # propagate upstream
    if i - K >= 0: prefetch(i - K)
# LoRA .grad now holds this micro-batch's gradient (accumulates across grad_accum)
```
Because only LoRA `A,B` require grad and the base is frozen, autograd builds a
*tiny* per‑layer graph (just the adapter path), then it's freed when the loop
moves on. Activations are the only large host‑resident state.

### 4.4 Optimizer + accumulation
- LoRA `.grad` **accumulates in place** across the `grad_accum` micro‑batches
  (don't zero between them) — no disk, no separate grad files.
- After `grad_accum` steps: clip → AdamW update on the in‑RAM adapters → zero
  grads. Adapter + moments are ~0.3 GB; the step is sub‑second.
- **Do not clear or re‑warm the weight cache** — the pinned weights are frozen
  and never change, so they stay valid forever. (Only the LoRA adapters change,
  and they live resident.)

---

## 5. Memory budget (Regime B, 32B‑4bit, rank‑16)

| Tier | Contents | Size |
|---|---|---|
| **CPU pinned RAM** | all NF4 layer weights (resident) | ~16 GB |
| | activation checkpoints, L+1 of them (bf16, seq 512, b=1) | ~0.05–0.5 GB |
| | LoRA A/B + Adam (m,v) | ~0.3 GB |
| **GPU VRAM** | weight slot ring `K · slot_bytes` (K=2–3, ~0.25 GB NF4 + bf16 dequant ~0.5 GB) | ~1.5–2.5 GB |
| | current activation + autograd workspace | ~0.2–0.5 GB |
| | resident LoRA adapters | tiny |
| **NVMe** | source shards (read once) + checkpoints | ~16 GB |

GPU peak is **bounded by `K` and one layer**, *independent of model depth* —
the same invariant the paper proves, but now with no allocator churn. Host RAM
holds the whole model + activations comfortably within 216 GB.

---

## 6. Latency model — why this is ~1000× faster than today

Per‑layer numbers for a 32B/64‑layer model (≈0.5 B params/layer, NF4 ≈ 256 MB):

| Quantity | Estimate | Notes |
|---|---|---|
| H2D transfer of one NF4 layer | **~10 ms** | 256 MB ÷ ~25 GB/s PCIe Gen4 |
| Layer compute (seq 512, b=1, bf16, A100) | **~10–25 ms** | memory‑bound at b=1; grows with batch |
| Per‑layer wall time *with overlap* | **≈ max(10, 10–25) ≈ 10–25 ms** | Eq. (3) in paper: `max(T_pcie, T_compute)` |
| Forward (64 layers) | **~0.7–1.6 s** | |
| Backward (≈2× forward) | **~1.4–3.2 s** | recompute + grad |
| Micro‑batch (fwd+bwd) | **~2–5 s** | |
| Step (`grad_accum=8`) + optimizer | **~16–40 s** | vs. **~19 h** today |

The win comes entirely from (1) **eliminating disk I/O after load**,
(2) **eliminating per‑layer GC/`empty_cache`/`malloc_trim`**, and
(3) **overlapping transfer with compute**. Bandwidth was never the limit:
moving the whole 16 GB once is ~0.6 s.

**Throughput rules of thumb**
- Increase batch size / sequence packing → compute per layer rises, transfer is
  fixed → transfer is hidden "for free" and the GPU stays busy. Streaming
  overhead amortizes toward zero as batch grows.
- `K=2` suffices when `T_compute ≳ T_pcie`; bump to `K=3` only if you see copy
  stalls (compute waiting on `ready` events).

---

## 7. Latency‑reduction techniques (ranked by impact)

1. **Resident pinned weights (kill disk).** Biggest single win in Regime B.
   Load once; never read NVMe again. Pinned is mandatory for async + bandwidth.
2. **Remove per‑layer reclamation.** Delete `gc.collect()` / `empty_cache()` /
   `malloc_trim(0)` from the hot loop. Use the slot ring + caching allocator.
3. **Checksum once at load**, not per pass. (Keep `verify` for the `verify`
   CLI and first load only.)
4. **Copy/compute stream overlap + double buffering** (`K=2–3`).
5. **Transfer NF4 (compact), dequant on GPU** — 4× less PCIe than fp16.
6. **`non_blocking=True` everywhere host↔device**, both weights and activations.
7. **Don't re‑warm caches per step** — frozen base never changes.
8. **Bigger effective batch** (micro‑batch packing) to make compute dominate.
9. **Optional, advanced:** keep activations on GPU when they fit (skip the
   round‑trip), or use selective recompute (store every Nth activation,
   recompute the rest) to trade a little compute for host‑RAM headroom.
10. **Optional, hardware‑dependent:** GPUDirect Storage (cuFile) for Regime C to
    DMA NVMe→VRAM directly, bypassing the CPU bounce buffer — only helps when
    you *are* disk‑bound (i.e., model > RAM).

---

## 8. Regime C fallback (model > CPU RAM)

When even CPU RAM can't hold the model, keep a **bounded hot‑set cache** in
pinned RAM (as many layers as fit) and stream the rest from NVMe with the same
double‑buffered copy stream, prefetching the *slowest* shards first. This is the
only regime where disk bandwidth matters; mitigate with NF4 (4× less to read),
`mmap`/page‑cache reuse across micro‑batches, and `K≥3` to hide NVMe latency.
Everything else (no per‑layer GC, dequant‑on‑GPU, stream overlap) is identical.

---

## 9. Summary answer to the question

- **Do we keep it in CPU and move to GPU?** Yes — in the target regime, keep the
  whole quantized model **resident in pinned CPU RAM** and stream layers to the
  GPU on a copy stream, double‑buffered, so transfer hides behind compute.
- **Where does latency go?** Today it goes to **disk hashing and per‑layer
  memory reclamation**, not PCIe. Remove those and overlap the transfers, and a
  step goes from ~19 h to ~tens of seconds.
- **On this specific A100 box**, the 32B‑4bit model fits in VRAM, so the
  genuinely fastest option is to **not stream at all** (Regime A, plain QLoRA);
  use streaming only for models that exceed VRAM.
