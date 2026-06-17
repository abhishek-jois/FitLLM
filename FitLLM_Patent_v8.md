**UNITED STATES PATENT APPLICATION** 

## **FitLLM** 

**Method and System for Adaptive Multi-Shard Parallel Inference and On-Device Training of Large Language Models Across a Three-Tier Memory Hierarchy on Consumer and Edge Hardware** 

|**Field**|**Detail**|
|---|---|
|IDF Reference|FITLLM-2026-002(Revised)|
|Applicaton No.|[TBD — To Be Assigned at Filing]|
|FilingDate|[DATE — To Be Completed Before Filing]|
|Assignee|QpiAI|
|Status|Pre-FilingInital Draf — Atorney-Client Privileged|
|Prior Applicaton|No. 144678-0132 — Method and System for Hardware-<br>Aware Deployment<br>and Optmizaton of Large Language Models (QpiAI,<br>fled 2025)|
|Base System|Extends AirLLM(Apache-2.0,lyogavin/airllm)|
|Implementaton|Working prototype — inference + QLoRA training<br>validated on LLaMA family|
|Performance Basis|All throughput projectons are analytcal / frst-<br>principles estmates.<br>Empirical benchmarks are pending and will be added<br>before fling.|



**CONFIDENTIAL — ATTORNEY-CLIENT PRIVILEGED** 

## **1.  Title of Invention** 

**Method and System for Adaptive Multi-Shard Parallel Inference and On-Device Training of Large Language Models Across a Three-Tier Memory Hierarchy on Consumer and Edge Hardware** 

## **2.  Field of the Invention** 

This invention relates to systems and methods for executing inference and training of large language models (LLMs) on resource-constrained computing hardware including consumer-grade devices with limited GPU video random-access memory (VRAM), CPU-only computing devices without a discrete GPU accelerator, and edge inference nodes with heterogeneous memory hierarchies. More particularly, the invention relates to adaptive memory orchestration across a three-tier hierarchy comprising non-volatile storage (NVMe SSD or equivalent), central processor random-access memory (CPU RAM), and accelerator memory (GPU VRAM, NPU SRAM, or equivalent fast on-chip memory), enabling inference and full parameter training of models with one billion to over one trillion parameters on devices with as little as 4 gigabytes of accelerator memory, and on devices with zero dedicated accelerator memory using CPU-only execution. 

## **3.  Background of the Invention** 

Large language models including the LLaMA 3.x, Qwen3, Mistral, Gemma 3, Phi-4, and DeepSeek-V3 families have demonstrated transformative capability across natural language tasks. These models range from 1 billion to over 670 billion parameters, with sparse MoE architectures reaching effective scales beyond one trillion parameters. Deployment on consumer hardware — typically 4–24 GB GPU VRAM — is severely constrained: a 70B-parameter model in FP16 requires ~140 GB VRAM for inference alone; full fine-tuning requires over 1.26 TB. Prior art systems address this gap only partially: 

- **AirLLM** (Li, 2023): layer-sharding inference, hardcoded k=1 shard regardless of available accelerator memory; no training; no speculative decoding; HuggingFace-specific. 

- **PIPO** (Liu et al., arXiv:2504.03664, 2025): pipelined offloading with a pipeline configured once at initialisation — does not recompute per token, does not adapt to KV cache growth; no training; no speculative decoding. 

- **FlexGen** (Sheng et al., ICML 2023): LP-based continuous placement fractions for batch throughput; OPT-specific; no per-token recomputation; no training. 

- **SpecExec** (Svirschevski et al., NeurIPS 2024): speculative decoding with static probability tree (fixed K); no rolling-window acceptance-rate control; no draft KV persistence; no training. 

- **Fiddler** (Kamahori et al., ICLR 2025): CPU-GPU orchestration routing MoE expert weights to CPU for sparse-expert execution; targets MoE architectures only; does not apply to dense transformer shardstreaming; no training; no speculative decoding. 

- **SpecOffload** (Song et al., arXiv:2505.10259, 2025): attention computation on CPU, FFN computation on GPU simultaneously with fixed layer-type assignment; does not adapt assignment dynamically per token; targets inference only; requires all attention weights in CPU RAM simultaneously. 

- **QLoRA** (Dettmers et al., NeurIPS 2023): requires full quantized model in GPU memory — infeasible at 4–8 GB VRAM for 70B+ models; no shard-streaming backward pass. 

- **ZeRO-Offload** (ATC 2021): multi-GPU distributed training; requires collective communication; no pershard independent NVMe file design. 

No prior system simultaneously addresses: (1) per-token dynamic shard width adapted to real-time available memory; (2) on-device training of 70B+ models with no minimum accelerator memory; (3) dynamic-K speculative decoding within shard-streaming; (4) heterogeneous CPU+accelerator compute placement decided per shard-batch; and (5) hardware-agnostic operation across GPU, NPU, Apple Silicon, and CPU-only configurations. 

## **4.  Summary of the Invention** 

FitLLM extends the AirLLM layer-sharding baseline in four orthogonal directions: 

- **Adaptive Multi-Shard Engine (AMSE):** probes accelerator memory and CPU RAM after each generated token; computes k = floor((free_accel_mem − margin) / shard_size); selects single_shard / multi_shard / full_model automatically. Unlike PIPO, recomputes k per token as KV cache grows. Operates on zero accelerator memory (CPU-only mode) by setting k=0 and routing all computation to CPU RAM. 

- **Complete On-Device Training System:** shard-wise reverse-order backward pass with mathematically provable accelerator memory ceiling; disk-resident per-shard AdamW optimizer on NVMe with no minimum accelerator memory requirement; QLoRA integration (~2 MB/layer gradient storage vs ~540 MB full — analytical). 

- **Speculative Decoding with Dynamic K Control:** Dynamic K Controller with rolling acceptance-rate EMA (distinct from static-K systems in Qualcomm US20250245430A1 and SpecExec); O(n) draft KV cache persistence (100× reduction at n=200 — mathematical identity). 

- **Heterogeneous CPU+Accelerator Shard Placement:** the AMSE dynamically assigns each shard batch either to accelerator compute or to CPU compute based on real-time profiling of available accelerator memory, CPU RAM, and per-layer arithmetic intensity from the roofline model. Unlike Fiddler (static MoE expert routing) and SpecOffload (static attention-CPU/FFN-GPU split), FitLLM's placement decisions are made per shard-batch per generated token, adapting to KV cache growth and co-resident workloads without user configuration. Activations from CPU-computed shard batches are transferred to accelerator memory for subsequent accelerator-computed batches via PCIe DMA. 

**Note:** _All throughput projections are first-principles analytical estimates pending empirical validation._ 

## **5.  Brief Description of the Drawings** 

|**Figure**|**Descripton**|
|---|---|
|FIG. 1|Full System Architecture — Four-Layer Component Map<br>with Three-Tier MemoryHierarchy|
|FIG. 2|Forward Pass — Three-Stage Pipeline Overlap (NVMe /<br>CPU RAM / Accelerator Memory)|
|FIG. 3|Backward Pass — Reverse-Order Gradient<br>Computaton;Provable Accelerator MemoryBound|
|FIG. 4|LoRA Architecture — Frozen W₀ on NVMe; Trainable<br>A/B in CPU RAM|
|FIG. 5|Speculatve Decoding — Dynamic K Controller; Prior Art<br>Distnctons|
|FIG. 6|Theoretcal Bandwidth Reducton Analysis — Six Factors<br>(analytcal only)|



|FIG. 7|Memory Layout — NVMe / CPU RAM / Accelerator<br>MemoryBudget Across All Operations|
|---|---|
|FIG. 8|Complete Training Loop — Forward, Backward, Disk-<br>Resident Optimizer Step|
|FIG. 9|Dynamic Shard Width k vs. Sequence Length — FitLLM<br>vs. PIPO vs. AirLLM(analytical)|
|FIG. 10|Roofline Analysis — Arithmetic Intensity by Layer Type;<br>Basis for Shard Placement|
|FIG. 11|Dynamic K Controller Convergence — Simulated for<br>Two Prompt Types|
|FIG. 12|Draft KV Cache Persistence — O(n²) vs. O(n) Draft<br>ComplexityAnalysis|



## **6.  Detailed Description of the Preferred Embodiments** 

## **6.1  System Architecture Overview** 

FIG. 1 illustrates the full system architecture across four functional layers (L0–L3) and three memory tiers. L0 is the Hardware Profiler; L1 the Adaptive Shard Scheduler (AMSE); L2 the KV Cache Orchestrator and Speculative Decoding engine; L3 the Memory Hierarchy. 

**Figure 1  — Full System Architecture** _L0 Hardware Profiler → L1 Adaptive Shard Scheduler → L2 KV Cache Orchestrator → L3 Memory Hierarchy. Red annotations mark key distinctions from PIPO, SpecExec, Qualcomm, Fiddler, and SpecOffload prior art._ 

The principal software components are: 

**Component** 

**Purpose** 

|Adaptve Shard Probe|Measures free accelerator memory and CPU RAM afer<br>each token; computes parallel shard count k; selects<br>executon strategy|
|---|---|
|Shard Scheduler|Prefetch pipeline, pinned memory allocaton, async I/O,<br>evicton; assigns each shard batch to accelerator or CPU<br>compute ter|
|Forward Engine|Layer-sequental forward pass with actvaton<br>checkpointngto CPU RAM|
|Backward Engine|Reverse shard-wise gradient computaton; reloads CPU<br>RAM actvatons; provable accelerator memoryceiling|
|Shard Optmizer|Per-shard disk-resident AdamW: load shard state from<br>NVMe, apply update, write back; no minimum<br>accelerator memoryrequirement|
|LoRA Manager|Inject A/B adapter matrices, freeze base weights, route<br>training gradients|
|Accelerated Inference|Speculatve decoding + Dynamic K Controller + draf KV<br>persistence + six acceleraton techniques|
|Model Analyzer|Framework-agnostc architecture registry for 30+<br>transformer families; shard splitng without model<br>instantaton|
|Integrity Verifer|SHA-256 checksum write at shard creaton; verify<br>before deserializaton;CLI validaton command|
|Checkpoint Manager|Save/restore adapter weights + optmizer state +<br>dataloader sample index|



## **6.2  Adaptive Multi-Shard Engine (AMSE) with Accelerator-Agnostic Memory Probing** 

**Distinction from PIPO:** PIPO profiles hardware once at initialisation. FitLLM's AMSE calls an accelerator memory introspection API after every generated token and recomputes k. FIG. 9 shows analytically that k decreases from ~53 at token 0 to ~2 at token 4096 (8 GB GPU, LLaMA 70B Q4 — analytical estimate). 

```
# Accelerator-agnostic memory probe (primary: CUDA GPU)
if backend == 'cuda':    free_accel = torch.cuda.mem_get_info()[0] / 1e9
elif backend == 'mps':   free_accel = torch.mps.current_allocated_memory() /
1e9
```

```
elif backend == 'cpu':   free_accel = 0.0   # CPU-only mode: k=0
else:                    free_accel = vendor_specific_memory_query() / 1e9
```

```
gpu_n      = floor( (free_accel_gb  −  0.75 GB margin) / shard_size_gb )
cpu_n      = floor( (free_cpu_gb    −  1.50 GB margin) / shard_size_gb )
parallel_n = max(0, min(gpu_n, cpu_n))   ← recomputed after EVERY generated
token
# parallel_n = 0 triggers CPU-only shard execution mode
```

**Figure 9  — Dynamic Shard Width k vs. Sequence Length** _Analytical: FitLLM k decreases from ~53 to ~2 as KV cache grows over 4096 tokens (8 GB GPU, LLaMA 70B Q4). PIPO static k=8 (OOM risk at long sequences); AirLLM fixed k=1 (wastes ~98% available memory). All values are analytical estimates._ 

**Figure 10  — Roofline Analysis: Arithmetic Intensity by Layer Type** _Attention layers (circles, teal) always memorybandwidth bound (b_i ≈ 1 FLOP/byte — CPU-friendly). FFN layers (triangles, amber) shift toward compute-bound with batch size above b_crit ≈ 18. This roofline-derived divergence is the first-principles basis for heterogeneous shard placement (§6.5)._ 

## **6.3  Forward Pass — Three-Stage Pipeline Overlap** 

FIG. 2: the Shard Scheduler overlaps GPU compute of shard batch i, PCIe DMA of batch i+1, and NVMe read of batch i+2, driving effective time toward max(T_compute, T_PCIe, T_NVMe) as a theoretical upper bound. Every hidden state hᵢ is written to CPU RAM as an activation checkpoint. 

**Figure 2  — Forward Pass: Three-Stage Pipeline Overlap** _GPU compute (teal), PCIe DMA (blue), NVMe read (amber) overlapping. Activation checkpoints hᵢ written to CPU RAM at each boundary. Effective time approaches max(T_compute, T_PCIe, T_NVMe)._ 

## **6.4  Alternative Embodiment: Direct NVMe-to-Accelerator Transfer (GPUDirect Storage)** 

**Empirical observation:** On PCIe topologies where the NVMe controller and GPU share the same PCIe switch, the effective bandwidth of NVMe→CPU→GPU (pinned staging buffer) and NVMe→GPU direct DMA are comparable. This enables an alternative embodiment where CPU staging is bypassed entirely. 

NVIDIA GPUDirect Storage (GDS), available via the cuFile API in CUDA 12.x and the nvidia-fs kernel module, enables DMA engines in NVMe controllers to write directly into GPU BAR1 memory without a CPU bounce buffer. In this alternative embodiment: 

- The Shard Scheduler issues cuFileRead() calls targeting GPU VRAM addresses directly from NVMe. 

- The CPU staging buffer (pinned CPU RAM) is eliminated, freeing CPU RAM for larger activation checkpoint storage — extending maximum supported sequence length. 

- PCIe bandwidth utilisation is not reduced by this path on topologies where NVMe and GPU share the same PCIe root complex; on asymmetric topologies, the system falls back to CPU staging. 

This alternative embodiment does not change the AMSE shard width formula, the speculative decoding pipeline, or the training loop — only the I/O path for shard loading is affected. GDS requires: NVIDIA RTX/data-center GPU with BAR1 aperture support; CUDA 12.x; NVMe drive on the same PCIe switch as the GPU. 

## **6.5  Heterogeneous CPU+Accelerator Per-Shard-Batch Dynamic Compute Placement** 

**Distinction from Fiddler and SpecOffload:** Fiddler (ICLR 2025) statically routes MoE expert weights to CPU at model load time. SpecOffload (2025) statically assigns all attention layers to CPU and all FFN layers to GPU for the entire generation session. FitLLM's Heterogeneous Placement Engine (HPE) makes a new placement decision for each shard batch after each generated token, guided by the AMSE hardware probe output and the per-layer arithmetic intensity from the roofline model. 

The placement decision for shard batch i at token t is: 

```
b_i    = arithmetic_intensity(layer_type_i)  # from FIG. 10 roofline
b_crit = peak_FLOPS(backend) / mem_bandwidth(backend)
```

```
if free_accel_gb > shard_size_gb * 1.2:  # headroom check
    placement = 'ACCELERATOR'  # GPU/NPU compute
else:
    placement = 'CPU'          # CPU RAM compute via torch CPU backend
# Dynamic result: at long contexts (large KV cache), AMSE may route
# later shard batches to CPU while earlier batches still use GPU.
# This is novel: no prior system makes per-shard-batch per-token
# heterogeneous placement decisions for dense transformer layers.
```

When a shard batch is assigned to CPU compute: the shard weights are loaded from NVMe into CPU RAM (avoiding the GPU memory bus entirely); the forward computation executes via PyTorch CPU operators; the resulting hidden state tensor is transferred to accelerator memory via PCIe for the subsequent accelerator-assigned batch. This enables continuous inference even as GPU VRAM is fully consumed by the KV cache, degrading gracefully to CPU-only execution rather than crashing with OOM. 

The CPU compute path is also available in zero-VRAM mode (k=0), enabling full LLM inference on computing devices with no discrete GPU accelerator — laptops, Raspberry Pi, edge MCU boards with sufficient CPU RAM — with the entire execution occurring in CPU RAM. 

## **6.6  Backward Pass — Shard-Wise Reverse-Order Gradient Computation** 

FIG. 3: the backward pass streams layers in reverse order with a provable accelerator memory ceiling. At any point only N shard tensors and two activation tensors coexist in accelerator memory: 

```
VRAM_backward ≤ N × shard_size_quant + 2 × activation_size + overhead
(mathematical invariant)
```

**Figure 3  — Backward Pass: Reverse-Order Shard-Wise Gradient Computation (Claim 18)** _Six-step procedure with CPU RAM activation reuse. Accelerator memory bound: N×shard + 2×act. For LLaMA 70B Q4, k=1: 575 MB (analytical). No prior training system executes a shard-streaming backward pass._ 

## **6.7  Parameter-Efficient QLoRA Integration** 

FIG. 4: frozen W₀ on NVMe in NF4 form; trainable A and B with output = W₀·x + (α/r)·B(Ax). Only adapter gradients accumulate — analytically ~2 MB/layer at rank-16 for 70B-scale model. 

**Figure 4  — LoRA Architecture: Frozen W₀ on NVMe; Trainable A/B in CPU RAM** _Frozen W₀ (NVMe); trainable A and B (CPU RAM). Backward: gradients only for A and B (~2 MB/layer vs ~540 MB — analytical at rank-16)._ 

## **6.8  Speculative Decoding with Dynamic K Controller** 

**Distinctions from Qualcomm US20250245430A1, SpecExec, and SpecOffload:** All three use fixed K. FitLLM's Dynamic K Controller maintains a circular buffer of W=50 per-step acceptance rates, computing rolling mean ᾱ, and adjusts K within [k_min, k_max]. FIG. 11 shows simulated convergence. Draft KV cache persistence (FIG. 12) reduces draft compute from O(n²) to O(n) — mathematical identity. 

**Figure 5  — Speculative Decoding with Dynamic K Controller** _Draft phase (small model, VRAM-resident) with persistent KV cache. Single shard traversal verifies all K tokens at approximately constant I/O cost. Rolling EMA K controller is novel over all prior static-K systems._ 

**Figure 11  — Dynamic K Controller Convergence** _Simulated: instruction-following K→k_max=12; adversarial prompts K→k_min=2. Prior art fixed K=4 (dashed)._ 

**Figure 12  — Draft KV Cache Persistence: O(n²) → O(n)** _100× reduction at n=200 tokens — mathematical identity. No prior speculative decoding system persists draft KV across decode steps._ 

## **6.9  Theoretical Bandwidth Reduction Analysis** 

**Note:** _FIG. 6 presents analytical reduction factors only — not empirical benchmarks. Actual performance depends on hardware, workload, and implementation._ 

**Figure 6  — Theoretical Bandwidth Reduction Analysis** _Cumulative analytical factors from AirLLM baseline: NF4 quantization, AMSE multi-shard, speculative decoding, pinned memory, fused kernels, adaptive layer skipping. All theoretical._ 

## **6.10  Memory Layout and Budget** 

**Figure 7  — Memory Layout: Three-Tier Budget Across All Operations** _Analytical: Forward Pass, Backward Pass, Optimizer Step — contents of each tier for LLaMA 70B LoRA rank-16._ 

## **6.11  Disk-Resident Per-Shard Optimizer File Layout** 

`shards/ layer_NN_weights.safetensors    ←  NF4 inference weights layer_NN_master_fp32.pt         ←  FP32 master weights for AMP layer_NN_optstate.pt            ←  { m: tensor, v: tensor, step: int } layer_NN_gradients.pt           ←  accumulated dL/dW (deleted after step) layer_NN_weights.sha256         ←  SHA-256 integrity checksum` 

Peak CPU RAM during optimizer step: analytically 3 × shard_size_fp32, independent of total model size. No minimum accelerator memory requirement. 

## **6.12  Complete Training Loop** 

**Figure 8  — Complete Training Loop Flowchart** _Three sequential phases: Forward (k shards, activations to CPU RAM, loss); Backward (reverse shards, CPU RAM activations reloaded, gradients to NVMe); Optimizer Step (AdamW from disk, requantize NF4, write back)._ 

## **6.13  Framework-Agnostic Model Architecture Registry (30+ Families)** 

The Model Analyzer maps model_type identifier strings (from HuggingFace config.json) to (layers_attr, embed_attr,  head_attr)  triples  for  30+  transformer  families  without  model  instantiation  or  GGUF conversion. The registry is organised by architectural class: 

|**Architecture Class**|**Model Families (model_type**<br>**identifiers)**|**Notes**|
|---|---|---|
|Dense Decoder-Only<br>(Standard Attention)|llama, qwen2, qwen3, mistral,<br>gemma, gemma3, phi3, falcon, yi,<br>granite,stablelm,cohere|~70% share identical attribute paths|
|Dense Decoder-Only<br>(Variant Attention)|bloom, gpt2, gpt_neox, opt,<br>chatglm, internlm2, baichuan,<br>starcoder2, phi4,llama4|6 distinct attribute path patterns|
|Mixture-of-Experts|mixtral,deepseek_v3, qwen3_moe,|MoE requires expert-aware shard|



|(Sparse Actvaton)|deepseek_v2,<br>mixtral_8x22b,switch_transformers|splitng|
|---|---|---|
|Emerging Architectures<br>(Recent 2024–2025)|deepseek_v3, llama4, gemma3,<br>phi4, cohere2,<br>granite_moe|Added Q2 2025 per HuggingFace<br>v4.51.0|



The registry enables zero-config model onboarding: given a HuggingFace model directory, the Model Analyzer reads config.json, resolves model_type to the appropriate attribute tuple, and performs shard splitting directly from safetensor checkpoint files — supporting any of the 30+ covered families without code changes. 

## **6.14  Per-Shard Cryptographic Integrity and Checkpoint Resume** 

The Integrity Verifier computes SHA-256 at shard creation and verifies before deserialization, raising an error identifying the corrupted shard by name. The Checkpoint Manager saves adapter weights + optimizer state + dataloader sample index at configurable intervals for exact-step resume. 

## **7.  First-Principles System Analysis** 

## **7.1  Optimal Shard Width — Derivation** 

```
k*(t) = floor( (V_total − V_overhead − V_KV(t)) / shard_size_w )
V_KV(t) ≈ t × num_kv_heads × head_dim × 2 × dtype_bytes / 1e9  (grows per
token)
LLaMA 70B Q4 (w ≈ 0.135 GB), 8 GB GPU — analytical estimates:
  t = 0:    k* ≈ floor((8.0 − 0.75) / 0.135) ≈ 53
  t → ∞:   k* → 0 → triggers CPU-only execution mode (novel vs PIPO)
```

## **7.2  Backward Pass Accelerator Memory Ceiling — Proof** 

```
VRAM_backward ≤ N × w_quant + 2 × act_size + ε  (mathematical invariant of the
procedure)
```

At any point, accelerator memory holds exactly N shard tensors, h_{i-1} and h_i from CPU RAM, and overhead ε. No other tensors are resident by construction of the backward procedure. 

## **7.3  O(n²) → O(n) Draft Complexity — Mathematical Proof** 

```
Without KV persistence: total = 1 + 2 + ... + n = n(n+1)/2 = O(n²)
With KV persistence:    total = n               = O(n)
Ratio at n = 200:  (200×201/2) / 200 = 100.5×   (exact arithmetic)
```

## **7.4  Optimal K for Speculative Decoding — Analytical** 

```
TPS(K) = (1 − α^{K+1}) / ((1−α) × (T_v + K × T_d))
In shard-streaming: T_v >> T_d → K* → k_max unless α is low.
Dynamic K Controller approximates K*(t) per token without explicit T_v, T_d
knowledge.
```

## **7.5  Roofline-Guided Heterogeneous Shard Placement — Analysis** 

FIG. 10 shows that at batch size b=1 (token decode), all transformer layer types operate in the memorybandwidth-bound regime. The critical batch size b_crit = peak_FLOPS / mem_bw is approximately 18 for a mid-range GPU in FP16. Since decode-phase inference always uses b=1, all layers are below b_crit, meaning: 

```
For any layer type in decode phase (b=1):
  GPU: bandwidth-bound, time ≈ param_bytes / GPU_mem_bandwidth
  CPU: bandwidth-bound, time ≈ param_bytes / CPU_mem_bandwidth
```

```
  Ratio = GPU_mem_bandwidth / CPU_mem_bandwidth ≈ 360 GB/s / 50 GB/s ≈ 7.2×
```

```
  → GPU is faster per layer when available
```

```
  → But when GPU memory is exhausted by KV cache, CPU execution is
    preferable to OOM-crashing or reloading from NVMe.
```

```
FitLLM's HPE maximises GPU utilisation first, gracefully degrades to CPU,
rather than static partitioning (Fiddler / SpecOffload approach).
```

## **8.  Claims** 

## **What is claimed is:** 

_Note on scope: 36 claims total — 9 independent, 27 dependent. USPTO base fee covers 20 claims; extra claim fees apply for claims 21-36. Extra independent claim fees apply beyond 3 independent claims. Attorney should review claim count and fee strategy before filing._ 

## **System Claims — Adaptive Multi-Shard Execution Engine** 

## **Claim 1  (Independent)** 

A system for executing inference of a large language model on a single computing device without distributed  data  parallel  collective  communication,  comprising:  a  hardware  profiler  configured  to measure,  after  each  generated  output  token  during  autoregressive  generation,  a  current  available accelerator memory value and a current available CPU RAM value via hardware memory introspection APIs, wherein said accelerator memory comprises at least one of GPU video random-access memory, NPU on-chip memory, Apple Metal allocator memory, or zero in the case of a CPU-only computing device; a shard width calculator configured to compute a parallel shard count k as the floor of available accelerator memory minus a safety margin divided by single transformer layer weight size, wherein k is recomputed after each generated output token such that k decreases as key-value cache tensors consume accelerator memory during generation; a strategy selector configured to assign single-shard when k equals one, multishard when k is greater than one and less than total layer count, full-model when k is greater than or equal to total layer count, and CPU-only when k equals zero; a shard scheduler configured to load k transformer layer weight tensors simultaneously into accelerator memory via asynchronous I/O; and an 

execution engine configured to process input tokens through the k resident layers, evict said layers, and load the next k layers until all transformer layers are processed. 

## **Claim 2  (Dependent on Claim 1 — Three-Stage Concurrent Pipeline)** 

The  system  of  Claim  1,  wherein  the  shard  scheduler  concurrently  executes:  computation  of transformer layers in shard batch i on the accelerator; asynchronous DMA transfer of shard batch i+1 from pinned CPU memory to accelerator memory via a dedicated compute stream; and sequential read of shard batch i+2 from non-volatile storage into pinned CPU memory via a dedicated I/O thread pool; such that effective time per shard batch approaches the maximum of compute time, transfer time, and storage read time. 

## **Claim 3  (Dependent on Claim 1 — KV-Cache-Adaptive Shard Width)** 

The system of Claim 1, wherein the shard width calculator recomputes k after each generated output token by measuring accelerator memory consumed by the key-value cache at that token position, such that k decreases monotonically as the key-value cache grows and the system adapts from multishard to single-shard execution without interrupting generation. 

## **Claim 4  (Dependent on Claim 1 — Graceful Degradation to CPU-Only Execution)** 

The system of Claim 1, wherein when k equals zero the shard scheduler loads transformer layer weight tensors into CPU RAM and executes forward pass computations via a CPU compute backend without requiring accelerator hardware, and wherein the system transitions automatically from accelerator-assisted  to  CPU-only  execution  mid-generation  as  key-value  cache  growth  reduces available accelerator memory to zero, continuing generation without restarting or reloading the model. 

## **Claim 5  (Dependent on Claim 1 — Heterogeneous Per-Shard-Batch Placement)** 

The system of Claim 1, further comprising a heterogeneous placement engine configured to assign each  shard  batch,  independently  after  each  generated  token,  to  accelerator  compute  when accelerator memory headroom exceeds a shard-size threshold, or to CPU compute when headroom is insufficient, transferring resulting hidden state tensors from CPU memory to accelerator memory for subsequent accelerator-assigned batches via PCIe DMA; wherein placement decisions are dynamic per shard-batch per token, distinct from static layer-type routing in Fiddler (Kamahori et al., ICLR 2025) and SpecOffload (Song et al., 2025). 

## **Claim 6  (Dependent on Claim 1 — Direct NVMe-to-Accelerator Transfer)** 

The system of Claim 1, wherein the shard scheduler, on computing devices where the non-volatile storage controller and accelerator share a common PCIe root complex and direct storage access is available, transfers shard data directly from non-volatile storage into accelerator memory via DMA without a CPU RAM staging buffer, and falls back to CPU-staged transfer where PCIe topology does not support direct transfer. 

## **Claim 7  (Dependent on Claim 1 — Dense Transformer Architecture Registry)** 

The system of Claim 1, further comprising a model architecture registry mapping model type identifier  strings  to  tuples  of  transformer  layer  attribute  path,  embedding  attribute  path,  and language model head attribute path for dense decoder-only transformer families including at least: 

llama, qwen2, qwen3, mistral, gemma, gemma3, phi3, phi4, falcon, yi, granite, stablelm, and cohere; and a model analyzer performing shard splitting from safetensor checkpoint files using said registry without loading any tensor into memory and without model-family-specific code branches. 

## **Claim 8  (Dependent on Claim 1 — MoE and Emerging Architecture Registry)** 

The system of Claim 1, further comprising a model architecture registry covering Mixture-of-Experts families  including  mixtral,  deepseek_v3,  deepseek_v2,  qwen3_moe,  and  switch_transformers, resolving  expert-layer  shard  boundaries  separately  from  dense-layer  boundaries;  and  recently introduced families including llama4 and gemma3 supporting forward compatibility via registry update without code changes. 

## **System Claims — Multi-Layer Shard Grouping and I/O Amortisation** 

## **Claim 9  (Independent)** 

A system for reducing non-volatile storage read operations during large language model inference on a single computing device, comprising: a shard width calculator computing parallel layer count k as the floor of available accelerator memory minus a safety margin divided by single transformer layer weight size, wherein k is greater than or equal to one; a shard grouper serialising k consecutive transformer layer weight tensors as a single logical shard unit and issuing a single sequential read from non-volatile storage per shard unit, reducing total non-volatile storage read operations per forward pass from L individual layer reads to the ceiling of L divided by k shard reads; and an execution engine processing all k resident layers before evicting the shard unit and loading the next; wherein on a device with 8 gigabytes of accelerator memory executing a 70-billion-parameter model in 4-bit precision with per-layer weight size approximately  0.135  gigabytes,  k  is  approximately  53,  reducing  storage  reads  from  80  to  2  — approximately 40-fold compared to single-layer loading. 

## **Claim 10  (Dependent on Claim 9 — Per-Token Shard Width Adaptation)** 

The system of Claim 9, wherein the shard width calculator recomputes k after each generated output token, reflecting key-value cache growth reducing available accelerator memory, such that the shard grouper adjusts group boundaries dynamically throughout the generation session. 

## **Claim 11  (Dependent on Claim 9 — Three-Strategy Shard Policy)** 

The system of Claim 9, wherein a strategy selector assigns single-shard execution when k equals one; multi-shard when k is greater than one and less than L; and full-model when k is greater than or equal to L, loading all layers in one shard unit with no further storage reads during the generation session. 

## **System Claims — Speculative Decoding with Training-Free Dynamic K Control** 

## **Claim 12  (Independent)** 

A system for speculative inference of a large language model wherein target model weights are streamed from non-volatile storage, comprising: a draft model maintained fully resident in accelerator memory; a speculative generator configured to generate K candidate tokens and record per-token probability distributions; a verifier executing a single shard-streaming forward pass of the target model over a sequence extended by K candidate tokens, verifying all K tokens simultaneously at a per-traversal storage I/O cost approximately constant with respect to K; an accept-reject sampler operating on the ratio of target to draft model probabilities with corrected sampling upon rejection; and a dynamic K controller 

configured to maintain a circular buffer of the most recent W per-step observed acceptance rate measurements,  compute  a  rolling  mean  acceptance  rate  from  said  buffer,  increment  K  toward  a configured maximum when the rolling mean exceeds an upper threshold, and decrement K toward a configured minimum when the rolling mean falls below a lower threshold; wherein said dynamic K controller requires no additional trained model components, no gradient computation, and no model fine-tuning,  computing  the  rolling  mean  solely  from  binary  accept/reject  outcomes  of  completed speculative steps. 

## **Claim 13  (Dependent on Claim 12 — Draft KV Cache Persistence, O(n) Complexity)** 

The system of Claim 12, wherein the draft model maintains a persistent key-value cache between consecutive decode steps within a generation session, passing only newly generated tokens with the cached key-value state as context at each step, reducing total draft model forward pass tokens from O(n-squared) to O(n) across a generation of n tokens, and resetting the persistent cache at the start of each new generation. 

## **Claim 14  (Dependent on Claim 12 — Constant-Cost Verification Exploitation)** 

The system of Claim 12, wherein K is selected based on the property that one complete shardstreaming forward pass of the target model transfers approximately the same bytes from non-volatile storage regardless of K, due to the fixed total parameter count of the target model, such that effective tokens per unit I/O cost scales approximately linearly with K at acceptance rates above a minimum threshold. 

## **Claim 15  (Dependent on Claims 1 and 12 — AMSE and Speculative Decoding Compounding)** 

The system of Claim 1 incorporating the speculative decoding system of Claim 12, wherein the multishard I/O amortisation of Claim 1 and the K-token-per-traversal amortisation of Claim 12 apply simultaneously, compounding multiplicatively in the regime where storage I/O is the dominant latency component. 

## **Claim 16  (Dependent on Claim 12 — Training-Free K Adaptation, No Prediction Head)** 

The  system  of  Claim  12,  wherein  the  dynamic  K  controller  operates  without  any  acceptance prediction head, without any additional neural network component beyond the existing draft and target models, and without any training signal or gradient computation for K adaptation, relying exclusively  on  the  binary  accept/reject  outcomes  produced  by  the  accept-reject  sampler  of completed  speculative  steps;  in  contrast  to  systems  that  train  an  auxiliary  model  to  predict acceptance probability. 

## **System Claims — On-Device Training with Disk-Resident Per-Layer File Store** 

## **Claim 17  (Independent)** 

A system for training a large language model on a single computing device without distributed data parallel collective communication, comprising: a per-layer file store on non-volatile storage wherein, for each transformer layer, a co-located bundle of independent files is maintained comprising: a quantized weight file for inference forward passes; an FP32 master weight file for mixed-precision parameter updates; an optimizer state file containing first moment tensor, second moment tensor, and step counter; and an accumulated gradient file that is created during the backward pass and deleted after the optimizer 

step; a forward engine loading transformer layer weight tensor groups sequentially and writing each hidden state tensor to CPU RAM at each group boundary; a backward engine loading weight tensor groups  in  reverse  order,  reloading  CPU  RAM  hidden  state  tensors  to  reconstruct  local  automatic differentiation graphs, computing weight gradients, and accumulating normalized gradients to per-layer gradient files; and an optimizer applying bias-corrected AdamW updates to FP32 master weights, requantizing to inference precision, writing all updated files to non-volatile storage, and deleting the gradient file; wherein peak CPU RAM during the optimizer step is bounded to a constant multiple of one layer FP32 weight size independent of total model parameter count, and no minimum accelerator memory is required beyond activation memory. 

## **Claim 18  (Independent — Shard-Wise Reverse-Order Backward Pass)** 

A system for computing gradients of a large language model on a single computing device comprising: a CPU  RAM  activation  store  configured  to  receive  and  retain  hidden  state  tensors  written  at  each transformer layer group boundary during a forward pass; a backward engine configured to iterate transformer layer groups in strictly reverse order from last group to first, and for each group: load the group's quantized weight tensors from non-volatile storage into accelerator memory; reload the stored CPU RAM hidden state tensors for the current and preceding group boundaries into accelerator memory; re-execute the forward computation through the group to reconstruct a local automatic differentiation computation graph; execute reverse-mode automatic differentiation with the incoming upstream gradient tensor to compute weight gradients and an upstream gradient for the preceding group; accumulate normalized weight gradients to per-layer gradient files on non-volatile storage; transfer the upstream gradient tensor to CPU RAM; and evict the group's weight tensors from accelerator memory before proceeding to the preceding group; wherein at any point during the backward pass, accelerator memory holds  exactly  the  current  group  of  weight  tensors,  the  two  reloaded  hidden  state  tensors,  and miscellaneous runtime overhead, and no other tensors, establishing a provable accelerator memory ceiling as a mathematical invariant of the procedure. 

## **Claim 19  (Dependent on Claim 18 — Provable Accelerator Memory Bound)** 

The system of Claim 18, wherein the accelerator memory ceiling is bounded to the product of the current parallel layer count N and the quantized per-layer weight tensor size, plus twice the hidden state tensor size, plus miscellaneous overhead; said bound holding as a mathematical invariant because no additional tensors are resident in accelerator memory by construction of the backward procedure. 

## **Claim 20  (Dependent on Claim 17 — Low-Rank Adapter Integration)** 

The system of Claim 17, wherein the backward engine and optimizer further support parameterefficient fine-tuning by injecting low-rank adapter matrices A and B into each linear layer, computing the forward pass as base output plus scaling factor times B applied to A applied to input, computing gradients  only  for  adapter  matrices  during  the  backward  pass,  storing  adapter  gradient  files separately from base weight gradient files on non-volatile storage, and maintaining adapter gradient tensors for all transformer layers simultaneously in CPU RAM. 

**Claim 21  (Dependent on Claims 17 and 20 — Zero Non-Volatile Storage I/O During LoRA Optimizer Step)** 

The system of Claims 17 and 20, wherein all optimizer state required for the adapter matrix parameter update — comprising first moment tensors, second moment tensors, and step counters for all adapter matrices across all transformer layers — is maintained simultaneously in CPU RAM throughout training, such that the optimizer step for low-rank adapter fine-tuning requires zero nonvolatile storage read or write operations; in contrast to full-parameter fine-tuning wherein each optimizer step requires loading and writing back per-layer optimizer state files from non-volatile storage; wherein for a 70-billion-parameter model at adapter rank 16, total adapter optimizer state for all layers is analytically approximately 320 megabytes. 

## **Claim 22  (Dependent on Claims 17 and 18 — Frozen Base Weight Backward Recomputation)** 

The system of Claims 17 and 18, wherein the backward engine reloads quantized base weight shard groups from non-volatile storage in reverse layer order during the backward pass in order to propagate the upstream gradient signal through the full layer Jacobian, computing the upstream gradient as the product of the incoming gradient and the combined weight matrix comprising the frozen base weights and low-rank adapter contribution, while computing parameter gradients only for adapter matrices; such that the non-volatile storage read pattern during the backward pass is identical in structure to the forward pass — ceiling of L divided by k shard reads in reverse order — regardless of whether full-parameter or adapter-only fine-tuning is performed. 

## **Claim 23  (Dependent on Claim 17 — Checkpoint Resume with Exact Data Position)** 

The system of Claim 17, further comprising a checkpoint manager saving at configurable intervals: serialized  adapter  weight  tensors,  CPU-resident  optimizer  state  snapshots,  a  dataloader  state encoding the last-processed training sample index, and training configuration metadata; and a resume mechanism restoring all state to continue from the exact step count and data position after any interruption. 

## **System Claims — Per-Layer Co-Located File Bundle Architecture** 

## **Claim 24  (Independent)** 

A storage architecture for large language model weight management on non-volatile storage comprising, for each transformer layer of the model, a co-located bundle of independent files stored within a common directory, said bundle comprising: a quantized inference weight file containing transformer layer weight tensors serialized in a self-describing tensor format; an FP32 master weight file containing the same layer's weight tensors in 32-bit floating point for mixed-precision optimizer updates; an optimizer state file containing at least a first moment tensor, a second moment tensor, and a step counter for adaptive moment estimation; an accumulated gradient file created during training backward passes and deleted after each optimizer step, such that the absence of this file indicates completion of the optimizer step for that layer; and a cryptographic integrity checksum file containing a hash of the quantized inference weight file  computed  immediately  after  serialization;  wherein  each  file  in  the  bundle  is  independently addressable and independently writable, enabling partial bundle updates without rewriting the full bundle. 

## **Claim 25  (Dependent on Claim 24 — Gradient File Lifecycle as Training Progress Indicator)** 

The  system  of  Claim  24,  wherein  the  presence  of  the  accumulated  gradient  file  for  a  given transformer layer indicates that the backward pass has completed for that layer but the optimizer 

step has not yet executed, and the absence of the accumulated gradient file indicates optimizer step completion for that layer, enabling fault-tolerant training resume by inspecting file presence without reading file contents. 

## **System Claims — Per-Shard Cryptographic Integrity Verification** 

## **Claim 26  (Independent)** 

A system for ensuring integrity of transformer layer weight shard files comprising: a shard serializer computing a cryptographic hash of each shard file byte content immediately after serialization and writing the hash to a companion checksum file; a shard deserializer computing and comparing the hash before tensor deserialization, raising an error identifying the shard file by name if hashes do not match, preventing silently corrupted weight tensors from reaching the accelerator; and an integrity verifier accessible via command-line interface reporting verified, missing, and corrupted shards. 

## **Method Claims — Adaptive Inference (with Hardware Anchors)** 

## **Claim 27  (Independent Method)** 

A  computer-implemented  method  for  executing  inference  of  a  large  language  model  on  a  single computing device comprising a PCIe-attached non-volatile storage device, a CPU with page-locked RAM, and an accelerator with on-chip memory, the method comprising: after each generated output token during autoregressive generation, querying the accelerator memory management hardware to measure current free accelerator memory; computing a parallel shard count k as the floor of free accelerator memory minus a safety margin divided by single transformer layer weight size; selecting single-shard, multi-shard, full-model, or CPU-only execution based on k; loading k transformer layer weight tensors simultaneously from non-volatile storage into accelerator memory via asynchronous DMA using pagelocked CPU RAM as a staging buffer and a dedicated accelerator compute stream for device-to-device transfer; executing a forward pass through k resident layers while issuing concurrent non-volatile storage read commands for the next k layers; evicting current k layers and activating prefetched layers; and repeating until all transformer layers are processed. 

## **Claim 28  (Dependent on Claim 27 — Speculative Decoding Method)** 

The method of Claim 27, further comprising: generating K candidate tokens using a draft model fully resident in accelerator memory, with a persistent key-value cache retained between decode steps to reduce draft computation from O(n-squared) to O(n); executing one shard-streaming verifier traversal over the extended sequence; accepting or rejecting tokens via accept-reject sampling; and adjusting K via a training-free rolling mean of observed binary accept/reject outcomes. 

## **Claim 29  (Dependent on Claim 27 — CPU-Only Mid-Generation Degradation Method)** 

The method of Claim 27, wherein when k equals zero during an ongoing generation session, the method continues generating tokens by loading transformer layer weight tensors into page-locked CPU RAM and executing forward pass computations on the CPU compute units, maintaining the accumulated key-value cache in CPU RAM, without restarting or reloading the model. 

## **Claim 30  (Dependent on Claim 27 — Heterogeneous Placement Method)** 

The method of Claim 27, further comprising, for each shard group after each generated token, assigning the shard group to accelerator compute when accelerator memory headroom exceeds 

shard group size, or to CPU compute otherwise, and transferring the resulting hidden state tensor to accelerator memory before the next accelerator-assigned shard group via PCIe DMA. 

## **Method Claims — On-Device Training (with Hardware Anchors)** 

## **Claim 31  (Independent Method)** 

A computer-implemented method for training a large language model on a single computing device comprising a PCIe-attached NVMe solid-state drive, page-locked CPU RAM, and an accelerator, without distributed data parallel collective communication, comprising: maintaining on the NVMe solid-state drive, for each transformer layer, a co-located bundle of independent files comprising at minimum a quantized weight file, an FP32 master weight file, an optimizer state file, and an accumulated gradient file; executing a forward pass by loading transformer layer weight tensor groups sequentially via PCIe DMA through page-locked CPU RAM into accelerator memory and writing each hidden state tensor to pagelocked CPU RAM at each group boundary; executing a backward pass by loading weight tensor groups in reverse  order  via  PCIe  DMA,  reloading  CPU  RAM  hidden  state  tensors  into  accelerator  memory, computing weight gradients via reverse-mode automatic differentiation, and accumulating normalized gradients to per-layer gradient files on the NVMe solid-state drive; and executing an optimizer step by sequentially loading each layer's optimizer state and FP32 master weights from the NVMe solid-state drive, applying bias-corrected adaptive moment estimation updates, re-quantizing to inference precision, writing all updated files back to the NVMe solid-state drive, and deleting the gradient file. 

## **Claim 32  (Dependent on Claim 31 — LoRA Training Method)** 

The method of Claim 31, further comprising injecting low-rank adapter matrices with base weights frozen  and  requires_grad  disabled,  computing  gradients  only  for  adapter  matrices  during  the backward pass, maintaining all adapter gradient tensors for all layers simultaneously in page-locked CPU RAM, and performing the optimizer step for adapter parameters entirely in CPU RAM without non-volatile storage I/O. 

## **Claim 33  (Dependent on Claim 31 — Checkpoint Resume Method)** 

The method of Claim 31, further comprising saving checkpoint files comprising serialized adapter weight tensors, optimizer state snapshots, a dataloader sample index, and training configuration, and restoring all state after any interruption for exact-step resume. 

## **Method Claims — Shard Integrity (with Hardware Anchors)** 

## **Claim 34  (Independent Method)** 

A computer-implemented method for ensuring integrity of transformer layer weight shard files stored on a non-volatile storage device, comprising: immediately after serializing each shard file to non-volatile storage, computing a cryptographic hash of the file byte content by reading the file back through the storage device and computing the hash in CPU RAM, and writing the hash to a companion checksum file co-located with the shard file on the same non-volatile storage device; at shard deserialization time, reading the shard file from non-volatile storage into CPU RAM, computing the cryptographic hash in CPU RAM, reading the companion checksum file, comparing computed and stored hashes; and raising an error identifying the shard file by name and path if hashes do not match, prior to any tensor deserialization or transfer to accelerator memory. 

## **Combined Acceleration and Computer-Readable Medium** 

## **Claim 35  (Dependent on Claim 1 — Combined Acceleration Stack)** 

The system of Claim 1, further comprising at least four of the following operating simultaneously: (a) 4-bit NormalFloat block-wise quantization with double quantization of constants; (b) speculative decoding  with  training-free  dynamic  K  control  per  Claim  12;  (c)  draft  model  key-value  cache persistence per Claim 13; (d) heterogeneous CPU and accelerator placement per Claim 5; (e) direct non-volatile  storage  to  accelerator  transfer  per  Claim  6;  (f)  adaptive  layer  skipping  wherein computation is bypassed when cosine similarity of consecutive hidden states exceeds a threshold; and (g) fused accelerator kernels including at least one of FlashAttention-2, fused root mean square normalisation, or fused gated linear unit activation. 

## **Claim 36  (Computer-Readable Medium)** 

A non-transitory computer-readable medium storing instructions that, when executed by one or more processors of a single computing device comprising a PCIe-attached non-volatile storage device and at least one of an accelerator or CPU compute unit, implement the method of any one of Claims 27, 31, or 34. 

## **9.  Abstract** 

FitLLM is a system and method for adaptive multi-shard parallel inference and on-device training of large language models ranging from one billion to over one trillion parameters, across a three-tier memory hierarchy — NVMe storage, CPU RAM, and accelerator memory (GPU VRAM, NPU SRAM, or CPU RAM in zero-accelerator mode) — on consumer and edge hardware. The Adaptive Multi-Shard Engine (AMSE) measures available accelerator memory and CPU RAM after each generated token, computes k = floor((free_accel  − margin)  /  shard_size),  and  selects  from  four  execution  strategies  automatically, including a CPU-only mode (k=0) for devices without dedicated accelerators. Unlike PIPO (Liu et al., 2025), FitLLM recomputes k per token to adapt to KV cache growth (FIG. 9). A new Heterogeneous Placement Engine dynamically assigns each shard batch to accelerator or CPU compute per token — distinct from static routing in Fiddler (ICLR 2025) and SpecOffload (2025). An alternative embodiment uses NVIDIA GPUDirect Storage to transfer shard data directly from NVMe to GPU VRAM, bypassing CPU staging. A disk-resident per-shard AdamW optimizer enables training on any single device with no minimum accelerator memory (proven bound: §7.2). A shard-wise backward pass has a mathematical VRAM ceiling (FIG. 3). A Dynamic K Controller (rolling EMA — distinct from Qualcomm US20250245430A1) and O(n) draft KV persistence (mathematical — FIG. 12) accelerate inference. A framework-agnostic registry covers 30+ transformer families including DeepSeek-V3, Qwen3, Gemma 3, Phi-4, and Llama 4. All throughput projections are first-principles analytical estimates pending empirical validation. 

## **10.  References to Prior Art** 

**Note:** _All references listed. Risk assessments reflect probability of examiner citation and strength of FitLLM's distinguishing arguments._ 

## **10.1  Layer-Sharding and Offloading Inference** 

- **[1] Li, G. (lyogavin).** _"AirLLM."_ GitHub / HuggingFace Blog, 2023. _FOUNDATIONAL BASE. k=1 fixed; HuggingFaceonly; no training; no speculative decoding._ **● NONE** 

- **[2] Liu, Y. et al..** _"PIPO: Pipelined Offloading for Efficient Inference on Consumer Devices."_ arXiv:2504.03664, 2025. _Static pipeline at init; no per-token recomputation; no training; no speculative decoding. Claims 1 and 9 distinguish by per-token dynamic k and k-layer shard grouping._ **● MEDIUM** 

- **[3] Sheng, Y. et al..** _"FlexGen."_ ICML 2023, arXiv:2303.06865, 2023. _Continuous LP placement fractions; batch throughput; OPT-specific; no per-token recompute._ **● MEDIUM** 

- **[4] Svirschevski, R. et al..** _"SpecExec."_ NeurIPS 2024, arXiv:2406.02532, 2024. _Static probability tree, fixed K, no rolling-window K, no draft KV persistence. Claims 12, 13, 16 distinguish on all three dimensions._ **● MEDIUM** 

- **[5]  Kamahori, K. et al..** _"Fiddler: CPU-GPU Orchestration for Fast Inference of MoE Models."_ ICLR 2025, arXiv:2402.07033, 2025. _Static MoE expert routing at model load; MoE-only; no training; no dynamic per-token placement. Claim 5 distinguishes by per-shard-batch per-token dynamic placement for dense models._ **● MEDIUM** 

- **[6] Song, X. et al..** _"SpecOffload: Unlocking Latent GPU Capacity for LLM Inference."_ arXiv:2505.10259, 2025. _Static attention-CPU / FFN-GPU assignment; no dynamic k; no training. Claim 5 distinguishes by per-token dynamic placement._ **● MEDIUM** 

- **[7]  Alizadeh, K. et al..** _"LLM in a Flash."_ ACL 2024, arXiv:2312.11514, 2024. _Apple Silicon flash offloading; platform-specific; no GPU shard parallelism; no training._ **● LOW** 

## **10.2  Granted Patents — Speculative Decoding (CRITICAL)** 

- **[8] Qualcomm Inc..** _"Efficient Speculative Decoding in Autoregressive Generative AI Models."_ US20250245430A1  /  US12373494,  Priority  April  2023,  2023-2025. _Covers  fundamental  accept-reject sampling paradigm. Does NOT claim: training-free rolling-window K adjustment; draft KV persistence; shardstreaming context. Claims 12, 13, 16 distinguish on all three dimensions._ **▲ CRITICAL — MUST CITE** 

## **10.3  Adaptive Speculative Decoding Systems (NEW — MUST CITE)** 

- **[9] Huang,  K.  et  al..** _"SpecDec++:  Boosting  Speculative  Decoding  via  Adaptive  Candidate  Lengths."_ arXiv:2405.19715, COLM 2025, 2024-2025. _Dynamically adjusts candidate length K on the fly using a trained acceptance  prediction  head;  stops  speculation  when  predicted  rejection  probability  exceeds  a  threshold. REQUIRES TRAINING an auxiliary prediction head. Claim 12 distinguishes by training-free rolling EMA on observed binary outcomes (no prediction head, no gradient computation). Claim 16 makes this distinction explicit._ **● MEDIUM** 

- **[10] Lu,  K.-W.  et  al..** _"AdaSD:  Adaptive  Speculative  Decoding  for  Efficient  Language  Model  Inference."_ arXiv:2512.11280, 2025. _Dynamically adjusts generation length and acceptance criteria using token entropy and Jensen-Shannon distance. Training-free but uses entropy-based thresholds rather than rolling acceptance rate EMA. No shard-streaming context._ **● MEDIUM** 

## **10.4  NVMe-Based Training Systems (NEW — MUST CITE)** 

- **[11]  Ren, J. et al..** _"ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning."_ SC21, arXiv:2104.07857, 2021. _NVMe optimizer offloading via DeepSpeed; REQUIRES multi-GPU distributed data parallel collective communication; subgroup partitioning across GPUs; no per-layer independent file bundle; no shard-streaming  reverse  backward  pass.  Claim  17  explicitly  requires  single  device  without  collective communication._ **● MEDIUM** 

- **[12] Tang, Y. et al..** _"Fuyou: Adding NVMe SSDs to Enable and Accelerate 100B Model Fine-tuning on a Single GPU."_ arXiv:2403.06504, 2024. _Single GPU NVMe-based fine-tuning using pipelined activation swapping_ 

_between GPU, CPU, and SSD. Focuses on activation swapping not per-layer optimizer file bundles; no shardstreaming reverse backward pass; no per-layer co-located file bundle with independent gradient lifecycle. Claims 17, 18, 24 distinguish by the specific file bundle architecture and reverse shard backward._ **● MEDIUM** 

- **[13]  Maurya, A. et al..** _"Deep Optimizer States: Towards Scalable Training of Transformer Models Using Interleaved Offloading."_ Middleware 2024, arXiv:2410.21316, 2024. _Splits LLM into subgroups scheduled on CPU or GPU based on performance model; integrates with DeepSpeed multi-GPU infrastructure. Does not implement shard-streaming reverse backward pass; no per-layer independent file bundle; requires distributed framework. Claim 18 (reverse shard backward) is entirely uncontested in this work._ **● MEDIUM** 

## **10.5  Quantization and Fine-Tuning** 

- **[14] Dettmers, T. et al..** _"QLoRA."_ NeurIPS 2023, arXiv:2305.14314, 2023. _Requires full quantized model in GPU. Claim 17 extends to sub-8GB via shard streaming._ **● MEDIUM** 

- **[15]  Rajbhandari, S. et al..** _"ZeRO-Offload."_ ATC 2021, 2021. _Multi-GPU collective communication; CPU RAM optimizer; no per-shard NVMe file design._ **● MEDIUM** 

- **[16] Hu, E. et al..** _"LoRA."_ ICLR 2022, arXiv:2106.09685, 2022. _Foundational technique used as component._ **● NONE** 

- **[17] Frantar, E. et al..** _"GPTQ."_ ICLR 2023, arXiv:2210.17323, 2023. _Quantization technique used as component._ **● NONE** 

## **10.6  Consumer Frameworks** 

- **[18] Gerganov, G. et al..** _"llama.cpp."_ GitHub: ggml-org/llama.cpp, 2023+. _Manual -ngl flag; GGUF-specific; no adaptive probe; no training._ **● MEDIUM** 

- **[19] Chen, H. et al..** _"KTransformers: CPU/GPU Hybrid Inference for MoE Models."_ ACM SOSP 2025, 2025. _MoEspecific CPU/GPU hybrid; static expert routing; no dynamic per-token placement; no training._ **● LOW** 

## **10.7  GPUDirect Storage** 

- **[20]  NVIDIA Corporation.** _"GPUDirect Storage."_ NVIDIA Magnum IO SDK, CUDA 12.x, 2019-2025. _Enabler technology for direct NVMe-to-GPU DMA. Claim 6 integrates GDS as an optional I/O path with dynamic CPUstaging fallback. GDS itself is not claimed._ **● NONE** 

## **10.8  Relationship to Prior QpiAI Application** 

Application No. 144678-0132 (QpiAI, 2025) covers the deployment pipeline: hardware specification ingestion, model/quantization selection, containerized deployment, and thermal monitoring. The present application covers the inference-time and training-time execution layer. No claim overlap. 

**Note:** _§101 advisory: All claims in this application describe specific hardware-software interactions involving PCIe-attached NVMe storage, page-locked CPU RAM, asynchronous DMA transfer, and accelerator memory management — operations that cannot be performed in the human mind and constitute specific improvements to computing technology. Per USPTO August 2025 guidance, claims are anchored to concrete hardware components (PCIe NVMe, pinned RAM, CUDA streams) rather than abstract mathematical processes, supporting subject matter eligibility under 35 U.S.C. § 101._ 

**11.  Points of Novelty Summary** 

|**#**|**Novel**<br>**Contributon**|**Key Distncton**|**Risk**|**Claims**|**Figure**|
|---|---|---|---|---|---|
|1|AMSE — per-<br>token dynamic k<br>with<br>accelerator-<br>agnostcprobing|AirLLM k=1<br>fxed. PIPO statc<br>at init. Extends<br>to CPU-only<br>(k=0).|LOW|1, 3, 27|FIG. 9|
|2|Heterogeneous<br>CPU+Accelerato<br>r per-shard-<br>batch dynamic<br>placement|Fiddler: statc<br>MoE routng.<br>SpecOfoad:<br>statc atn/FFN<br>split. FitLLM:<br>per-shard-batch<br>per-token<br>dynamic<br>decision for<br>dense models.|VERY LOW|5, 29|FIG. 10|
|3|Direct<br>NVMe→GPU via<br>GPUDirect<br>Storage<br>(alternatve<br>embodiment)|No LLM<br>inference<br>framework<br>integrates GDS<br>with shard-<br>streaming and<br>dynamic CPU-<br>stagingfallback.|LOW|6|—|
|4|Disk-resident<br>per-shard<br>AdamW —<br>single device, no<br>collectve comm|QLoRA: full<br>model in GPU.<br>ZeRO-Ofoad:<br>mult-GPU. No<br>prior: per-shard<br>NVMe fles on<br>single consumer<br>device.|VERY LOW|17, 20, 31, 32|FIG. 8|
|5|Shard-wise<br>reverse<br>backward + CPU<br>RAM actvaton<br>reuse|No training<br>system executes<br>shard-streaming<br>backward.<br>VRAM bound is<br>a mathematcal<br>invariant.|LOW|17, 18, 19, 31|FIG. 3|
|6|Speculatve<br>decoding —<br>constant-cost<br>traversal|SpecExec and<br>SpecOfoad:<br>statc K. No<br>system exploits<br>constant I/O per<br>traversal in<br>shard-<br>streaming.|LOW|12, 14, 28|FIG. 5|
|7|Dynamic K —<br>rolling<br>acceptance EMA<br>window|Qualcomm<br>US20250245430<br>A1, SpecExec,<br>SpecOfoad: all<br>statc K. No<br>prior: rolling<br>window +<br>bounded|VERY LOW|12, 16|FIG. 11|



|||control.||||
|---|---|---|---|---|---|
|8|Draf KV<br>persistence —<br>O(n²) → O(n)<br>mathematcal<br>identty|No speculatve<br>decoding system<br>persists draf KV<br>across steps.|VERY LOW|13, 28|FIG. 12|
|9|30+ family<br>framework-<br>agnostc registry<br>(incl. DeepSeek,<br>Qwen3,<br>Gemma3)|AirLLM: HF only.<br>llama.cpp: GGUF<br>only. No system:<br>model_type→a<br>tr tuple for 30+<br>families.|LOW|7, 8|FIG. 1|
|10|Per-shard SHA-<br>256 integrity<br>verifcaton|No LLM<br>framework<br>writes per-fle<br>checksums<br>alongside<br>weight shards.|VERY LOW|26, 34|—|
|11|Mult-layer<br>shard grouping<br>— I/O<br>amortsaton (k<br>layers per NVMe<br>read)|AirLLM loads<br>exactly 1 layer<br>per NVMe read<br>regardless of<br>free memory.<br>FitLLM groups k<br>layers per read,<br>reducing reads<br>from L to<br>ceil(L/k). 40x<br>reducton on 8<br>GB GPU running<br>70B Q4<br>(analytcal).|LOW|9, 10, 11|FIG. 9|
|12|Checkpoint<br>resume with<br>exact<br>dataloader state|HuggingFace/<br>PEFT: weights<br>only, not data<br>positon.|LOW|23, 33|—|



## **12.  Inventor Declaration and Signatures** 

The undersigned declare that: the statements in this application are true to the best of their knowledge and belief; they believe they are the original inventors of the subject matter claimed; they have reviewed and understand the contents; and they acknowledge the duty under 37 C.F.R. § 1.56 to disclose all information known to be material to patentability. 

|**Inventor**|**Assignee —QpiAI**|
|---|---|
|Signature:|Authorized Signatory:|



___________________________ ___________________________ Printed Name: Title: ___________________________ ___________________________ Date: Date: ___________________________ ___________________________ 

