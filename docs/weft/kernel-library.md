# Normal author programs

[中文](kernel-library.zh-CN.md) · [Handbook](index.md)

The library is `tilelang.weft.kernels`. Its [catalog](https://github.com/Tianchen-ai/tilelang-weft/blob/main/tilelang/weft/kernels/catalog.py)
stores source functions and finite source defaults. Selecting a catalog entry
invokes its ordinary TileLang JIT factory; target passes never dispatch on these names.

| Family | Source module | Author structure |
|---|---|---|
| Pointwise, SiLU, SwiGLU, softplus, exp/log, selection | `activation` | Independent Parallel tiles and explicit tail guards; stable sigmoid/softplus macros |
| RMSNorm, LayerNorm, residual RMSNorm | `normalization` | Explicit row reductions and bounded streaming affine epilogues |
| Softmax, split/interleaved RoPE | `attention` | Stable row softmax and explicit rotary index choices |
| Online attention | `attention.online_attention` | Query stage, m/l/o state, bounded key Levels and explicit tail score mask |
| KV update and causal grouped-query attention | `attention` | Caller-owned cache, populated prefix and head/group mapping |
| GEMM and batched projection | `linear` | Author blocks, transpose, bias/SiLU epilogue and local engine requirement |
| Dense CPU panels | `panel_dense` | In-call panel transposes, optional F16 activation rounding and an explicit outer-product loop |
| Batched packed and codebook projections | `panel_projection`, `indexed_projection` | Decode a weight record once, reuse across activation rows, and close integer scale/correction groups |
| Q4_K/Q6_K × Q8_K | `quantized` | Shared Encoding fields, typed grouped partials, scale/correction branches |
| Integer-field dequantization | `quantized.dequantize` | Q1, Q4/Q5/Q8, K formats and TQ2 fields; explicit numerical decoding |
| Lookup and sub-byte formats | `codebook` | IQ1/IQ2/IQ3/IQ4, MXFP4/NVFP4 and TQ1 author programs with explicit tables |
| Activation preparation | `preparation` | Caller-owned Q8_0/Q8_1/Q8_K output fields, rounding and group sums |
| Packed projection | `composition.packed_projection` | Explicit decode and dense projection programs with caller-owned workspace; prepare/compute can be invoked separately |
| Complete packed dot and projection, 24 formats | `packed.vec_dot`, `packed.projection` | Record-local decode, author-declared partials, per-record activation scale and preparation inside projection |
| IME1/IME2 and AMX/VNNI | `matrix`, `layouts` | Explicit persistent layouts and local contraction engines |
| GGML row deployment | `deployment.vec_dot` | Dynamic K, typed View ABI, Q4_K/Q6_K scale branches and checked role binding |
| MoE | `moe` | Top-k routing, stable grouping, ragged expert ranges and weighted combination |
| State-space programs | `state` | Welford, Top-K, selective recurrence and caller-owned causal history |

For example:

```python
from tilelang.weft.kernels.catalog import get, programs

print([program.name for program in programs(family="normalization")])
kernel = get("matmul").compile(M=5, N=19, K=37,
    block_m=4, block_n=16, block_k=16, engine="vector")
kernel(a, b, c)
```

Static blocking choices describe the author program. Runtime scalar inputs,
Tensor shape/stride binding, cached compilation and output allocation continue
through TileLang. More than one factory may implement the same mathematical
family with different author layouts; that is distinct from a compiler selecting
local instructions for one program.

Shared format declarations are defined once in `weft_core.formats.ggml` and
adapted to TileLang by `ggml_formats`. Tables remain explicit input storage.
`packed.projection` uses explicit integer block partials and F32 scale updates.
Q4_K and Q6_K retain typed integer
partials and separate scale/correction branches through the shared `partials`
macros. `composition.packed_projection` separately exposes `acc_dtype` for its
explicit decode-to-workspace and dense multiplication program. `linear.matmul`
factory separately declares input, accumulator and output types; its existing
default remains F32. Target passes preserve these numerical choices.
Persistent interleave requires a separate preparation kernel, demonstrated by
[example 05](https://github.com/Tianchen-ai/tilelang-weft/blob/main/examples/weft/05_derived_layout.py).

`row_partials` provides source-selected deferred row reductions and grouped
integer products for x86. `scale_groups` in the single-row Q6_K factory bounds
the author's simultaneously exposed scale groups; it does not select an ISA.
The indexed x86 variant consumes the fixed GGML codebooks and prepared Q8_K
activations in [-127, 127], allowing exact sign transfer before I16 pair products.
The indexed program reconstructs the stored even-parity signs from seven bits.
`preparation.quantize(chunk=...)` exposes a finite 32/64/128/256-element source
window: the block maximum is shared, while the author chooses the quantization
and group-sum lifetime. Numeric View ranges such as `weights.q[:, 32:64]`
retain their named axes and lower to Canonical subviews. `matrix.k_layout_projection`
consumes the explicit GGML interleaved model layout and includes activation
quantization in the same call.

`subblock_partials` exposes independent record accumulators and typed integer
partials for short records, Q6_K and nonlinear grids. NVFP4 retains unsigned
biased codes and I32 quad products before scaling. IQ3_XXS shares metadata
and sign rows across its explicit group, entry and payload axes. Indexed programs keep entry and
payload axes separate through encoded table reads and flatten their result with
ordinary `T.reshape`. Dequantizers write explicit bounded output windows; their
`chunk` parameter selects the author window rather than an ISA. GGML programs
declare the reference's disjoint-output contract while allowing read-only
inputs to share storage, so snapshot and supply passes can use those facts.

Panel workspaces are invocation storage. `panel_dense.projection` populates
`Scratch` and `PackedActivation` during the call; `round_f16` chooses whether
activation is rounded to F16 before F32 accumulation. Packed projections also
quantize activation during the call. Codebook panels retain explicit integer
corrections and populate `ScalePanel` and `CorrectionPanel` before their row
consumers. `decode_columns` bounds the author's decoding cohort independently
of the matrix tile. Loads after these writes capture the populated panels;
`T.stage` instead defines a snapshot in its Level's initialization phase.
`ActivationSums` and `GroupSums` retain each activation group's correction sum
across output-column tiles. Their preparation remains inside the projection.
Factories specialize optional workspaces to zero elements when unused, and
native metadata retains those source Tensor extents.

`subblock_partials.block32(partials=1)` expresses a scalar dot and scale update
for each record. `partials=8` retains eight grouped products across records.
These are explicit numerical organizations; target passes select the physical
implementation of the requested organization.

The product owns these normal programs. Bench owns workloads, external references,
numerical comparisons and timing boundaries. Public comparative performance is
**Under evaluation / 测试中**.

The independent [TileLang–Weft Bench](https://github.com/Tianchen-ai/tilelang-weft-bench)
binds 150 workload IDs to 553 platform cases, sharing the fixed references with
Triton–Weft. Its full suites select ordinary factories from this package.
[Example 13](https://github.com/Tianchen-ai/tilelang-weft/blob/main/examples/weft/13_packed_dot.py)
shows a complete packed call with explicit input bytes. `T.load(value, at=point)`
projects a numeric field along its existing named Level axis; compressed fields
retain the Encoding's record/group relationship.
