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
| Packed subblock projection | `subblock_projection` | Keep a Q1/NVFP4 weight cohort across activation rows, with explicit subblock scales and in-call quantization |
| Short-record interleaved projections | `short_projection`, `layouts` | Author-declared persistent row cohorts, in-call Q8 activation preparation and state retained across K records |
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

`codebook.short_records(records=1/2/4/8)` selects an explicit short-record
cohort. A single record retains scalar metadata and one payload axis; a cohort
uses the Encoding-aware reshape to keep its record and payload axes separate.
The codebook is a saved stage. `cache_hint="streaming"` is a source store hint
whose alignment and alias legality are checked by the target memory pass.

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
The grouped program reads each complete record once and projects its fields,
retaining packed sign planes alongside the payload. The scalar program reads
the selected fields directly. Both keep the source read boundary explicit.
These are explicit numerical organizations; target passes select the physical
implementation of the requested organization.

Affine short records retain the weight scale/minimum and activation scale/sum as
two-element metadata tiles. Their conversion and multiplication are expressed
in one `T.Parallel(2)` operation before the independent product and correction
updates. This keeps the author's metadata work grouped through the common
frontend, alongside the explicitly grouped integer payload products.

`indexed_projection.packed_words` builds byte panels during the call and reuses
them across activation rows. Group factors, biased products and the exact
activation-sum correction remain separate integer expressions. Its `rows` and
`columns` parameters describe that reuse; the common compiler selects VNNI or
another legal vector implementation. `ternary_planes` and `iq1_planes` retain
their group/entry/payload axes until the author's final reduction.

`attention.rope` exposes `rows_per_program`, `heads_per_program` and `pair_tile`.
The grid covers those explicit cohorts. Within each pair Level, coefficient
stages are shared across heads, while each head reads both activation halves
before writing either result. Partial rows, heads and pair tiles retain bounds.

`short_projection.projection` consumes the `Q4_0Rows`, `Q4_1Rows`, `Q5_0Rows`,
`Q5_1Rows`, `Q8_0Rows`, `IQ4_NLRows` or `MXFP4Rows` derived Encoding. These layouts retain each GGML
record and explicitly interleave four-byte payload chunks across a row cohort.
The caller prepares that persistent weight object before invocation. The
projection itself quantizes activation, computes any required integer block
sums, then retains an output tile's state across all K records. Its `rows` and
`columns` parameters describe author reuse, while `engine="vector"` constrains
the local dot. This program shares the same core as ordinary Tensor programs.
The indexed formats load their explicit codebook once and retain the signed
values through a biased unsigned dot plus the exact activation-sum correction.
MXFP4 also consumes its supplied exponent table; neither table is embedded in
the backend. `projection.row_projection` retains F32 lane sums over F16 weights
and the original F32 activation until the final reduction.

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

Short-record quantizers accept `records=1/2/4/8`. A cohort exposes separate
record and payload axes, so the extrema and encoded scales are computed once
per record while the quantized lanes share their inverse. The public Tensor
shape and serialized Encoding remain unchanged. Integer-field dequantizers
use the same finite cohorts; `rows=1` traverses contiguous storage directly,
while larger row groups retain the explicit row reuse program.

`decoding.ternary_digits` is the shared author macro for the positive radix-3
codes used by row dots, dequantization and invocation panels. It spells the
byte-width wrap and the five constant powers in the DSL. `panel_projection`
consumes these codes in its ordinary preparation and contraction program;
preparing the panels remains part of the measured call.

`codebook.tq1_0(chunk=16/32/64/128/256)` bounds each decoded output window.
Its `cache_hint="streaming"` option uses the existing checked store hint for
outputs that will not be consumed by this kernel. The compiler retains the
normal store when the alignment and storage contract do not permit a native
streaming packet; the decoded values and preparation boundary are unchanged.

`panel_projection.projection` exposes `unroll_decode` to expand the finite
column decode loop in the author program. `pairwise=True, engine="vector"`
uses an invocation-local `(pair, column, term)` I16 panel and typed two-term
I32 partials. Activation quantization, panel preparation and contraction remain
inside the call; both choices use the ordinary frontend and shared core.

`row_partials.k_projection` reuses the grouped Q2/Q3/Q4/Q5/Q6 dot body after
in-call Q8_K quantization. The indexed projection wrappers likewise reuse the
same integer partials as their row-dot programs. Source parameters choose
column cohorts and record windows; target passes choose local instructions.

`indexed_lanes.scaled_entries` expresses IQ2_S/IQ3_S entry and payload axes
inside a 64-element Level window. It applies integer scale groups before the
record's floating scale. IQ4_XS uses a staged 16-entry codebook and a 32-element
subtile with an explicit scale update. Field projection consumes the existing
Level point, keeping both the record read and the codebook snapshot visible to
the shared compiler.
