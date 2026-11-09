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
| Packed subblock projection | `subblock_projection` | Keep a Q1/NVFP4/TQ2/IQ4_XS weight cohort across activation rows, with explicit subblock scales and in-call quantization |
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

`linear.blocked_matmul` closes each complete K contraction inside explicit
`tile_m`/`tile_n` Levels and `rows`/`columns` output cohorts. It uses ordinary
`T.gemm` with the weight's storage order retained. `linear.matmul` separately
exposes a tiled K loop and a grid of independent output programs.
`projection.staged_f16_projection` converts activation to the caller's F16
workspace inside the call, stages a weight-column cohort and an activation-row
cohort, then uses ordinary full-K `T.gemm` for each output tile. The source
parameters separately expose those reuse scopes and the output tile dimensions.

`attention.rope(head_major=True)` traverses each head's contiguous pair range;
the default traverses pair tiles first and stages coefficients across the head
cohort. Both use the same rotary macro. `causal_attention` stages a typed query
View and carries its online numerator as state. Its token/head cohort parameters
bound the serial work per grid instance without changing the causal KV prefix.

Shared format declarations are defined once in `weft_core.formats.ggml` and
adapted to TileLang by `ggml_formats`. Tables remain explicit input storage.

`codebook.iq1_s(chunk=32/64/128/256)` retains separate entry and eight-element
payload axes. Entry scale and delta are shared across their payload; a typed
I8X8 table View supplies the lookup. IQ2_XS/XXS likewise retain entry/sign rows
and use U16 table offsets, whose complete ranges fit their declared tables.
The final reshape preserves the output's row-major order and the per-entry
floating-point formula.
`codebook.iq1_windows` and `iq2_windows` expose nested record, scale-group and
eight-element table-row Levels. Each entry selects one contiguous I8X8 codebook
row; IQ2 selects its sign row independently.
The scale metadata belongs to the original record snapshot, and the output
retains the public `(M, K)` layout. The author shares metadata within these
bounded windows without preparing a persistent decoded array.
IQ2 codebook and sign rows contain I8 elements. `iq2_windows(product_dtype=...)`
selects an explicit I16 or I32 product before conversion to F32. Both represent
the complete I8-by-I8 product range exactly. I16 exposes widening multiply and
widening integer-to-float conversion; I32 can keep short payloads in the same
RVV configuration as their F32 consumers. The author selects this numerical
organization independently of the target's physical LMUL choice.
`codebook.indexed_windows` uses the same bounded table-row structure for IQ1_M,
IQ2_S and IQ3_XXS. The IQ1_M half-bit lookup and the IQ2_S packed signs remain
explicit source inputs and fields. `iq3_windows` pairs two four-element table
rows inside an eight-element sign window, sharing its scalar scale at the
32-element parent Level. Its sign operation consumes the original packed-bit
field, and its public output remains `(M, K)`.
`quantized.field_windows` decodes an integer-format record through explicit
Level subtiles. A scale-group window selects its payload and scalar metadata
from the same record snapshot, reusing the regular decoding macro. This source
organization exposes the Encoding's field relation without creating a separate
decoded record array or changing the caller's stored layout.
`chunk` can subdivide that scale group into 8/16/32-element source windows.
An entire short record uses its existing record Level directly.
Q1 and TQ2 also accept their complete 128/256-element record as a window. Their
signed codes stay in I8 until floating conversion. Affine group windows compute
their scalar scale and minimum coefficients before the payload calculation and
reuse them across the window.
For Q2_K/Q3_K/Q6_K the window program converts the scalar group scale before
the payload multiplication. The author makes that numerical organization
explicit, allowing one scalar scale conversion to serve the whole window.
Its Q6_K decoder keeps the reconstructed code in I8, whose exact range is
[-32, 31], until conversion to F32. The common decoding macro retains an explicit
`code_dtype` source argument; the ordinary whole-record program uses I32.
`indexed_lanes.xxs_subgroups` expresses IQ3_XXS as 64-element Level windows.
Each window retains two integer scale groups, codebook rows and sign rows before
the block sum. Explicit pair-entry, pair and payload axes preserve each sign
row's eight elements when it becomes two four-element codebook rows.
The signed payload and activation values meet in the ordinary
`T.reduce_dot`; this exposes local reuse without a full decoded record.
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
`q6_planes` also accepts multiple output rows. Its serial row loop retains the
same plane/lane partials and per-record scaling for each row, without exposing
all rows as simultaneously resident values.
The shared `q6_plane_value` macro accepts `accumulation="block"` to finish the
exact I32 record sum before its F32 update, or `"lanes"` to retain F32 lane
sums until the row ends. `scaled_projection(organization="block_planes")`
uses that same block-scaled macro in a non-inlined row call after in-call
activation preparation. `lanes` and `planes` remain explicit author dimensions.
Its `rows_per_program` optionally partitions those rows into independent CPU
programs. Each retains the same plane/lane calculation, with the full native
grid covering every row. The default keeps all rows in one program.
`subblock_partials.block32` and the indexed `vec_dot`, `scaled_entries` and
`xxs_subgroups` factories expose the same row-cohort choice. The explicit grid
changes the program's traversal boundary, while each factory reuses its existing
row macro and preserves input preparation and numerical grouping.

For RISC-V, `row_partials.affine_groups` expresses Q2_K/Q4_K/Q5_K scale groups
as Level subtiles, keeping the minimum correction in a separate integer branch.
`scaled_groups` uses 16-term subtiles for Q3_K and Q6_K. Each group reads payload
and scalar metadata from the original record snapshot, performs its I32 partial,
and applies its integer scale before the record's floating-point update.
The affine `scale_projection` accepts `organization="groups"` for Q2_K to
reuse those explicitly closed group dots after in-call Q8_K preparation.
Its default `"lanes"` keeps the alternative bounded I16 lane accumulation.
`scaled_planes` instead retains the explicit `(4, 2, 16)` payload organization
inside a 128-element sub-Level for each half-record dot, with independent plane/group partials and their
integer scales. These factories expose different source reuse scopes; neither
changes the stored record or the final per-record floating update.
`q2_lanes` and `scaled_lanes` close each row in a common numerical helper.
Their row traversal and complete projection call that same helper, so a row's
numerical lanes finish before the next row begins. Q2 retains four exact scaled
products per I16 lane before widening;
Q3/Q6 retain I32 group-scaled lanes and the per-record F32 update. This is an
author numerical structure. The Q2 `scale_projection` and `scaled_projection`
include activation quantization in their original call. Q2's projection closes
each output row in `T.macro(noinline=True)`, keeping its complete numerical
lanes within that call. Q3/Q6 close the activation-row batch in their existing
non-inlined helper.
`staged_projection` can retain the same lane organization inside its declared
weight-column and activation-row stages. Q2 uses 16 exact I16 lanes; Q3/Q6
choose 16 or 32 scaled I32 lanes. The default grouped-plane organization remains
available. Short affine projections update their minimum and product branches
directly into the F32 state, exposing two local FMA opportunities without an
extra combined contribution temporary.
Its multi-output affine tiles retain one I32 sum per output across the record's
16/32-element groups. A saved coefficient field preserves its Encoding group
projection; the minimum correction remains a separate dot against Q8_K sums.
`short_projection.staged_projection` puts its entire staged compute traversal
in one non-inlined macro call after preparation, retaining the same Level tree
and natural input ABI. The call closes preparation temporaries without adding
a call at each output tile or K record.
`subblock_projection.projection` uses the same explicit `batch_rows`,
`staged_columns`, `rows` and `columns` reuse scopes for Q1/NVFP4. The output
tile is state across all K records and is stored once. NVFP4 retains separate
16-element codebook dots and subblock scales; Q1 retains its centered I8 codes.
Activation quantization and local encoded stages remain inside the invocation.
The TQ2_0 variant closes one signed 256-term dot per weight record, sharing the
same output state and encoded reuse scopes. Its Q8_K preparation and workspace
remain explicit inside the projection.
IQ4_XS uses the same stages with 32-element codebook dots, explicit low/high
scale fields and a signed I32 group factor. Its Q8_K preparation is also counted
inside the invocation.
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

`codebook.metadata_windows` stages the 16-element codebook and keeps IQ4_XS
or NVFP4 scale metadata scalar in each 32/16-element Level window. The payload
uses an indexed read of that saved table. `ternary_planes` reuses TQ1's
32-byte, 16-byte and four-byte payloads across independent digit/payload axes,
then reshapes each decoded plane into its contiguous output window. Its power
table is an explicit staged input; the payload loads remain in the record body.

The shared Q4_1/Q5_1 decoder expresses scaling and minimum correction in one
affine expression. Q5 reconstructs its five-bit code in U8 before widening the
complete value. These source expressions expose the legal local FMA and avoid
wide bitfield work without changing the decoded range or output layout.

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
`indexed_projection.projection` exposes the same `extreme_policy` and
`inverse_policy` source parameters as its row-program counterparts. Selecting
in-call integer panels retains the workload's activation quantization contract;
panel construction and its workspaces remain part of the complete invocation.

`subblock_partials.block32(partials=1)` expresses a scalar dot and scale update
for each record. `partials=8` retains eight grouped products across records.
The grouped program reads each complete record once and projects its fields,
retaining packed sign planes alongside the payload. The scalar program reads
the selected fields directly. Both keep the source read boundary explicit.
These are explicit numerical organizations; target passes select the physical
implementation of the requested organization.

The same `block32_rows` macro is used by `block32` and the complete
`subblock_partials.projection` for Q4/Q5/Q8, IQ4_NL and MXFP4 short records.
Projection quantizes its activation inside the call before invoking that row
program. `records` and `partials` retain the same source grouping in both
factories; `extreme_policy` and `inverse_policy` preserve the preparation contract.
IQ4_XS likewise shares `iq4_group_rows` between its dot and projection, retaining
the explicit grouped partials and codebook snapshot through both entry points.

`short_projection.staged_projection` consumes the original encoded weight layout.
The author selects an outer weight-column stage, an activation-row stage, and a
smaller row/column output tile whose state remains live over K. This reuses the
staged bytes without persistent repacking. Activation quantization stays inside
the call; the same View, Level and contraction path supports the target's chosen
physical vector layout.

`indexed_lanes.staged_projection` applies the same outer column and activation-row
stages to indexed records. It calls the existing IQ1/IQ2/IQ3/IQ4 row macros,
retaining their group partials, explicit codebook/sign inputs and record scaling.
`batch_rows`, `staged_columns` and `groups_per_step` describe source reuse and
grouping; Q8_K quantization remains inside the invocation. The stages retain
natural encoded storage rather than requiring persistent weight preparation.

`subblock_partials.window_dot` uses closed integer dots within each encoded
record. Q1_0 retains four Q8_0 subrecord scales before applying its weight scale;
the grouped `subblocks` variant also leaves activation in I8 and supplies
explicit signed weight codes to widening grouped MAC, preserving the -128 case.
NVFP4 shares the staged codebook while closing each 16-element scaled product;
TQ2_0 closes one 256-element signed dot. `rows_per_program` selects the author's
row cohort, and the native grid covers all cohorts.
`rows_per_program` can retain several complete rows inside one instance.
`window_projection` reuses the same macro after quantizing activation into the caller-provided `Quantized`
buffer during the call. Its preparation cost and output layouts stay explicit.

`row_partials.scale_projection` reuses `affine_group_rows` for Q2_K/Q4_K/Q5_K,
retaining separate scaled integer and minimum-correction sums. Q2_K keeps its
minimum coefficients in I16 and explicitly contracts them with the Q8_K group sums.
Both scale and indexed projection factories expose the existing activation
extreme and inverse policies, so a workload keeps its required quantization
contract while reusing a row program.

`row_partials.staged_projection` reuses the same affine or signed grouped row
macros across bounded weight-column and activation-row stages. It covers the
five K formats without a decoded panel or persistent repacking. The public
`batch_rows` and `staged_columns` bound the author's reuse, and Q8_K preparation
remains inside the complete projection.

For Q3_K/Q6_K, explicit `rows` and `columns` also keep a two-dimensional
output tile live across K. A decoded 64-term weight window serves each
16-term contraction with the activation-row cohort; integer scale updates
close before the record's floating update. These are source reuse dimensions,
independent of the backend's SIMD width and register layout.

Grouped affine short records retain the weight scale/minimum and activation scale/sum as
two-element metadata tiles. Their conversion and multiplication are expressed
in one `T.Parallel(2)` operation before the independent product and correction
updates. This keeps the author's metadata work grouped through the common
frontend, alongside the explicitly grouped integer payload products.
The scalar-record program instead adds its product and correction within each
record before updating its floating sum, matching that source accumulation order.
Q4_1 and Q8_0 consume the saved payload field directly when no numerical decode
is needed, retaining its Encoding width through contraction.

`indexed_projection.packed_words` builds byte panels during the call and reuses
them across activation rows. Group factors, biased products and the exact
activation-sum correction remain separate integer expressions. Its `rows` and
`columns` parameters describe that reuse; the common compiler selects VNNI or
another legal vector implementation. `ternary_planes` and `iq1_planes` retain
their group/entry/payload axes until the author's final reduction.
`iq1_planes(format="IQ1_S"/"IQ1_M", groups_per_step=...)` bounds the free
group cohort independently of entry/payload contraction. Both formats use one
shared row macro. IQ1_M retains separate grid and signed-delta partials and its
explicit half-bit lookup; IQ1_S retains its activation-group sum correction.

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

`panel_projection.staged_affine` retains encoded Q4_K/Q5_K weights across an
explicit activation-row cohort. Its `staged_columns`, `block_k`, `batch_rows`,
`rows` and `columns` parameters define the author's reuse boundaries. Q8_K
activation preparation stays inside the call; a full-K output tile starts from
zero, while partial K tiles explicitly accumulate the stored output. The source
keeps I32 scale updates, grouped products and pair-sum corrections visible.

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

`indexed_lanes.vec_dot` and `indexed_lanes.projection` share the ordinary
`indexed_rows` macro for IQ1_S/IQ1_M/IQ2_XS/IQ2_XXS/IQ2_S. The author retains
sixteen-lane integer partials, I32 group accumulation, explicit codebook inputs
and per-record floating scales. `groups_per_step` bounds source unrolling
without specifying a hardware vector width; sixteen-group formats also accept
`groups_per_step=16`. Projection prepares its caller's
Q8_K buffer once inside the invocation; `extreme_policy` preserves the selected
quantization tie rule before the same indexed row program consumes that buffer.

The bounded `scaled_entries_rows` and `xxs_subgroups_rows` macros likewise serve
their dot factories and `indexed_lanes.entry_projection`. The same entry/payload
Levels, codebook snapshots and integer scale groups survive in the projection;
activation quantization remains inside its invocation. `row_partials.scaled_projection`
reuses `scaled_planes_rows` or `scaled_lanes_rows` for Q3_K/Q6_K. Its `organization`
parameter selects block integer sums or deferred lane partials in the author
program, independently of target vector width.
The lane row helper has an explicit native call boundary; projection reuses
that same complete K reduction after its in-call activation preparation.

IQ1_M's indexed row program reconstructs its F16 scale with ordinary
`T.reinterpret`, then closes each signed numerator dot with `T.reduce_dot`.
The scale, delta, integer accumulation and final eighth-unit factor remain
explicit numerical operations in the source.

`indexed_lanes.iq1_scaled` expresses the IQ1 codebook alphabet `{-1,0,1}` as
an eighth-unit integer numerator. Its signed I16 product includes the odd
group scale before widening to I32, followed by one record sum. The same
`iq1_scaled_rows` macro is available in `projection` and `staged_projection`
with `numerical_form="scaled"`; Q8_K preparation remains inside the call.
An explicit biased unsigned two-bit representation preserves a bound on that
codebook range for shared integer analysis, rather than treating it as arbitrary I8.
The bias uses byte arithmetic before the two-bit conversion, whose low bits
are unchanged by byte wraparound. Odd group factors stay in I8 until their
explicit conversion at the I16 product.

The staged IQ1 program also accepts `rows` and `columns` output tiles.
Its `groups_per_step` bounds the groups computed inside each record sub-Level;
the exact I32 subtotal carries across these windows before the record's F32
scale update. Weight and activation stages retain their declared outer reuse
scopes, while each local contraction preserves shared group and independent
row/column axes.

For single-row/single-column output issues, `numerical_form="planes"` in the
staged IQ1 factory reuses `iq1_plane_rows`: entry/payload integer dots and the
delta correction remain separate. The staged factory reconstructs the F16
block scale with `T.reinterpret`; the standalone factory retains its explicit
scale-table input contract.
