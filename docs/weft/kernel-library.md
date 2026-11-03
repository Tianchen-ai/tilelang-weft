# Normal author programs

[中文](kernel-library.zh-CN.md) · [Handbook](index.md)

The library is `tilelang.weft.kernels`. Its [catalog](https://github.com/Tianchen-ai/tilelang-weft/blob/main/tilelang/weft/kernels/catalog.py)
stores source functions and finite source defaults. Selecting a catalog entry
invokes its ordinary TileLang JIT factory; target passes never dispatch on these names.

| Family | Source module | Author structure |
|---|---|---|
| Pointwise, SiLU, SwiGLU, softplus, exp/log, selection | `activation` | Independent Parallel tiles and explicit tail guards; stable sigmoid/softplus macros |
| RMSNorm, LayerNorm, residual RMSNorm | `normalization` | Row snapshots, reductions and explicit affine epilogues |
| Softmax, split/interleaved RoPE | `attention` | Stable row softmax and explicit rotary index choices |
| Online attention | `attention.online_attention` | Query stage, m/l/o state, bounded key Levels and explicit tail score mask |
| KV update and causal grouped-query attention | `attention` | Caller-owned cache, populated prefix and head/group mapping |
| GEMM and batched projection | `linear` | Author blocks, transpose, bias/SiLU epilogue and local engine requirement |
| Q4_K/Q6_K × Q8_K | `quantized` | Shared Encoding fields, typed grouped partials, scale/correction branches |
| Integer-field dequantization | `quantized.dequantize` | Q1, Q4/Q5/Q8, K formats and TQ2 fields; explicit numerical decoding |
| Lookup and sub-byte formats | `codebook` | IQ1/IQ2/IQ3/IQ4, MXFP4/NVFP4 and TQ1 author programs with explicit tables |
| Activation preparation | `preparation` | Caller-owned Q8_0/Q8_1/Q8_K output fields, rounding and group sums |
| Packed projection | `composition.packed_projection` | Explicit decode and dense projection programs with caller-owned workspace; prepare/compute can be invoked separately |
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
Persistent interleave requires a separate preparation kernel, demonstrated by
[example 05](https://github.com/Tianchen-ai/tilelang-weft/blob/main/examples/weft/05_derived_layout.py).

The product owns these normal programs. Bench owns workloads, external references,
numerical comparisons and timing boundaries. Public comparative performance is
**Under evaluation / 测试中**.
