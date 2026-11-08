# Native frontend and shared compiler seams

[Handbook](index.md) · [中文](upstream.zh-CN.md)

The fork starts at TileLang v0.1.15 and retains its pinned TVM submodule.
Upstream GPU backends keep their existing pipeline/codegen path.

| Location | Responsibility |
|---|---|
| `tilelang/language/eager/ast.py` | Common rewriting, return regions and declaration discovery |
| `tilelang/language/eager/builder.py` | Existing scopes, arguments, callbacks and locations |
| `tilelang/language/allocate.py` | Statement-level CPU allocations at their author's lexical position |
| `tilelang/backend/module.py` | Exclusive artifact compiler and CPU iteration ownership contract |
| `tilelang/engine/lower.py` | Selected backend checks and artifact compilation |
| `tilelang/weft/frontend` | Native TIRx tile/control/model import into shared Canonical |
| `src/weft` | Register target and explicit source-model intrinsics |
| `tilelang/weft/pipeline.py` | Installed opaque compiler ABI and target toolchain |
| `tilelang/jit` and `tilelang/cache` | Existing JITKernel and atomic native-artifact persistence |
| `tilelang/profiler` and `tilelang/autotuner` | CPU timer and shared alias-preserving trial storage |
| `tilelang/weft/kernels` | Ordinary author programs |

`libweft_compiler` owns its MLIR contexts and modules behind C handles. TVM and
MLIR C++ objects do not cross the boundary. The shared native runtime is independent
of the compiler and both frontends. Target passes have one source owner in
Triton–Weft's `third_party/weft` subproject.

Weft helper and Level phases use native TIRx attribute regions, rather than
scheduled tensor blocks. Local buffers are allocated where declared, including
inside helpers and birth regions. The importer builds reads, writes, captures
and handoffs from their typed statements and View operations; it does not use
tensor-block access inference for opaque Encoding handles.
Nested Levels inherit a tail from an enclosing Level on the same logical
axis. Divisibility of a parent's nominal partition does not prove its final
active window is full; shared target analysis may refine that fact later.

The shared `NormalizeCanonicalControl` pass inlines helpers with one final return
and no `noinline` constraint, including their explicit buffer updates. Nonlocal
returns and explicit `noinline` constraints retain typed function regions. Both
frontends use this normalization before physical layout and resource analysis.

`T.gemm` retains each operand's storage-coordinate order. Transpose flags bind
the explicit M/N/K axes of the Canonical outer contraction; they do not insert
a separate numerical transpose. Encoded fields and saved reads therefore keep
their original supply relation for the shared physical compiler.

Local Tensor indexing retains Canonical extraction coordinates, including
Encoding fields. Explicit `T.load(table, at=indices, bounds="in_bounds")` remains
the indexed-table operation. An integer scalar index returns a zero-dimensional
local Value; an index tile retains its shape and axes. Both use the same
Canonical lookup and explicit bounds contract. A state's birth Level determines its lifetime;
its numerical lanes use that Level's coordinate only when their extent matches
the declared partition. A 16-lane partial born in a one-row Level keeps its own
numerical axis.

Native artifacts retain the source Tensor shapes separately from logical View
layouts. This also represents specialized zero-sized optional arguments with
no memory effects, without introducing a dummy read or write.

