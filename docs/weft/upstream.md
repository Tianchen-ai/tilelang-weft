# Native frontend and shared compiler seams

[Handbook](index.md) · [中文](upstream.zh-CN.md)

The fork starts at TileLang v0.1.15 and retains its pinned TVM submodule.
Upstream GPU backends keep their existing pipeline/codegen path.

| Location | Responsibility |
|---|---|
| `tilelang/language/eager/ast.py` | Common rewriting, return regions and declaration discovery |
| `tilelang/language/eager/builder.py` | Existing scopes, arguments, callbacks and locations |
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

