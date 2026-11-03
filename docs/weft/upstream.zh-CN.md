# 原生前端与共同编译核心的接缝

[English](upstream.md) · [工具书](index.zh-CN.md)

fork 基于 TileLang v0.1.15，保留该版本固定的 TVM 子模块。
上游 GPU 后端继续使用原有的 pipeline/codegen。

| 位置 | 职责 |
|---|---|
| `tilelang/language/eager/ast.py` | 共同 AST 改写、返回区域和声明发现 |
| `tilelang/language/eager/builder.py` | 原有作用域、参数、callback 与源码位置 |
| `tilelang/backend/module.py` | 独占 artifact compiler 接缝与 CPU 迭代所有权 |
| `tilelang/engine/lower.py` | 选定后端的校验和 artifact 编译 |
| `tilelang/weft/frontend` | 原生 TIRx tile/control/model 导入共同 Canonical |
| `src/weft` | 注册目标与明确的源模型 intrinsic |
| `tilelang/weft/pipeline.py` | 安装后的不透明 compiler ABI 与目标工具链 |
| `tilelang/jit`、`tilelang/cache` | 原有 JITKernel 和原生产物的原子持久化 |
| `tilelang/profiler`、`tilelang/autotuner` | CPU timer 与保留别名关系的私有试跑存储 |
| `tilelang/weft/kernels` | 普通作者程序 |

TIRx 在 tile 展开前保留 Encoding、Level、state/stage 和数值边界；importer
将它们导入共同 Canonical，再调用同一份目标 passes。物理布局、指令、packing、
流水与资源决定继续由共同核心完成。

`libweft_compiler` 通过 C handle 持有自己的 MLIR context/module，TVM 与 MLIR
的 C++ 对象不跨 ABI。共同 native runtime 与编译器和两种前端相互独立。
目标 passes 的唯一源码归属为 Triton–Weft 的 `third_party/weft` 子项目。
