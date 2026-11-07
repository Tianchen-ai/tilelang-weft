# 原生前端与共同编译核心的接缝

[English](upstream.md) · [工具书](index.zh-CN.md)

fork 基于 TileLang v0.1.15，保留该版本固定的 TVM 子模块。
上游 GPU 后端继续使用原有的 pipeline/codegen。

| 位置 | 职责 |
|---|---|
| `tilelang/language/eager/ast.py` | 共同 AST 改写、返回区域和声明发现 |
| `tilelang/language/eager/builder.py` | 原有作用域、参数、callback 与源码位置 |
| `tilelang/language/allocate.py` | CPU 局部 buffer 在作者词法位置形成语句级分配 |
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

Weft helper 和 Level 的三个阶段使用原生 TIRx 属性区域。局部 buffer 在声明处
分配，包括 helper 和初始化区域内部。importer 从有类型的语句及 View 操作建立
读写、capture 和 handoff；不把 Encoding 的不透明句柄交给 tensor block 访存推断。

局部 Tensor 索引保留 Canonical extraction 坐标，包括 Encoding 字段。
显式 `T.load(table, at=indices, bounds="in_bounds")` 继续表示索引表读取。
state 的诞生 Level 规定生命周期；只有数值长度与该 Level 的分区相同时，才使用其坐标。
在单行 Level 内建立的 16-lane partial 保留自己的数值轴。

原生产物分别保存源 Tensor 形状与逻辑 View 布局。这也能表示特化后零长度、没有访存
effect 的可选参数，无需增加虚假的读取或写入。
