# 普通作者程序

独立的 [TileLang–Weft Bench](https://github.com/Tianchen-ai/tilelang-weft-bench) 将 150 个
workload ID 绑定到 553 个平台用例，与 Triton–Weft 共用固定参考。
[示例 13](https://github.com/Tianchen-ai/tilelang-weft/blob/main/examples/weft/13_packed_dot.py)
展示明确输入 bytes 的完整 packed 调用。`T.load(value, at=point)` 沿 Value 已有的具名
Level 轴投影字段，压缩字段保留 Encoding 的 record/group 关系。

[English](kernel-library.md) · [工具书](index.zh-CN.md)

程序库位于 `tilelang.weft.kernels`。[catalog](https://github.com/Tianchen-ai/tilelang-weft/blob/main/tilelang/weft/kernels/catalog.py)
保存实际源码函数和有限的作者参数。选择条目调用普通 TileLang JIT factory，目标 pass
不按这些名字分派实现。

| 类别 | 源码模块 | 作者表达的结构 |
|---|---|---|
| Pointwise、SiLU、SwiGLU、softplus、exp/log、选择 | `activation` | 独立 Parallel tile、显式尾部及稳定 sigmoid/softplus macro |
| RMSNorm、LayerNorm、residual RMSNorm | `normalization` | 显式行归约和有界流式 affine epilogue |
| Softmax、split/interleaved RoPE | `attention` | 稳定行 softmax 和明确 rotary 坐标 |
| Online attention | `attention.online_attention` | query stage、m/l/o 状态、key Level 和尾部 score mask |
| KV 更新、causal GQA | `attention` | 调用者 cache、已填充前缀及 head/group 映射 |
| GEMM、批量投影 | `linear` | 作者分块、转置、bias/SiLU 后处理及局部 engine 约束 |
| CPU dense panel | `panel_dense` | 调用内转置 panel、可选 F16 activation 舍入及显式 outer-product 循环 |
| 批量 packed 与 codebook 投影 | `panel_projection`、`indexed_projection` | 每个权重 record 解码一次，跨 activation 行复用，闭合整数 scale/correction 分组 |
| Q4_K/Q6_K × Q8_K | `quantized` | 共同 Encoding 字段、有限位宽 partial、scale 与校正支路 |
| 整数字段解码 | `quantized.dequantize` | Q1、Q4/Q5/Q8、K 格式和 TQ2 的显式数值解码 |
| Lookup、sub-byte 格式 | `codebook` | IQ1/IQ2/IQ3/IQ4、MXFP4/NVFP4、TQ1 与显式表输入 |
| Activation preparation | `preparation` | 调用者 Q8_0/Q8_1/Q8_K 输出字段、舍入和分组和 |
| Packed 投影 | `composition.packed_projection` | 显式解码和 dense 投影两个程序，workspace 归调用者；可分别 prepare/compute |
| 24 格式完整 packed dot 与投影 | `packed.vec_dot`、`packed.projection` | record 内解码、作者声明的 partial、逐 record activation scale，投影内包含准备 |
| IME1/IME2 与 AMX/VNNI | `matrix`、`layouts` | 显式持久布局及局部 contraction engine |
| GGML 行部署 | `deployment.vec_dot` | 动态 K、typed View ABI、Q4_K/Q6_K scale 支路和明确角色绑定 |
| MoE | `moe` | Top-k 路由、稳定分组、不等长 expert 范围和加权合并 |
| 状态程序 | `state` | Welford、Top-K、selective recurrence 和调用者卷积历史 |

例如：

```python
from tilelang.weft.kernels.catalog import get, programs

print([program.name for program in programs(family="normalization")])
kernel = get("matmul").compile(M=5, N=19, K=37,
    block_m=4, block_n=16, block_k=16, engine="vector")
kernel(a, b, c)
```

静态分块描述作者程序。运行时标量、Tensor shape/stride 绑定、编译缓存和输出分配继续
使用 TileLang。相同数学类别可以有不同作者布局；编译器则为同一程序选择局部机器实现。

格式声明统一位于 `weft_core.formats.ggml`，`ggml_formats` 将其接入 TileLang；
lookup 表仍是显式输入。持久 interleave 由独立准备程序生成，完整流程见
[示例 05](https://github.com/Tianchen-ai/tilelang-weft/blob/main/examples/weft/05_derived_layout.py)。

`packed.projection` 使用显式整数 block partial 和 F32 scale 更新。
Q4_K、Q6_K 通过共享 `partials` 宏保留整数 partial 和独立的
scale/correction 支路。`composition.packed_projection` 的显式解码到 workspace
再做 dense 乘法程序另有 `acc_dtype` 参数。普通 `linear.matmul` 分别声明输入、局部操作数、累加和
输出类型，原有默认仍为 F32；目标 passes 保留这些数值选择。

`row_partials` 提供 x86 的作者级延迟归约与分组整数计算。单行 Q6_K 的
`scale_groups` 限定作者同时暴露的 scale group 数量，不选择 ISA。
索引 x86 变体消费固定 GGML codebook 与范围为 [-127, 127] 的已准备 Q8_K
activation，将符号精确转移到 activation，再显式进行 I16 成对乘积。
索引程序由已存储的七位重建偶校验符号。`preparation.quantize(chunk=...)`
公开 32/64/128/256 元素的有限源码窗口：共享 block 极值，作者决定量化与分组和
的生命周期。`weights.q[:, 32:64]` 等数值 View 范围保留具名轴并进入 Canonical
subview。`matrix.k_layout_projection` 消费明确的 GGML interleaved 模型布局，
在同一次调用中完成 activation 量化。

`subblock_partials` 为短 record、Q6_K 和非线性 grid 显式表达独立 record 累加与
整数 partial；NVFP4 使用无符号偏移编码和 I32 四项乘积后再缩放。IQ3_XXS 显式保留
group、entry、payload 轴，让 metadata 和符号行在对应范围内复用。索引程序在编码表
读取中分别保留 entry、payload 轴，最后通过普通 `T.reshape` 展平。行解码直接写出
有界输出窗口，`chunk` 表示作者窗口，不选择 ISA。GGML 程序声明与参考一致的
输出独立存储合同，同时允许只读输入共享存储，供快照与供给 passes 使用。

Panel workspace 属于单次调用的存储。`panel_dense.projection` 在调用内填充
`Scratch` 和 `PackedActivation`；`round_f16` 决定 activation 是否先舍入到 F16，
之后均按 F32 累加。Packed 投影也在调用内量化 activation。Codebook panel 保留
显式整数校正，在行消费者开始前填充 `ScalePanel` 与 `CorrectionPanel`。
`decode_columns` 限定作者的解码 cohort，与矩阵 tile 分开。写入后的普通 load
捕获已填充的 panel；`T.stage` 则在所属 Level 的初始化阶段定义读取快照。
`ActivationSums` 与 `GroupSums` 将各 activation group 的校正和复用于多个输出列 tile，
准备工作仍在 projection 调用内完成。Factory 将无需使用的可选 workspace 特化为零个
元素，native metadata 保留相应的源码 Tensor 长度。

`subblock_partials.block32(partials=1)` 表达每条 record 的标量 dot 与 scale 更新；
`partials=8` 则跨 record 保留八个分组乘积。分组程序一次读取完整 record，再投影字段，
使 packed sign plane 与 payload 一起保留；标量程序直接读取选定字段。两者均明确保留
源码的读取边界。这些是作者明确选择的数值组织，目标 passes
负责实现所选组织的物理计算。

产品仓库维护普通程序，Bench 管理 workload、外部参考、数值比较与计时范围。
公开性能结果为 **测试中 / Under evaluation**。
