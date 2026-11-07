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
| Packed 子块投影 | `subblock_projection` | 跨 activation 行保留 Q1/NVFP4 权重 cohort，显式子块缩放和调用内量化 |
| 短 record 交错投影 | `short_projection`、`layouts` | 作者声明持久行 cohort，调用内准备 Q8 activation，state 跨 K record 保留 |
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

`linear.blocked_matmul` 在显式 `tile_m`/`tile_n` Level 和 `rows`/`columns` 输出 cohort
内完成整个 K 的缩并，以普通 `T.gemm` 保留权重的存储顺序。`linear.matmul` 则单独表达
分块 K 循环及各输出程序的 grid。

格式声明统一位于 `weft_core.formats.ggml`，`ggml_formats` 将其接入 TileLang；
lookup 表仍是显式输入。持久 interleave 由独立准备程序生成，完整流程见
[示例 05](https://github.com/Tianchen-ai/tilelang-weft/blob/main/examples/weft/05_derived_layout.py)。

`codebook.iq1_s(chunk=32/64/128/256)` 分离 entry 与八元素 payload 轴，
同一 entry 复用 scale 和 delta，通过 typed I8X8 表 View 查值。
IQ2_XS/XXS 同样保留 entry/sign 行，并使用完整范围可容纳于已声明表内的 U16
表偏移。最终 reshape 保持输出行主序和逐 entry 的浮点公式。
`codebook.iq1_windows` 与 `iq2_windows` 显式嵌套 record、scale-group 和八元素表行
Level。每个 entry 选择一个连续的 I8X8 codebook 行，IQ2 另外选择对应
的 sign 行。scale metadata 保留原 record 快照，输出仍为公开的 `(M, K)` 布局。
作者在有界窗口内复用 metadata，无需持久化的解码数组。
IQ2 的 codebook 与 sign 行均为 I8 元素，源码显式以 I16 计算乘积，再转换为 F32。
I16 能精确容纳完整的 I8×I8 乘积范围，共同目标 passes 因而可以直接选择
widening multiply 与 widening 整数到浮点转换。
`codebook.indexed_windows` 将同一表行窗口结构用于 IQ1_M、IQ2_S 与 IQ3_XXS。
IQ1_M 的 half-bit 查表与 IQ2_S 的 packed signs 仍是显式源码输入及字段。
`iq3_windows` 在八元素符号窗口内配对两个四元素表行，并在父层 32 元素 Level
共享标量 scale。符号运算直接消费原 packed-bit 字段，公开输出仍为 `(M, K)`。

`indexed_lanes.xxs_subgroups` 用 64 元素的 Level 窗口表达 IQ3_XXS。
每个窗口保留两个整数 scale group，以及各自的 codebook 行和 sign 行，随后加入 block 和。
显式的 pair-entry、pair、payload 轴保留一行八个 sign 元素与两行四元素 codebook 的对应关系。
有符号 payload 与 activation 通过普通 `T.reduce_dot` 计算，直接暴露局部复用，
不先建立完整的解码 record 数组。
`packed.projection` 使用显式整数 block partial 和 F32 scale 更新。
Q4_K、Q6_K 通过共享 `partials` 宏保留整数 partial 和独立的
scale/correction 支路。`composition.packed_projection` 的显式解码到 workspace
再做 dense 乘法程序另有 `acc_dtype` 参数。普通 `linear.matmul` 分别声明输入、局部操作数、累加和
输出类型，原有默认仍为 F32；目标 passes 保留这些数值选择。

`row_partials` 提供 x86 的作者级延迟归约与分组整数计算。单行 Q6_K 的
`scale_groups` 限定作者同时暴露的 scale group 数量，不选择 ISA。

RISC-V 的 `row_partials.affine_groups` 以 Level 子块表达 Q2_K/Q4_K/Q5_K
的 scale group，并在独立整数支路中计算 minimum 修正；`scaled_groups` 为 Q3_K
和 Q6_K 使用 16 项子块。每组从原 record 快照读取 payload 与标量 metadata，
完成 I32 partial 和整数 scale 更新，再执行 record 的浮点累加。
`scaled_planes` 则在每个半 record 的 128 元素子 Level 中保留 `(4, 2, 16)` payload 组织，显式计算
plane/group partial 及其整数 scale。两个 factory 表达不同的源码复用范围；
存储 record 和最终逐 record 浮点更新保持一致。
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

`codebook.short_records(records=1/2/4/8)` 选择显式短 record cohort。单条 record 保留
标量 metadata 和一条 payload 轴；多条 record 通过 Encoding-aware reshape 分离
record 与 payload 轴。Codebook 是保存的 stage；`cache_hint="streaming"` 表达源码
store 提示，目标访存 pass 仍会检查 alignment 和 alias 合法性。

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

`subblock_partials.window_dot` 在各编码 record 内闭合整数 dot。Q1_0 先保留四个
Q8_0 子 record 的 scale，再应用权重 scale；NVFP4 共享 staged codebook，分别闭合
各 16 元素缩放乘积；TQ2_0 闭合一个 256 元素有符号 dot。`rows_per_program` 表达
作者的行 cohort，native grid 覆盖全部 cohort。`window_projection` 在调用内将
activation 量化到调用者提供的 `Quantized` buffer，再复用同一宏，准备成本与
输出布局均保持显式。

Affine 短 record 将权重 scale/minimum 和 activation scale/sum 分别保留为双元素
metadata tile，以一个 `T.Parallel(2)` 表达转换和乘法，再分别更新 payload 乘积与
校正累加器。作者的 metadata 分组与整数 payload 分组一起通过共同前端保留。

`indexed_projection.packed_words` 在调用内生成 byte panel，跨 activation 行复用。
分组 factor、带偏置的乘积与精确 activation sum 校正保持独立的整数表达式。
`rows`、`columns` 表达这层复用，共同编译器选择 VNNI 或其他合法向量实现。
`ternary_planes` 与 `iq1_planes` 保留 group/entry/payload 轴，直到作者指定的最终归约。

`attention.rope` 公开 `rows_per_program`、`heads_per_program` 与 `pair_tile`，grid
覆盖这些显式 cohort。每个 pair Level 的系数 stage 跨 head 复用，每个 head 在写出前
先读取 activation 的两半。行、head 与 pair tile 的尾部继续受有效范围约束。

`short_projection.projection` 消费 `Q4_0Rows`、`Q4_1Rows`、`Q5_0Rows`、
`Q5_1Rows`、`Q8_0Rows`、`IQ4_NLRows` 或 `MXFP4Rows` 派生 Encoding。这些布局保留原 GGML record，并明确以四字节
payload chunk 跨行 cohort 交错。调用者在 invocation 前准备持久权重；projection 自身
量化 activation、计算所需整数 block sum，再跨全部 K record 保留输出 tile 的 state。
`rows`、`columns` 表达作者复用结构，`engine="vector"` 约束局部 dot。它与普通 Tensor
程序使用同一个编译核心。
查表格式显式读取一次 codebook，以带偏置的无符号 dot 和精确 activation sum 修正保留
有符号值。MXFP4 还消费传入的指数表；这些表均不内嵌到后端。
`projection.row_projection` 使用原 F32 activation 与 F16 权重，跨 K 保留 F32 lane 累加，
最后才进行归约。

产品仓库维护普通程序，Bench 管理 workload、外部参考、数值比较与计时范围。
公开性能结果为 **测试中 / Under evaluation**。

短 record 量化器支持 `records=1/2/4/8`。一个 cohort 显式区分 record 与 payload
轴，每条 record 的极值和编码 scale 只计算一次，其量化 lanes 共享 inverse。
公开 Tensor shape 和序列化 Encoding 保持不变。整数 field 反量化复用同一有限
cohort；`rows=1` 直接遍历连续存储，更大的 row group 保留作者声明的跨行复用。

`decoding.ternary_digits` 是行 dot、反量化和调用内面板共用的 radix-3 正数码宏。
字节位宽回绕与五个常量幂直接写在 DSL 中。`panel_projection` 在普通准备与
contraction 程序中消费这些码，面板准备仍计入调用耗时。

`codebook.tq1_0(chunk=16/32/64/128/256)` 限定每次解码的输出窗口；
`cache_hint="streaming"` 复用已有 store 提示，表达本 kernel 不再消费该输出。
编译器核对 alignment 和存储合同，只有合法时生成 streaming packet；其它情况保留
普通 store。解码数值和准备工作的边界保持不变。

`panel_projection.projection` 的 `unroll_decode` 由作者选择是否展开有限的列解码循环。
`pairwise=True, engine="vector"` 使用调用内的 `(pair, column, term)` I16 面板，
并显式表达两项乘积的 I32 partial。Activation 量化、面板准备和 contraction 都保留
在调用内；两种选择均经过普通前端与共享 core。

`panel_projection.staged_affine` 在显式 activation 行 cohort 中复用 encoded
Q4_K/Q5_K 权重。`staged_columns`、`block_k`、`batch_rows`、`rows` 和 `columns`
规定作者的复用范围。Q8_K activation 准备仍在调用内；完整 K 的输出 tile 从零累加，
部分 K tile 显式读取和更新输出。I32 scale 更新、分组乘积与配对求和修正均在源码中表达。

`row_partials.k_projection` 在调用内完成 Q8_K 量化后，复用 Q2/Q3/Q4/Q5/Q6
分组 dot 的主体。indexed projection 同样复用对应行 dot 的整数 partial。
源码参数选择列 cohort 和 record 窗口，目标 passes 选择局部机器指令。

`projection.staged_f16_projection` 在调用内把 activation 转成调用方的 F16 workspace，
分别 stage 一组权重列和 activation 行，再用普通完整 K 的 `T.gemm` 计算各输出 tile。
源码参数分别表达复用作用域和输出 tile 大小。

`quantized.field_windows` 通过显式 Level 子块解码整数格式 record。scale-group 窗口
从同一读取快照选择 payload 与标量 metadata，复用常规解码宏。这种作者结构直接表达
Encoding 的字段关系，不另建解码后的完整 record 数组，也不改变调用方存储布局。
Q2_K/Q3_K/Q6_K 的窗口程序先转换标量 group scale，再乘 payload。作者在源码中
明确这项数值组织，使一次标量 scale 转换可供整个窗口使用。
其中 Q6_K 将重建 code 保留为 I8，精确范围为 [-32, 31]，再转换为 F32。
共同解码宏保留显式的 `code_dtype` 源码参数，普通整 record 程序仍使用 I32。

`indexed_lanes.vec_dot` 与 `indexed_lanes.projection` 复用普通 `indexed_rows`
宏，覆盖 IQ1_S/IQ1_M/IQ2_XS/IQ2_XXS/IQ2_S。作者保留 16 lane 的 I16 pair partial、
I32 group 累加、显式 codebook 输入和逐 record 浮点 scale。`groups_per_step`
约束源码展开范围，不指定硬件向量宽度。projection 在调用内只准备一次调用方的 Q8_K
缓冲；`extreme_policy` 保留选定的量化并列极值规则，然后由相同的 indexed row 程序消费。

`indexed_lanes.scaled_entries` 在 64 元素 Level 窗口中明确表达 IQ2_S/IQ3_S 的
entry、payload 轴，先完成整数 scale group，再应用 record 的浮点缩放。IQ4_XS
使用 staged 的 16 项码本与 32 元素子块，显式更新逐组缩放结果。字段投影直接消费
已有 Level point，让共享编译器同时看见 record 读取与码本快照的作用域。
