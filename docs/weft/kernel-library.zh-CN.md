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
| Packed 子块投影 | `subblock_projection` | 跨 activation 行保留 Q1/NVFP4/TQ2/IQ4_XS 权重 cohort，显式子块缩放和调用内量化 |
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

`panel_projection.projection(format="Q2_K", panel_dtype="int8")`
由作者在调用内面板中，将四位整数 scale 乘入两位 code。乘积范围为 `[0, 45]`，
I8 精确保留整个整数值，并能作为有符号矩阵操作数。`engine` 约束该面板 contraction，
I16 activation block-sum 校正明确采用向量路线。activation 量化、面板准备和校正均计入调用。

Q1/NVFP4 的 `subblock_projection.projection(organization="records")`
每次解码一个有界权重窗口，在推进 K 前供串行 activation 行共享。decode 保留输出
累加器，prefill 更新显式输出存储；默认 `"staged"` 则在编码复用作用域内，跨 K
保留输出 tile。两者是作者选择的不同遍历与驻留方式，均计入相同的 activation
准备与数值工作。
第一代布局的 `matrix.ime` 暴露 `dot_width=8/16/32`。32 元素分组直接完成一次
record contraction，更小的分组保留显式子 Level 和跨贡献的整数 state。
`engine` 继续约束具体 contraction，同一格式可以用适合 SIMD 或 IME 的作者分组。

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

`panel_dense.projection(unroll_k=1/2/4/8)` 公开有界的源码 outer-product 循环展开。
Panel 准备仍在调用内，`rows`、`columns` 选择输出 cohort；展开后每个累加器仍按相同
K 顺序接收乘积。

对于 GGML 正常准备、code 范围为 `[-127, 127]` 的 Q8_0 activation，
`subblock_partials.subblocks(sign_transfer=True)` 及完整 `projection` 使用 GGML x86
同样的字节符号转移：把 Q1 符号转到 I8 activation，以 U1 的一相乘，再将四项累加到
I32。I8 取负保留源码声明的回绕语义；默认写法表达中心化权重乘法。两种组织共用
record scale、逐行宏与准备协议。

`linear.blocked_matmul` 在显式 `tile_m`/`tile_n` Level 和 `rows`/`columns` 输出 cohort
内完成整个 K 的缩并，静态 K 使用普通 `T.gemm` 并保留权重的存储顺序。
`dynamic_k=True` 将具名运行时 K 经 ABI View 和完整局部 Value 传给
`T.reduce_dot(..., over="k")`；输出 tile 保持有界，由共同目标编译器遍历 K，
同一编译 kernel 可以接受不同的运行时归约长度。`linear.matmul` 则单独表达分块 K
循环及各输出程序的 grid。

`attention.rope(head_major=True)` 按 head 遍历连续的 pair 范围；默认先遍历 pair tile，
再跨 head cohort 复用 staged 系数。`attention.rope_runtime(pair_tile=32)` 保留动态的
token、head 和 pair 长度，在一个 CPU program 中遍历。两种形式都在可原地写入的输出
store 之前保留两个输入快照。
`causal_attention` 以 typed View 保存 query stage，以 state 保留 online numerator。

token/head cohort 参数限定每个 grid 实例的串行工作，不改变有效 causal KV 前缀。

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
IQ2 的 codebook 与 sign 行均为 I8 元素。`iq2_windows(product_dtype=...)` 显式选择
I16 或 I32 乘积，再转换为 F32；两者都能精确容纳完整的 I8×I8 乘积范围。
I16 公开 widening multiply 与 widening 整数到浮点转换，I32 则可让短 payload
与 F32 消费者保持同一 RVV 配置。这项作者数值选择与后端的物理 LMUL 选择分开。
`codebook.indexed_windows` 将同一表行窗口结构用于 IQ1_M、IQ2_S 与 IQ3_XXS。
IQ1_M 的 half-bit 查表与 IQ2_S 的 packed signs 仍是显式源码输入及字段。
`iq3_windows` 在八元素符号窗口内配对两个四元素表行，并在父层 32 元素 Level
共享标量 scale。符号运算直接消费原 packed-bit 字段，公开输出仍为 `(M, K)`。

`indexed_lanes.xxs_subgroups` 用 64 元素的 Level 窗口表达 IQ3_XXS。
每个窗口保留两个整数 scale group，以及各自的 codebook 行和 sign 行，随后加入 block 和。
显式的 pair-entry、pair、payload 轴保留一行八个 sign 元素与两行四元素 codebook 的对应关系。
有符号 payload 与 activation 通过普通 `T.reduce_dot` 计算，直接暴露局部复用，
不先建立完整的解码 record 数组。
`quantized.field_windows(chunk=...)` 可将 scale group 细分成 8/16/32 元素的
作者窗口；完整短 record 直接复用已有 record Level。
`code_dtype="int8"/"int16"/"int32"` 显式选择重建后的整数 code 位宽，再执行
原有浮点转换。默认仍为 I8，较宽类型可以让短 payload 计算与 F32 消费者保持一致的
机器位宽。
Q1、TQ2 也支持完整 128/256 元素 record 窗口，有符号码在转换为浮点前保持 I8。
仿射分组窗口先计算标量 scale 与 minimum 系数，再在整个 payload 窗口中复用。
`packed.projection` 使用显式整数 block partial 和 F32 scale 更新。
Q4_K、Q6_K 通过共享 `partials` 宏保留整数 partial 和独立的
scale/correction 支路。`composition.packed_projection` 的显式解码到 workspace
再做 dense 乘法程序另有 `acc_dtype` 参数。普通 `linear.matmul` 分别声明输入、局部操作数、累加和
输出类型，原有默认仍为 F32；目标 passes 保留这些数值选择。

`row_partials` 提供 x86 的作者级延迟归约与分组整数计算。单行 Q6_K 的
`scale_groups` 限定作者同时暴露的 scale group 数量，不选择 ISA。
IQ3_XXS 逐行程序显式选择 payload、metadata 和标量 scale 字段，并将 `[1, 31]`
的组 scale 保留为 I16，直到 I32 乘积和校正支路使用时再 widen；dot 与完整 decode
投影共用这一宏。
`q6_planes` 也接受多行输出。显式串行行循环为每行保留相同的 plane/lane partial
及逐 record 缩放，不把所有输出行同时暴露为驻留值。
共同的 `q6_plane_value` 宏通过 `accumulation="block"` 在每个 record 的 F32 更新前
闭合精确 I32 总和，或通过 `"lanes"` 将 F32 lane partial 保留到行末。
`scaled_projection(organization="block_planes")` 在调用内准备 activation 后，
以非内联逐行调用复用同一 block-scaled 宏；`lanes`、`planes` 仍为显式作者维度。

RISC-V 的 `row_partials.affine_groups` 以 Level 子块表达 Q2_K/Q4_K/Q5_K
的 scale group，并在独立整数支路中计算 minimum 修正。
Q4_K/Q5_K 也可指定 `organization="record"`，显式形成同一 record 中八个独立的
32 元素整数 dot，应用整数 scale 并归约后，再执行原有浮点更新。payload 保留有限
无符号码类型，minimum 修正先 widen 再相加 activation 分组和。
`scale_projection` 在调用内准备 activation 后复用同一 record 宏。
`scaled_groups` 为 Q3_K 和 Q6_K 使用 16 项子块。每组从原 record 快照读取 payload 与标量 metadata，
完成 I32 partial 和整数 scale 更新，再执行 record 的浮点累加。
仿射 `scale_projection` 可通过 `organization="groups"` 为 Q2_K 复用逐组闭合的
整数 dot，Q8_K 准备仍在调用内完成；默认 `"lanes"` 保留另一种有界 I16 lane 累加。
`scaled_planes` 则在每个半 record 的 128 元素子 Level 中保留 `(4, 2, 16)` payload 组织，显式计算
plane/group partial 及其整数 scale。两个 factory 表达不同的源码复用范围；
存储 record 和最终逐 record 浮点更新保持一致。
`q2_lanes` 和 `scaled_lanes` 在共同数值 helper 中完成单行。多行遍历和完整 projection
调用相同 helper，每行的数值 lanes 在下一行开始前收口。
Q2 在每个 I16 lane 中累加四项精确缩放乘积，再 widen；Q3/Q6 保留 I32
group-scaled lanes 和逐 record 的 F32 更新。这是作者的数值结构。Q2 的
`scale_projection` 与 `scaled_projection` 仍将 activation 量化计入原调用。
Q2 projection 通过 `T.macro(noinline=True)` 在一次调用中闭合每个输出行的完整数值
lanes；Q3/Q6 则在已有的不内联 helper 中闭合一个 activation 行的计算批次。
`staged_projection` 也可以在已声明的权重列 stage 和 activation 行 stage 中保留相同的
lanes 组织。Q2 使用 16 个精确 I16 lanes；Q3/Q6 选择 16 或 32 个缩放 I32 lanes。
默认的分组 plane 组织继续保留。短 record affine projection 将 minimum 和 product
支路直接更新到 F32 state，显式形成两处局部 FMA 机会，减少合并 contribution 的临时值。
多输出 affine tile 为各输出保留跨 record 内 16/32 元素分组的 I32 总和；保存的
coefficient 字段从与 payload 相同的权重 record 快照投影，保留 Encoding 分组关系，
不再单独读取或建立 coefficient stage。minimum 修正仍与 Q8_K 分组和执行独立 dot。
`short_projection.staged_projection` 在准备完成后，通过一次不内联的 macro 调用执行整个
staged 计算遍历，保留原 Level 树和自然输入 ABI。调用边界结束准备阶段临时量的驻留，
不为每个输出 tile 或 K record 增加函数调用。
`subblock_projection.projection` 为 Q1/NVFP4 提供相同的显式 `batch_rows`、
`staged_columns`、`rows` 和 `columns` 复用作用域。输出 tile 的 state 跨全部 K record
保留，只在完成后写出。NVFP4 保留独立的 16 元素码本 dot 与子块 scale，Q1 保留
中心化 I8 code；activation 量化和局部 encoded stage 都在本次调用内完成。
TQ2_0 变体为每个权重 record 闭合一个 256 项无符号码 dot，然后在 I32 中减去
Q8_K activation 的分组和。这是精确整数恒等式
`sum((q - 1) * x) = sum(q * x) - sum(x)`；已有分组和跨输出列复用，不再构造
中心化字节 tile。输出 state 和编码作用域保持一致，Q8_K 准备与 workspace
仍显式保留在 projection 调用内。
IQ4_XS 在相同 stage 中保留 32 元素码本 dot、显式低/高 scale 字段与有符号 I32
group factor；Q8_K 准备也计入本次调用。
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

Q4/Q5/Q8、IQ4_NL 和 MXFP4 短 record 的 `block32` 与完整
`subblock_partials.projection` 共用 `block32_rows` 宏。Projection 在调用内完成
activation 量化，再执行同一行程序。两个 factory 中的 `records` 与 `partials`
保留相同的源码分组；`extreme_policy` 与 `inverse_policy` 保留准备阶段的数值合同。
IQ4_XS 的 dot 与 projection 同样共用 `iq4_group_rows`，在两个入口中保留显式
分组 partial 与 codebook 读取快照。

`subblock_partials.window_dot` 在各编码 record 内闭合整数 dot。Q1_0 先保留四个
Q8_0 子 record 的 scale，再应用权重 scale；NVFP4 共享 staged codebook，分别闭合
各 16 元素缩放乘积；TQ2_0 闭合一个 256 元素有符号 dot。`rows_per_program` 表达
作者的行 cohort，native grid 覆盖全部 cohort。`window_projection` 在调用内将
activation 量化到调用者提供的 `Quantized` buffer，再复用同一宏，准备成本与
输出布局均保持显式。
分组 `subblocks` 变体也保留原 I8 activation，以显式有符号权重码执行 widening
grouped MAC，完整保留 -128 的计算。
`rows_per_program` 也可以让一个实例保留多行完整计算。

`short_projection.staged_projection` 消费原始编码权重布局。作者分别指定外层权重列
stage、activation 行 stage，以及在整个 K 上保留 state 的较小行列输出 tile。
它复用 staged bytes，activation 量化仍计入调用，不要求持久 repack；相同 View、
Level 和 contraction 主链交由目标编译器选择物理向量布局。

`row_partials.scale_projection` 为 Q2_K/Q4_K/Q5_K 复用 `affine_group_rows`，
分别保留带 scale 的整数和及 minimum 校正和。Q2_K 以 I16 表示 minimum
系数，与 Q8_K 的分组和显式 contraction。scale 与 indexed projection
factory 均公开已有 activation extreme/inverse 策略，复用行程序时保持 workload
要求的量化合同。

`row_partials.staged_projection` 在有界的权重列和 activation 行 stage 中复用同一组
affine 或 signed 分组行宏，覆盖五种 K 格式，无需解码面板或持久重排。
`batch_rows` 与 `staged_columns` 限定作者的复用范围，Q8_K 准备仍保留在完整 projection 内。
显式指定 `engine="matrix"` 时，affine tile 先将 Q2/Q4/Q5 的 payload 转为有符号
I8，再与 Q8_K contraction。三种码值的范围分别为 `0..3`、`0..15`、`0..31`，
均可由 I8 精确表示。这项作者转换提供统一的 I8 矩阵操作数，保留 activation 位宽、
整数 scale 分组、源 Encoding 和存储中的权重字节。

Q3_K/Q6_K 还可通过显式 `rows`、`columns` 在 K 遍历中保留二维输出 tile。
解码后的 64 元素权重窗口服务 activation 行 cohort 的各次 16 元素 contraction；
整数 scale 更新先于 record 浮点更新收口。这些是源码复用维度，与后端 SIMD 宽度和
寄存器布局独立。

`codebook.metadata_windows` 将 16 元素 codebook 保存为 stage，并在各 32/16 元素
Level 窗口中以标量复用 IQ4_XS/NVFP4 scale metadata，再索引读取保存的表。
`ternary_planes` 以独立的 digit/payload 轴复用 TQ1 的 32-byte、16-byte 与四 byte payload，
再将各平面的解码结果 reshape 到连续输出窗口。幂表是显式 stage 输入，payload
读取保留在 record body 中。

共享 Q4_1/Q5_1 解码器用一个仿射表达式表示 scale 和 minimum 校正。
Q5 先在 U8 中构成完整五位码，再扩大位宽。这些源码表达暴露了合法的局部
FMA，并减少宽位域运算，保持解码范围和输出布局。

Affine 短 record 将权重 scale/minimum 和 activation scale/sum 分别保留为双元素
metadata tile，以一个 `T.Parallel(2)` 表达转换和乘法，再分别更新 payload 乘积与
校正累加器。作者的 metadata 分组与整数 payload 分组一起通过共同前端保留。
标量 record 程序则在每条 record 内合并乘积与校正，再更新浮点和，保留该作者程序的
累加顺序。Q4_1 与 Q8_0 无需数值解码时直接消费保存的 payload 字段，让 contraction
继续保留 Encoding 的位宽信息。

`indexed_projection.packed_words` 在调用内生成 byte panel，跨 activation 行复用。
分组 factor、带偏置的乘积与精确 activation sum 校正保持独立的整数表达式。
`rows`、`columns` 表达这层复用，共同编译器选择 VNNI 或其他合法向量实现。
`ternary_planes` 与 `iq1_planes` 保留 group/entry/payload 轴，直到作者指定的最终归约。
`iq1_planes(format="IQ1_S"/"IQ1_M", groups_per_step=...)` 独立控制 free group
cohort，通过同一个行 macro 收口。IQ1_M 保留 grid 与 signed-delta 两条整数 partial
以及显式 half-bit 查表；IQ1_S 保留 activation group sum 校正。

`attention.rope` 公开 `rows_per_program`、`heads_per_program` 与 `pair_tile`，grid
覆盖这些显式 cohort。每个 pair Level 的系数 stage 跨 head 复用，每个 head 在写出前
先读取 activation 的两半。行、head 与 pair tile 的尾部继续受有效范围约束。

`short_projection.projection` 消费 `Q4_0Rows`、`Q4_1Rows`、`Q5_0Rows`、
`Q5_1Rows`、`Q8_0Rows`、`IQ4_NLRows` 或 `MXFP4Rows` 派生 Encoding。这些布局保留原 GGML record，并明确以四字节
payload chunk 跨行 cohort 交错。调用者在 invocation 前准备持久权重；projection 自身
量化 activation、计算所需整数 block sum，再跨全部 K record 保留输出 tile 的 state。
`rows`、`columns` 表达作者复用结构，`engine="vector"` 约束局部 dot。它与普通 Tensor
程序使用同一个编译核心。
`indexed_projection.projection` 与行程序一样公开 `extreme_policy` 和
`inverse_policy`。调用内整数 panel 保留 workload 的 activation 量化合同；panel
构造和 workspace 的读写继续计入完整调用。
查表格式显式读取一次 codebook，以带偏置的无符号 dot 和精确 activation sum 修正保留
有符号值。MXFP4 还消费传入的指数表；这些表均不内嵌到后端。
`projection.row_projection` 使用原 F32 activation 与 F16 权重，跨 K 保留 F32 lane 累加，
最后才进行归约。

产品仓库维护普通程序，Bench 管理 workload、外部参考、数值比较与计时范围。
[结果工具书](https://tianchen-ai.github.io/tilelang-weft-bench/results.html)
公布原生耗时、原始样本与源码绑定。

`row_partials.q6_planes` 的 `rows_per_program` 可将多行计算显式分成独立 CPU
program，每个 program 保留相同的 plane/lane 数值组织，完整 native grid 覆盖全部行。
默认仍由一个 program 遍历所有行。
`subblock_partials.block32` 以及 indexed `vec_dot`、`scaled_entries`、
`xxs_subgroups` 提供相同的行 cohort 选择。显式 grid 改变 program 的遍历边界，
每个 factory 仍复用原行宏，保留输入准备和数值分组。
`block32(row_calls=True)` 和 `iq1_scaled(row_calls=True)` 则把每行收口到一个
原生非内联 helper。program 的普通串行行循环经 helper 调用相同的数值宏，
保留输入布局与 record partial。这项作者选择限定每行的活跃值范围，不增加
Python 调用，也不改变完整 grid 的计时范围。

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

`row_partials.scaled_projection(organization="planes", planes=2/4)`
由作者选择每批整数 dot 的 plane 数量。两种写法保留 128 元素 Level、精确的
scale-group 整数和，以及逐 record 的浮点更新；双 plane 批次先完成当前 partial，
再计算下一批，缩短同时存活的向量范围。

`row_partials.staged_projection` 用 `engine` 约束 tile contraction，
用 `decode_window` 指定作者的 Q3_K/Q6_K 有界解码范围。
行列分块、staged panel、整数 scale 分组和 activation 准备仍在源码中明确表达。
8×8 分块配合 16 元素解码窗口，可以通过同一 factory 表达 IME2 矩阵程序。

`panel_projection.projection` 的 `unroll_decode` 由作者选择是否展开有限的列解码循环。
`pairwise=True, engine="vector"` 使用调用内的 `(pair, column, term)` I16 面板，
并显式表达两项乘积的 I32 partial。Activation 量化、面板准备和 contraction 都保留
在调用内；两种选择均经过普通前端与共享 core。

`panel_projection.record_projection` 为 Q2_K、Q6_K 在两个有序 record 之间保留
一个输出 tile。显式调用内 workspace 保存两份整数面板及其 scale/minimum metadata。
填充完成后，各面板的独立读取快照位于 activation 行 Level 之前，使矩阵 packing
能够被各行消费者复用。tile 的 F32 state 按 record 原顺序更新，每两个 record
才写回一次。量化、解码与面板构造均保留在完整调用内。编译器根据这些普通 Value
及其消费者选择物理 packing、AMX 操作和资源放置。

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
宏，覆盖 IQ1_S/IQ1_M/IQ2_XS/IQ2_XXS/IQ2_S。作者保留 16 lane 的整数 partial、
I32 group 累加、显式 codebook 输入和逐 record 浮点 scale。`groups_per_step`
约束源码展开范围，不指定硬件向量宽度；有 16 组的格式还可使用 `groups_per_step=16`。
projection 在调用内只准备一次调用方的 Q8_K
缓冲；`extreme_policy` 保留选定的量化并列极值规则，然后由相同的 indexed row 程序消费。

`indexed_lanes.staged_projection` 为 indexed record 增加外层权重列与 activation 行
stage，复用已有 IQ1/IQ2/IQ3/IQ4 行宏，保留分组 partial、显式 codebook/sign 输入及
逐 record 缩放。`batch_rows`、`staged_columns` 与 `groups_per_step` 表达源码复用和
分组；Q8_K 量化仍在调用内。stage 保存自然编码布局，无需调用外的持久权重准备。

`indexed_lanes.scaled_entries` 在 64 元素 Level 窗口中明确表达 IQ2_S/IQ3_S 的
entry、payload 轴，先完成整数 scale group，再应用 record 的浮点缩放。
lookup 地址及 packed metadata 运算可通过 `index_dtype="uint16"` 或 `"uint32"` 指定宽度；
编码存储、sign、I32 partial 与 F32 record 更新保持声明的类型。IQ2_S 在窗口遍历前读取
record 的 F16 scale，在同一读取快照中保留 Encoding 的自然对齐。IQ4_XS
使用 staged 的 16 项码本与 32 元素子块，显式更新逐组缩放结果。字段投影直接消费
已有 Level point，让共享编译器同时看见 record 读取与码本快照的作用域。

有界的 `scaled_entries_rows` 与 `xxs_subgroups_rows` 宏也同时服务各自的 dot
factory 和 `indexed_lanes.entry_projection`。相同的 entry/payload Level、码本快照
与整数 scale group 在 projection 中保留，activation 量化仍位于调用内部。
`row_partials.scaled_projection` 为 Q3_K/Q6_K 复用 `scaled_planes_rows` 或
`scaled_lanes_rows`；`organization` 在作者程序中选择逐 block 的整数和或延迟归约的
lane partial，与目标向量宽度独立。
`organization="planes"` 以及直接调用的 `row_partials.scaled_planes` 通过 `planes=2/4`
选择每批整数 dot 的平面数；128 元素 Level、精确整数分组及每个 record 的一次浮点更新不变。
两平面批次在下一批开始前完成 partial，缩短同时存活的向量范围。
`xxs_subgroups` 的 `index_dtype="uint16"/"uint32"` 指定作者的字节偏移算术，
与目标 gather 指令的索引选择分开。

IQ1_M 的 indexed row 程序通过普通 `T.reinterpret` 重建 F16 scale，再以
`T.reduce_dot` 完成各组有符号 numerator dot。Scale、delta、整数累加及最终
八分之一系数仍是源码中显式的数值操作。

`indexed_lanes.iq1_scaled` 将 IQ1 codebook 的 `{-1,0,1}` 字母表表达为八分之一
单位的整数 numerator。有符号 I16 乘积先包含奇数 group scale，再扩宽为 I32，
最后完成一次 record 求和。`projection` 和 `staged_projection` 通过
`numerical_form="scaled"` 复用同一 `iq1_scaled_rows` 宏；Q8_K 准备仍计入调用。
显式带偏置的 unsigned two-bit 表示将 codebook 范围边界传给共同整数分析，而非任意 I8。
偏置先使用字节算术，字节回绕不改变后续 two-bit 转换的低位。奇数 group factor
保持 I8，直到 I16 乘积处再显式扩宽。

分阶段 IQ1 程序还接受 `rows`、`columns` 输出分块。`groups_per_step` 限定每个 record
子 Level 内同时计算的 group 数，精确 I32 小计跨窗口携带，之后仍在原 record 边界执行
F32 scale 更新。权重和 activation stage 保留声明的外层复用范围；局部 contraction
保留共享 group 与彼此独立的行、列轴。

单行、单列输出 issue 可在分阶段 IQ1 factory 中选用 `numerical_form="planes"`，
复用 `iq1_plane_rows` 的 entry/payload 整数 dot，并将 delta 修正保持为独立支路。
分阶段 factory 使用 `T.reinterpret` 重建 F16 block scale；独立 factory 继续保留
显式 scale-table 输入合同。
