# 普通作者程序

[English](kernel-library.md) · [工具书](index.zh-CN.md)

程序库位于 `tilelang.weft.kernels`。[catalog](https://github.com/Tianchen-ai/tilelang-weft/blob/main/tilelang/weft/kernels/catalog.py)
保存实际源码函数和有限的作者参数。选择条目调用普通 TileLang JIT factory，目标 pass
不按这些名字分派实现。

| 类别 | 源码模块 | 作者表达的结构 |
|---|---|---|
| Pointwise、SiLU、SwiGLU、softplus、exp/log、选择 | `activation` | 独立 Parallel tile、显式尾部及稳定 sigmoid/softplus macro |
| RMSNorm、LayerNorm、residual RMSNorm | `normalization` | 行快照、归约和显式 affine epilogue |
| Softmax、split/interleaved RoPE | `attention` | 稳定行 softmax 和明确 rotary 坐标 |
| Online attention | `attention.online_attention` | query stage、m/l/o 状态、key Level 和尾部 score mask |
| KV 更新、causal GQA | `attention` | 调用者 cache、已填充前缀及 head/group 映射 |
| GEMM、批量投影 | `linear` | 作者分块、转置、bias/SiLU 后处理及局部 engine 约束 |
| Q4_K/Q6_K × Q8_K | `quantized` | 共同 Encoding 字段、有限位宽 partial、scale 与校正支路 |
| 整数字段解码 | `quantized.dequantize` | Q1、Q4/Q5/Q8、K 格式和 TQ2 的显式数值解码 |
| Lookup、sub-byte 格式 | `codebook` | IQ1/IQ2/IQ3/IQ4、MXFP4/NVFP4、TQ1 与显式表输入 |
| Activation preparation | `preparation` | 调用者 Q8_0/Q8_1/Q8_K 输出字段、舍入和分组和 |
| Packed 投影 | `composition.packed_projection` | 显式解码和 dense 投影两个程序，workspace 归调用者；可分别 prepare/compute |
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

产品仓库维护普通程序，Bench 管理 workload、外部参考、数值比较与计时范围。
公开性能结果为 **测试中 / Under evaluation**。
