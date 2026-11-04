# Weft CPU 示例

[English](README.md) · [工具书](../../docs/weft/index.zh-CN.md)

安装 TileLang–Weft 与匹配 `weft-core` wheel 后运行。
每个示例编译原生 TileLang 源码，并检查 CPU 结果。

| 文件 | 展示内容 |
|---|---|
| `01_pointwise.py` | Tensor、Parallel、尾部 mask 和原生 grid |
| `02_tiles.py` | 局部 copy、归约与标准 GEMM |
| `03_dynamic.py` | 动态 shape/stride、多输出及普通控制 |
| `04_encoding_levels.py` | packed 字段、嵌套 Level、快照和有限位宽 partial |
| `05_derived_layout.py` | 显式持久 preparation 与 derived interleave 消费者 |
| `06_autotune_export.py`、`native/affine.cpp` | CPU 调优、原生导出与独立 C++ 调用 |
| `07_llama_package.py`、`native/ggml_rows.cpp` | 动态 packed-row ABI、GGML 角色绑定、参考检查与实际替换计数 |

```sh
python examples/weft/01_pointwise.py
python examples/weft/04_encoding_levels.py
python examples/weft/06_autotune_export.py --output /tmp/tilelang-affine
```

使用安装后的 core SDK 构建 C++ 消费者：

```sh
SDK=$(python -c 'import weft_core; print(weft_core.sdk_path())')
c++ -std=c++17 examples/weft/native/affine.cpp -I/tmp/tilelang-affine \
  -I"$SDK/include" -L"$SDK/lib" -Wl,-rpath,"$SDK/lib" -lweft_runtime -o /tmp/affine-app
/tmp/affine-app /tmp/tilelang-affine
```

应用校验 worker 能力后借用 typed 原生入口，并持有对应库的生命周期；链接依赖为
native runtime 与系统 C++ 库。

普通可复用作者程序位于 `tilelang.weft.kernels`；参考比较与计时放在独立 Bench 项目。
`13_packed_dot.py` 构造 Q4_0/Q8_0 records，调用 TileLang–Weft Bench 选择的同一个普通 packed factory。
