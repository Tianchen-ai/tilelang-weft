# 原生应用与 llama.cpp

[English](native-deployment.md) · [工具书](index.zh-CN.md)

TileLang 函数是源程序。共同编译器生成原生库、typed C header、Encoding/目标
metadata 和诊断 IR；应用 worker 通过 `libweft_runtime` 直接调用原生库。

## 导出数值 kernel

[示例 06](https://github.com/Tianchen-ai/tilelang-weft/blob/main/examples/weft/06_autotune_export.py)
通过 TileLang 调优，再导出选定的 artifact：

```sh
python examples/weft/06_autotune_export.py --output /tmp/affine-artifact
SDK=$(python -c 'import weft_core; print(weft_core.sdk_path())')
c++ -std=c++17 examples/weft/native/affine.cpp -I/tmp/affine-artifact \
  -I"$SDK/include" -L"$SDK/lib" -Wl,-rpath,"$SDK/lib" \
  -lweft_runtime -o /tmp/affine-app
/tmp/affine-app /tmp/affine-artifact
```

C++ 消费者读取 manifest、检查执行能力，并在使用 typed entry 时持有原生库。
应用只链接 runtime 和系统库。

## 绑定明确的 GGML 角色

[示例 07](https://github.com/Tianchen-ai/tilelang-weft/blob/main/examples/weft/07_llama_package.py)
编译普通 `deployment.vec_dot` 程序，实现 Q4_K/Q6_K × Q8_K。
每个导出入口接收 packed 权重行、activation 行、一个 F32 输出和运行时 `K`。
Q4_K 保留作者的 i16 grouped partial、显式 i32 扩展和独立 scale/min 分支。
package 校验器检查 ABI 与布局，再绑定 `ggml.vec_dot/q4_K`、`q6_K`。

```sh
python examples/weft/07_llama_package.py --output /tmp/ggml-package
```

使用已有的 GGML CPU 构建运行这两个角色：

```sh
LLAMA=/path/to/llama.cpp
GGML_LIB=/path/to/llama-build/bin
SDK=$(python -c 'import weft_core; print(weft_core.sdk_path())')
c++ -std=c++17 examples/weft/native/ggml_rows.cpp \
  "$SDK/share/llama.cpp/adapter.cpp" \
  -I"$SDK/include" -I"$SDK/share/llama.cpp/include" \
  -I"$LLAMA/ggml/include" -I"$LLAMA/ggml/src" -I"$LLAMA/ggml/src/ggml-cpu" \
  -L"$SDK/lib" -L"$GGML_LIB" -Wl,-rpath,"$SDK/lib" -Wl,-rpath,"$GGML_LIB" \
  -lweft_runtime -lggml-cpu -lggml-base -pthread -o /tmp/ggml-rows
WEFT_GGML_PACKAGE=/tmp/ggml-package /tmp/ggml-rows
```

该应用使用 GGML 生成量化输入，进入经过能力检查的 worker node，调用选定原生入口，
与 GGML 结果比较并检查替换计数。这一进程直接执行原生代码。

## 构建应用后端

给示例 07 增加 `--llama-source "$LLAMA"`，将已审阅的调用位置 hook 安装到外部
checkout。CMake 增加 `-DGGML_WEFT=ON -DCMAKE_PREFIX_PATH="$SDK"`；
`WeftLlama.cmake` 将 adapter 编入 CPU 后端并链接共同 runtime。
启动应用时通过 `WEFT_GGML_PACKAGE` 选择上面的 package。

activation 量化、权重布局和 worker 划分继续由 GGML 管理。kernel 在现有 worker
中调用单实例入口。Bench 的 `prepare-model`、`validate-model`、`e2e`
负责模型输入核验、命中记录，以及分开的 prefill/decode 测量。
