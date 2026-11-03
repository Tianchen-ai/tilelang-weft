# Native applications and llama.cpp

[中文](native-deployment.zh-CN.md) · [Handbook](index.md)

A TileLang function is the source program. The shared compiler emits a native
library, typed C header, Encoding/target metadata and diagnostic IR. Application
workers call that library directly through `libweft_runtime`.

## Export a numerical kernel

[Example 06](https://github.com/Tianchen-ai/tilelang-weft/blob/main/examples/weft/06_autotune_export.py)
uses TileLang autotuning, then exports the selected artifact:

```sh
python examples/weft/06_autotune_export.py --output /tmp/affine-artifact
SDK=$(python -c 'import weft_core; print(weft_core.sdk_path())')
c++ -std=c++17 examples/weft/native/affine.cpp -I/tmp/affine-artifact \
  -I"$SDK/include" -L"$SDK/lib" -Wl,-rpath,"$SDK/lib" \
  -lweft_runtime -o /tmp/affine-app
/tmp/affine-app /tmp/affine-artifact
```

The C++ consumer opens the manifest, validates its execution requirements and
retains the library while its typed entry is in use. Compiler, Python and TVM
are absent from the application's link dependencies.

## Bind exact GGML roles

[Example 07](https://github.com/Tianchen-ai/tilelang-weft/blob/main/examples/weft/07_llama_package.py)
compiles the ordinary `deployment.vec_dot` programs for Q4_K/Q6_K × Q8_K.
Each exported entry takes packed weight/activation rows, one F32 output and
runtime `K`. The author program retains Q4_K's i16 grouped partials, explicit
i32 widening and separate scale/min branches. The package validator checks
these ABI/layout requirements before binding `ggml.vec_dot/q4_K` and `q6_K`.

```sh
python examples/weft/07_llama_package.py --output /tmp/ggml-package
```

To exercise those roles against an existing GGML CPU build:

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

This application creates GGML-quantized inputs, enters a checked worker node,
calls the selected native entry, compares the result with GGML and checks its
replacement counter. It does not include Python in that process.

## Build an application backend

Pass `--llama-source "$LLAMA"` to example 07 to install the reviewed call-site
hooks in that external checkout. Configure its CMake project with
`-DGGML_WEFT=ON -DCMAKE_PREFIX_PATH="$SDK"`; the included `WeftLlama.cmake`
compiles the adapter and links the common runtime. Launch the built application
with `WEFT_GGML_PACKAGE` selecting the package above.

GGML continues to own activation quantization, the weight representation and
worker partition. Kernels run through the single-instance entry inside that
worker. Bench's `prepare-model`, `validate-model` and `e2e` commands own model
input checking, observed hits and separate prefill/decode measurements.
