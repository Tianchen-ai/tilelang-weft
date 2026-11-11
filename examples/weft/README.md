# Weft CPU examples

[中文](README.zh-CN.md) · [Handbook](../../docs/weft/index.md)

Run with installed TileLang–Weft and matching `weft-core` wheels.
The Python examples compile native TileLang programs and check CPU results.
Example 07 exports a checked GGML role package; its C++ consumer checks the
native results and replacement counts against GGML.

| File | What it shows |
|---|---|
| `01_pointwise.py` | Tensor, Parallel, masked tail and native grid |
| `02_tiles.py` | Local copy, reduction and standard GEMM |
| `03_dynamic.py` | Runtime shape/stride, multiple outputs and ordinary control |
| `04_encoding_levels.py` | Packed fields, nested Level, snapshots and typed partials |
| `05_derived_layout.py` | Explicit persistent preparation and derived interleave consumption |
| `06_autotune_export.py` and `native/affine.cpp` | CPU tuning, native export and a standalone C++ consumer |
| `07_llama_package.py` and `native/ggml_rows.cpp` | Dynamic packed-row ABI, GGML role binding, reference checks and observed replacement calls |
| `13_packed_dot.py` | Q4_0/Q8_0 Encoding records passed to the ordinary packed-dot JIT factory |

```sh
python examples/weft/01_pointwise.py
python examples/weft/04_encoding_levels.py
python examples/weft/13_packed_dot.py
python examples/weft/06_autotune_export.py --output /tmp/tilelang-affine
```

Build the C++ consumer with the matching installed core SDK:

```sh
SDK=$(python -c 'import weft_core; print(weft_core.sdk_path())')
c++ -std=c++17 examples/weft/native/affine.cpp -I/tmp/tilelang-affine \
  -I"$SDK/include" -L"$SDK/lib" -Wl,-rpath,"$SDK/lib" -lweft_runtime -o /tmp/affine-app
/tmp/affine-app /tmp/tilelang-affine
```

The application borrows the checked typed entry and retains its native library.
It links the native runtime and system C++ libraries.

Normal reusable author programs live in `tilelang.weft.kernels`.
Comparison and timing belong to the independent Bench project.
Complete GGML build/run commands are in the [deployment handbook](../../docs/weft/native-deployment.md).
