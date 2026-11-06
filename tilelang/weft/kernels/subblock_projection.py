"""Invocation-local subblock decoding shared across activation rows."""

import tilelang
from tilelang.weft import language as T
from . import ggml_formats as F
from .preparation import quantize_rows


@tilelang.jit(target="weft")
def projection(M, N, K, format="NVFP4", columns=8,
               extreme_policy="first", inverse_policy="direct"):
    assert format in ("Q1_0", "NVFP4")
    encoding = getattr(F, format)
    elements = encoding.schema.elements
    width = 32 if format == "Q1_0" else 16
    assert M > 0 and N > 0 and columns > 0 and N % columns == 0
    assert K > 0 and K % elements == 0
    weight_bytes = K // elements * (encoding.schema.storage_bits // 8)

    @T.prim_func
    def main(
        Weights: T.Tensor((N, weight_bytes), "uint8"),
        Activation: T.Tensor((M, K), "float32"),
        Quantized: T.Tensor((M, K // 32 * 34), "uint8"),
        Output: T.Tensor((M, N), "float32"),
        Codebook: T.Tensor((16 if format == "NVFP4" else 0,), "int8"),
        Scale: T.Tensor((256 if format == "NVFP4" else 0,), "float32"),
    ):
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Activation": "input",
            "Codebook": "input", "Scale": "input", "Quantized": "workspace", "Output": "output"}})
        weights = T.view(Weights, shape=(N, K), encoding=encoding, axes=("n", "k"), abi=True)
        source = T.view(Activation, shape=(M, K), axes=("m", "k"), abi=True)
        prepared = T.view(Quantized, shape=(M, K), encoding=F.Q8_0, axes=("m", "k"), abi=True)
        output = T.view(Output, shape=(M, N), axes=("m", "n"), abi=True)
        if format == "NVFP4":
            codebook = T.view(Codebook, shape=(16,), axes=("code",), abi=True)
            scales = T.view(Scale, shape=(256,), axes=("scale",), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            quantize_rows(source, prepared, M, K, "Q8_0", 32,
                          extreme_policy, inverse_policy, "dynamic", 32)
            if format == "NVFP4":
                table = T.stage(T.load(codebook))
            with T.level.cols(N, group=columns) as col:
                if M == 1:
                    accumulated = T.state((columns,), "float32", init=0)
                else:
                    for m in T.serial(M):
                        zero = T.state((columns,), "float32", init=0)
                        T.copy(zero, output[m, col])
                with T.level.blocks(K, extent=elements) as block:
                    weight = T.load(weights[col, block])
                    q, d = weight.q, weight.d
                    for part in T.unroll(elements // width):
                        if format == "NVFP4":
                            codes = T.load(table, at=q[:, part * width : (part + 1) * width], bounds="in_bounds")
                            selected = T.load(scales, at=d[:, part], bounds="in_bounds")
                            factors = T.reshape(selected, (columns,))
                        else:
                            codes = q[:, part * width : (part + 1) * width]
                            factors = d
                        for m in T.serial(M):
                            activation = T.load(prepared[m, block][part * width // 32 * 32 : part * width // 32 * 32 + 32])
                            values = activation.q[part * width % 32 : part * width % 32 + width]
                            products = T.alloc_local((columns,), "int32")
                            T.reduce_dot(codes, values, products, over="k", acc_dtype="int32", engine="vector")
                            if format == "Q1_0":
                                wide = T.alloc_local((width,), "int32")
                                total = T.alloc_local((1,), "int32")
                                for k in T.Parallel(width):
                                    wide[k] = T.cast(activation.q[part * width % 32 + k], "int32")
                                T.reduce_sum(wide, total, dim=0)
                            prior = accumulated if M == 1 else T.load(output[m, col])
                            result = T.alloc_local((columns,), "float32")
                            for c in T.Parallel(columns):
                                integer = products[c] * 2 - total[0] if format == "Q1_0" else products[c]
                                result[c] = prior[c] + (T.cast(factors[c], "float32") * T.cast(activation.d[()], "float32")) * T.cast(integer, "float32")
                            if M == 1:
                                T.copy(result, accumulated)
                            else:
                                T.copy(result, output[m, col])
                if M == 1:
                    T.copy(accumulated, output[0, col])

    return main
