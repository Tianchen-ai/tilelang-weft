"""Quantized projections with explicit interleaved weights and live tile sums."""

import tilelang
from tilelang.weft import language as T
from . import ggml_formats as F, layouts
from .preparation import quantize_rows


@tilelang.jit(target="weft")
def projection(M, N, K, format="Q4_0", rows=4, columns=16,
               extreme_policy="first", inverse_policy="direct"):
    """Consume persistent row interleave; prepare activation inside this call.

    Each output tile retains its floating accumulator across all weight records.
    The encoding, cohort and separate preparation phase are author choices.
    Local vector layout and instruction selection remain compiler decisions.
    """
    assert format in ("Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "IQ4_NL", "MXFP4")
    assert M > 0 and N > 0 and K > 0 and K % 32 == 0
    assert M % rows == 0 and N % columns == 0
    encoding = getattr(F, format)
    persistent = getattr(layouts, format + "Rows")[columns]
    affine = format in ("Q4_1", "Q5_1")
    lookup = format in ("IQ4_NL", "MXFP4")
    bias = {"Q4_0": 8, "Q5_0": 16, "Q8_0": 128, "IQ4_NL": 128, "MXFP4": 128}.get(format, 0)
    partner = F.Q8_1 if affine else F.Q8_0
    weight_bytes = K // 32 * (encoding.schema.storage_bits // 8)
    activation_bytes = K // 32 * (partner.schema.storage_bits // 8)

    @T.prim_func
    def main(
        Weights: T.Tensor((N, weight_bytes), "uint8"),
        Activation: T.Tensor((M, K), "float32"),
        Quantized: T.Tensor((M, activation_bytes), "uint8"),
        Output: T.Tensor((M, N), "float32"),
        Sums: T.Tensor((M, K // 32 * 2) if bias else (0,), "uint8"),
        Codebook: T.Tensor((16 if lookup else 0,), "int8"),
        Scale: T.Tensor((256 if format == "MXFP4" else 0,), "float32"),
    ):
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Activation": "input",
                     "Quantized": "quantized", "Output": "output", "Sums": "sums",
                     "Codebook": "input", "Scale": "input"}})
        weights = T.view(Weights, shape=(N, K), encoding=persistent, axes=("n", "k"), abi=True)
        source = T.view(Activation, shape=(M, K), axes=("m", "k"), abi=True)
        prepared = T.view(Quantized, shape=(M, K), encoding=partner, axes=("m", "k"), abi=True)
        output = T.view(Output, shape=(M, N), axes=("m", "n"), abi=True)
        if bias:
            sums = T.view(Sums, shape=(M, K), encoding=layouts.BlockSum32, axes=("m", "k"), abi=True)
        if lookup:
            codebook = T.view(Codebook, shape=(16,), axes=("code",), abi=True)
        if format == "MXFP4":
            scale_table = T.view(Scale, shape=(256,), axes=("exponent",), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            if lookup:
                table = T.load(codebook)
            quantize_rows(source, prepared, M, K, "Q8_1" if affine else "Q8_0", 32,
                          extreme_policy, inverse_policy, "dynamic", 32)
            if bias:
                for m in T.serial(M):
                    with T.level.blocks(K, extent=32) as block:
                        values = T.load(prepared[m, block].q)
                        widened = T.alloc_local((32,), "int16")
                        reduced = T.alloc_local((1,), "int16")
                        for k in T.Parallel(32):
                            widened[k] = T.cast(values[k], "int16")
                        T.reduce_sum(widened, reduced, dim=0)
                        T.copy(reduced, sums[m, block].value)
            with T.level.cols(N, group=columns) as col:
                with T.level.rows(M, group=rows) as row:
                    total = T.state((rows, columns), "float32", init=0)
                    correction = T.state((rows, columns), "float32", init=0)
                    with T.level.blocks(K, extent=32) as block:
                        weight, activation = weights[col, block], prepared[row, block]
                        low = T.load(weight.q)
                        if format in ("Q5_0", "Q5_1"):
                            high = T.load(weight.qh)
                        inputs = T.load(activation.q)
                        codes = T.alloc_local((columns, 32), "uint8")
                        products = T.alloc_local((rows, columns), "int32")
                        for c, k in T.Parallel(columns, 32):
                            if format in ("Q5_0", "Q5_1"):
                                codes[c, k] = T.cast(low[c, k], "uint8") | (T.cast(high[c, k], "uint8") << T.uint8(4))
                            elif lookup:
                                codes[c, k] = T.cast(table[T.cast(low[c, k], "uint8")], "uint8") ^ T.uint8(128)
                            elif format == "Q8_0":
                                codes[c, k] = T.cast(low[c, k], "uint8") ^ T.uint8(128)
                            else:
                                codes[c, k] = T.cast(low[c, k], "uint8")
                        T.reduce_dot(inputs, codes, products, over="k", acc_dtype="int32", engine="vector")
                        xd = T.load(activation.d)
                        if format == "MXFP4":
                            exponent = T.load(weight.e)
                            wd = T.load(scale_table, at=exponent, bounds="in_bounds")
                        else:
                            wd = T.load(weight.d)
                        if bias:
                            block_sums = T.load(sums[row, block].value)
                        if affine:
                            minimum, activation_sum = T.load(weight.m), T.load(activation.s)
                        for r, c in T.Parallel(rows, columns):
                            integer = products[r, c] - T.int32(bias) * T.cast(block_sums[r], "int32") if bias else products[r, c]
                            total[r, c] += (T.cast(xd[r], "float32") * T.cast(wd[c], "float32")) * T.cast(integer, "float32")
                            if affine:
                                correction[r, c] += T.cast(activation_sum[r], "float32") * T.cast(minimum[c], "float32")
                    if affine:
                        for r, c in T.Parallel(rows, columns):
                            total[r, c] += correction[r, c]
                    T.copy(total, output[row, col])

    return main
