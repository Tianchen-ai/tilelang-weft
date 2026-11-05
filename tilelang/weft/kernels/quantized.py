"""Encoded author programs retain field layouts, typed partials and scale order."""

import tilelang
from tilelang.weft import language as T
from . import ggml_formats as F
from .decoding import decode_dequantize, decode_dequantize_iq4_nl
from . import partials


@tilelang.jit(target="weft")
def q4_k_q8_k(M, K, rows=4):
    assert K > 0 and K % 256 == 0

    @T.prim_func
    def main(
        Weights: T.Tensor((M, K // 256 * 144), "uint8"), Activation: T.Tensor((K // 256 * 292,), "uint8"), Output: T.Tensor((M,), "float32")
    ):
        T.func_attr({"weft.alias_groups": {"Output": "output"}})
        wv = T.view(Weights, shape=(M, K), encoding=F.Q4_K, axes=("m", "k"), abi=True)
        xv = T.view(Activation, shape=(K,), encoding=F.Q8_K, axes=("k",))
        yv = T.view(Output, shape=(M,), axes=("m",), abi=True)
        with T.Kernel(1):
            with T.level.rows(M, group=rows) as row:
                total = T.state((rows,), "float32", init=0)
                with T.level.blocks(K, extent=256) as block:
                    weight = T.stage(T.load(wv[row, block]))
                    activation = T.stage(T.load(xv[block]))
                    partials.q4_k(weight, activation, total, rows)
                T.copy(total, yv[row])

    return main


@tilelang.jit(target="weft")
def q6_k_q8_k(M, K, rows=1):
    assert K > 0 and K % 256 == 0

    @T.prim_func
    def main(
        Weights: T.Tensor((M, K // 256 * 210), "uint8"), Activation: T.Tensor((K // 256 * 292,), "uint8"), Output: T.Tensor((M,), "float32")
    ):
        T.func_attr({"weft.alias_groups": {"Output": "output"}})
        wv = T.view(Weights, shape=(M, K), encoding=F.Q6_K, axes=("m", "k"), abi=True)
        xv = T.view(Activation, shape=(K,), encoding=F.Q8_K, axes=("k",))
        yv = T.view(Output, shape=(M,), axes=("m",), abi=True)
        with T.Kernel(1):
            with T.level.rows(M, group=rows) as row:
                total = T.state((rows,), "float32", init=0)
                with T.level.blocks(K, extent=256) as block:
                    weight = T.stage(T.load(wv[row, block]))
                    activation = T.stage(T.load(xv[block]))
                    partials.q6_k(weight, activation, total, rows)
                T.copy(total, yv[row])

    return main


@tilelang.jit(target="weft")
def dequantize(M, K, format="Q4_0", rows=4):
    if format not in ("Q1_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q8_1", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K", "TQ2_0"):
        raise ValueError(f"no integer-field author dequantizer for {format}")
    encoding = getattr(F, format)
    block = encoding.schema.elements
    record_bytes = encoding.schema.storage_bits // 8
    assert K > 0 and K % block == 0

    @T.prim_func
    def main(Weights: T.Tensor((M, K // block * record_bytes), "uint8"), Output: T.Tensor((M, K), "float32")):
        T.func_attr({"weft.alias_groups": {"Output": "output"}})
        wv = T.view(Weights, shape=(M, K), encoding=encoding, axes=("m", "k"), abi=True)
        yv = T.view(Output, shape=(M, K), axes=("m", "k"), abi=True)
        with T.Kernel(1):
            with T.level.rows(M, group=rows) as row:
                with T.level.blocks(K, extent=block) as part:
                    record = T.load(wv[row, part])
                    result = T.alloc_local((rows, block), "float32")
                    if block > 32:
                        for segment in T.unroll(block // 32):
                            chunk = T.alloc_local((rows, 32), "float32")
                            decode_dequantize(record, chunk, rows, format, block, 0, 0, 0, 0, 0, 0, offset=segment * 32, width=32)
                            for r, k in T.Parallel(rows, 32):
                                result[r, segment * 32 + k] = chunk[r, k]
                    else:
                        decode_dequantize(record, result, rows, format, block, 0, 0, 0, 0, 0, 0)
                    T.copy(result, yv[row, part])

    return main


@tilelang.jit(target="weft")
def dequantize_iq4_nl(M, K, rows=1):
    assert K > 0 and K % 32 == 0

    @T.prim_func
    def main(Weights: T.Tensor((M, K // 32 * 18), "uint8"), Codebook: T.Tensor((16,), "int8"), Output: T.Tensor((M, K), "float32")):
        T.func_attr({"weft.alias_groups": {"Output": "output"}})
        wv = T.view(Weights, shape=(M, K), encoding=F.IQ4_NL, axes=("m", "k"), abi=True)
        yv = T.view(Output, shape=(M, K), axes=("m", "k"), abi=True)
        with T.Kernel(1):
            table = T.alloc_local((16,), "int8")
            T.copy(Codebook, table)
            with T.level.rows(M, group=rows) as row:
                with T.level.blocks(K, extent=32) as block:
                    record = T.load(wv[row, block])
                    result = T.alloc_local((rows, 32), "float32")
                    decode_dequantize_iq4_nl(record, result, rows, "DEQUANTIZE_IQ4_NL", 32, 0, 0, table, 0, 0, 0)
                    T.copy(result, yv[row, block])

    return main
