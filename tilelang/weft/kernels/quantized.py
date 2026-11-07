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
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Activation": "input", "Output": "output"}})
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
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Activation": "input", "Output": "output"}})
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
def dequantize(M, K, format="Q4_0", rows=4, chunk=32, records=1, cache_hint="normal"):
    if format not in ("Q1_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q8_1", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K", "TQ2_0"):
        raise ValueError(f"no integer-field author dequantizer for {format}")
    encoding = getattr(F, format)
    block = encoding.schema.elements
    assert records in (1, 2, 4, 8) and (records == 1 or block == 32)
    assert chunk in (4, 8, 16, 32, 64, 128, 256) and block % chunk == 0
    record_bytes = encoding.schema.storage_bits // 8
    assert K > 0 and K % (block * records) == 0

    @T.prim_func
    def main(Weights: T.Tensor((M, K // block * record_bytes), "uint8"), Output: T.Tensor((M, K), "float32")):
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Output": "output"}})
        if rows == 1:
            wv = T.view(Weights, shape=(M * K,), encoding=encoding, axes=("k",))
            yv = T.view(Output, shape=(M * K,), axes=("k",))
        else:
            wv = T.view(Weights, shape=(M, K), encoding=encoding, axes=("m", "k"), abi=True)
            yv = T.view(Output, shape=(M, K), axes=("m", "k"), abi=True)
        with T.Kernel(1):
            if rows == 1:
                with T.level.blocks(M * K, extent=block * records) as part:
                    if format == "Q8_0" and records == 1:
                        record = wv[part]
                        scale = T.load(record.d)
                        for segment in T.unroll(block // chunk):
                            codes = T.load(record.q[segment * chunk : (segment + 1) * chunk])
                            values = T.alloc_local((chunk,), "float32")
                            for lane in T.Parallel(chunk):
                                values[lane] = T.cast(codes[lane], "float32") * T.cast(scale[()], "float32")
                            T.copy(values, yv[part][segment * chunk : (segment + 1) * chunk], annotations={"cache_hint": cache_hint})
                    elif records > 1:
                        record = T.reshape(T.load(wv[part]), (records, block), axes=("record", "k"))
                        result = T.alloc_local((records, block), "float32")
                        decode_dequantize(record, result, records, format, block, 0, 0, 0, 0, 0, 0)
                        flattened = T.reshape(result, (records * block,))
                        T.copy(flattened, yv[part], annotations={"cache_hint": cache_hint})
                    else:
                        record = T.reshape(T.load(wv[part]), (1, block), axes=("record", "k"))
                        for segment in T.unroll(block // chunk):
                            values = T.alloc_local((1, chunk), "float32")
                            decode_dequantize(record, values, 1, format, block, 0, 0, 0, 0, 0, 0,
                                              offset=segment * chunk, width=chunk)
                            T.copy(values[0, :], yv[part][segment * chunk : (segment + 1) * chunk], annotations={"cache_hint": cache_hint})
            else:
                with T.level.rows(M, group=rows) as row:
                    with T.level.blocks(K, extent=block * records) as part:
                        loaded = T.load(wv[row, part])
                        if records > 1:
                            record = T.reshape(loaded, (rows * records, block), axes=("record", "k"))
                            result = T.alloc_local((rows * records, block), "float32")
                            decode_dequantize(record, result, rows * records, format, block, 0, 0, 0, 0, 0, 0)
                            flattened = T.reshape(result, (rows, block * records))
                            T.copy(flattened, yv[row, part], annotations={"cache_hint": cache_hint})
                        else:
                            for segment in T.unroll(block // chunk):
                                values = T.alloc_local((rows, chunk), "float32")
                                decode_dequantize(loaded, values, rows, format, block, 0, 0, 0, 0, 0, 0, offset=segment * chunk, width=chunk)
                                T.copy(values, yv[row, part][:, segment * chunk : (segment + 1) * chunk], annotations={"cache_hint": cache_hint})

    return main


@tilelang.jit(target="weft")
def field_windows(M, K, format="Q4_K", chunk=None):
    """Decode one scale group at a time from its original record snapshot."""
    assert format in ("Q1_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "TQ2_0")
    encoding = getattr(F, format)
    elements = encoding.schema.elements
    group_width = 16 if format in ("Q2_K", "Q3_K", "Q6_K") else 32
    width = group_width if chunk is None else chunk
    assert width in (8, 16, 32) and group_width % width == 0
    record_bytes = encoding.schema.storage_bits // 8
    assert M > 0 and K > 0 and K % elements == 0

    @T.prim_func
    def main(Weights: T.Tensor((M, K // elements * record_bytes), "uint8"), Output: T.Tensor((M, K), "float32")):
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Output": "output"}})
        weights = T.view(Weights, shape=(M * K,), encoding=encoding, axes=("k",))
        output = T.view(Output, shape=(M * K,), axes=("k",))
        with T.Kernel(1, cpu_num_threads=1):
            with T.level.blocks(M * K, extent=elements) as block:
                record = T.load(weights[block])
                if elements == width:
                    result = T.alloc_local((1, width), "float32")
                    decode_dequantize(record, result, 1, format, elements, 0, 0, 0, 0, 0, 0,
                                      width=width, single=True, code_dtype="int8")
                    T.copy(result[0, :], output[block])
                else:
                    with T.level.subtiles(block, extent=width) as window:
                        result = T.alloc_local((1, width), "float32")
                        decode_dequantize(record, result, 1, format, elements, 0, 0, 0, 0, 0, 0,
                                          width=width, single=True, point=window, code_dtype="int8")
                        T.copy(result[0, :], output[window])

    return main


@tilelang.jit(target="weft")
def dequantize_iq4_nl(M, K, rows=1):
    assert K > 0 and K % 32 == 0

    @T.prim_func
    def main(Weights: T.Tensor((M, K // 32 * 18), "uint8"), Codebook: T.Tensor((16,), "int8"), Output: T.Tensor((M, K), "float32")):
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Codebook": "input", "Output": "output"}})
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
