"""Block-local packed arithmetic, sharing admitted fields and decoded tiles.

The caller supplies serialized weights and activations. Decoding stays inside
the block lifetime; no persistent dense weight matrix is introduced.
"""

import tilelang
from tilelang.weft import language as T
from . import ggml_formats as F
from . import decoding as D
from .preparation import quantize_rows
from . import partials


def decoder(format):
    if format in ("Q1_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q8_1", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K", "TQ2_0"):
        return D.decode_dequantize
    elif format == "IQ4_NL":
        return D.decode_dequantize_iq4_nl
    elif format == "IQ1_S":
        return D.decode_iq1_s
    elif format == "IQ1_M":
        return D.decode_iq1_m
    elif format == "IQ2_S":
        return D.decode_iq2_s
    elif format in ("IQ2_XS", "IQ2_XXS"):
        return D.decode_iq2_indexed
    elif format == "IQ3_S":
        return D.decode_iq3_s
    elif format == "IQ3_XXS":
        return D.decode_iq3_xxs
    elif format == "IQ4_XS":
        return D.decode_iq4_xs
    elif format == "TQ1_0":
        return D.decode_tq1_0
    elif format in ("MXFP4", "NVFP4"):
        return D.decode_fp4
    raise ValueError(f"no packed decoder for {format}")


@T.macro
def accumulate(weight, activation_view, part, total, rows, block, format, activation, partner, decode,
               Grid, Signs, Codebook, Scale, HalfBits, Powers):
    if format == "Q4_K":
        x = T.load(activation_view[part])
        partials.q4_k(weight, x, total, rows, "float64")
    elif format == "Q6_K":
        x = T.load(activation_view[part])
        partials.q6_k(weight, x, total, rows, "float64")
    elif format in ("Q4_1", "Q5_1"):
        x = T.stage(T.load(activation_view[part]))
        q = weight.q
        if format == "Q5_1":
            high = weight.qh
        integer = T.alloc_local((rows, 32), "int8")
        partial = T.alloc_local((rows, 1), "int32")
        for r, k in T.Parallel(rows, 32):
            if format == "Q5_1":
                integer[r, k] = T.cast(T.cast(q[r, k], "int32") | (T.cast(high[r, k], "int32") << 4), "int8")
            else:
                integer[r, k] = T.cast(q[r, k], "int8")
        T.mac_groups(integer, x.q, partial, n=32, into="int32", over="k")
        for r in T.Parallel(rows):
            total[r] += T.cast(weight.d[r], "float64") * T.cast(x.d[()], "float64") * T.cast(partial[r, 0], "float64")
            total[r] += T.cast(weight.m[r], "float64") * T.cast(x.s[()], "float64")
    elif block == partner.schema.elements:
        x = T.stage(T.load(activation_view[part]))
        q = x.q
        scale = x.ds if activation == "Q8_K" else x.d
        for segment in T.serial(block // 32):
            value = T.alloc_local((rows, 32), "float32")
            products = T.alloc_local((rows, 32), "float64")
            subtotal = T.alloc_local((rows,), "float64")
            decode(weight, value, rows, format, block, Grid, Signs, Codebook, Scale, HalfBits, Powers,
                   offset=segment * 32, width=32)
            for r, k in T.Parallel(rows, 32):
                products[r, k] = T.cast(value[r, k], "float64") * T.cast(q[segment * 32 + k], "float64") * T.cast(scale[()], "float64")
            T.reduce_sum(products, subtotal, dim=1)
            for r in T.Parallel(rows):
                total[r] += subtotal[r]
    else:
        codes = weight.q
        weight_scale = weight.d
        with T.level.subtiles(part, extent=partner.schema.elements) as sub:
            x = T.stage(T.load(activation_view[sub]))
            if format == "Q1_0":
                q = T.load(codes, at=sub)
                centered = T.alloc_local((rows, 32), "int8")
                partial = T.alloc_local((rows, 1), "int32")
                for r, k in T.Parallel(rows, 32):
                    centered[r, k] = T.cast(q[r, k], "int8") * 2 - 1
                T.mac_groups(centered, x.q, partial, n=32, into="int32", over="k")
                for r in T.Parallel(rows):
                    total[r] += T.cast(weight_scale[r], "float64") * T.cast(x.d[()], "float64") * T.cast(partial[r, 0], "float64")
            else:
                activation_codes = x.q
                with T.level.subtiles(sub, extent=16) as lane:
                    q = T.load(codes, at=lane)
                    xq = T.load(activation_codes, at=lane)
                    scale = T.load(weight_scale, at=lane)
                    decoded = T.alloc_local((rows, 16), "int8")
                    partial = T.alloc_local((rows, 1), "int32")
                    for r, k in T.Parallel(rows, 16):
                        decoded[r, k] = Codebook[T.cast(q[r, k], "int32")]
                    T.mac_groups(decoded, xq, partial, n=16, into="int32", over="k")
                    for r in T.Parallel(rows):
                        total[r] += T.cast(Scale[T.cast(scale[r], "int32")], "float64") * T.cast(x.d[()], "float64") * T.cast(partial[r, 0], "float64")


def table_sizes(format):
    return (
        {"IQ1_S": 16384, "IQ1_M": 16384, "IQ2_S": 8192, "IQ2_XS": 4096,
         "IQ2_XXS": 2048, "IQ3_S": 2048, "IQ3_XXS": 1024}.get(format, 0),
        1024 if format in ("IQ2_XS", "IQ2_XXS", "IQ3_XXS") else 0,
        16 if format in ("IQ4_NL", "IQ4_XS", "MXFP4", "NVFP4") else 0,
        256 if format in ("MXFP4", "NVFP4") else 0,
        65536 if format == "IQ1_M" else 0,
        5 if format == "TQ1_0" else 0,
    )


@tilelang.jit(target="weft")
def vec_dot(M, K, format="Q4_0", activation="Q8_0", rows=1):
    encoding, partner = getattr(F, format), getattr(F, activation)
    block = encoding.schema.elements
    weight_bytes = K // block * (encoding.schema.storage_bits // 8)
    activation_bytes = K // partner.schema.elements * (partner.schema.storage_bits // 8)
    assert K > 0 and K % block == 0 and K % partner.schema.elements == 0
    grid_size, signs_size, codebook_size, scale_size, half_size, powers_size = table_sizes(format)
    decode = decoder(format)

    @T.prim_func
    def main(
        Weights: T.Tensor((M, weight_bytes), "uint8"),
        Activation: T.Tensor((activation_bytes,), "uint8"),
        Output: T.Tensor((M,), "float32"),
        Grid: T.Tensor((grid_size,), "int8"), Signs: T.Tensor((signs_size,), "int8"),
        Codebook: T.Tensor((codebook_size,), "int8"), Scale: T.Tensor((scale_size,), "float32"),
        HalfBits: T.Tensor((half_size,), "float32"), Powers: T.Tensor((powers_size,), "uint32"),
    ):
        weights = T.view(Weights, shape=(M, K), encoding=encoding, axes=("m", "k"))
        activation_view = T.view(Activation, shape=(K,), encoding=partner, axes=("k",))
        output = T.view(Output, shape=(M,), axes=("m",))
        with T.Kernel(1, cpu_num_threads=1):
            with T.level.rows(M, group=rows) as row:
                total = T.state((rows,), "float64", init=0)
                with T.level.blocks(K, extent=block) as part:
                    weight = T.stage(T.load(weights[row, part]))
                    accumulate(weight, activation_view, part, total, rows, block, format, activation, partner, decode,
                               Grid, Signs, Codebook, Scale, HalfBits, Powers)
                T.copy(total, output[row])

    return main


@tilelang.jit(target="weft")
def projection(M, N, K, format="Q4_0", activation="Q8_0", rows=4,
               extreme_policy="first", inverse_policy="reciprocal"):
    encoding, partner = getattr(F, format), getattr(F, activation)
    block = encoding.schema.elements
    weight_bytes = K // block * (encoding.schema.storage_bits // 8)
    activation_bytes = K // partner.schema.elements * (partner.schema.storage_bits // 8)
    grid_size, signs_size, codebook_size, scale_size, half_size, powers_size = table_sizes(format)
    decode = decoder(format)

    @T.prim_func
    def main(
        Weights: T.Tensor((N, weight_bytes), "uint8"), Activation: T.Tensor((M, K), "float32"),
        Quantized: T.Tensor((M, activation_bytes), "uint8"), Output: T.Tensor((M, N), "float32"),
        Grid: T.Tensor((grid_size,), "int8"), Signs: T.Tensor((signs_size,), "int8"),
        Codebook: T.Tensor((codebook_size,), "int8"), Scale: T.Tensor((scale_size,), "float32"),
        HalfBits: T.Tensor((half_size,), "float32"), Powers: T.Tensor((powers_size,), "uint32"),
    ):
        weights = T.view(Weights, shape=(N, K), encoding=encoding, axes=("n", "k"))
        source = T.view(Activation, shape=(M, K), axes=("m", "k"))
        quantized = T.view(Quantized, shape=(M, K), encoding=partner, axes=("m", "k"))
        output = T.view(Output, shape=(M, N), axes=("m", "n"))
        with T.Kernel(1, cpu_num_threads=1):
            quantize_rows(source, quantized, M, K, activation, partner.schema.elements, extreme_policy, inverse_policy, "rne")
            for m in T.serial(M):
                with T.level.cols(N, group=rows) as column:
                    total = T.state((rows,), "float64", init=0)
                    with T.level.blocks(K, extent=block) as part:
                        weight = T.stage(T.load(weights[column, part]))
                        accumulate(weight, quantized[m, :], part, total, rows, block, format, activation, partner, decode,
                                   Grid, Signs, Codebook, Scale, HalfBits, Powers)
                    T.copy(total, output[m, column])

    return main
