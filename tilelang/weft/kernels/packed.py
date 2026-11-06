"""Block-local packed arithmetic, sharing admitted fields and decoded tiles.

The caller supplies serialized weights and activations. Decoding stays inside
the block lifetime; no persistent dense weight matrix is introduced.
"""

import tilelang
from tilelang.weft import language as T
from . import ggml_formats as F
from .preparation import quantize_rows
from . import partials
from . import integer_dot


@T.macro
def accumulate(
    weight,
    activation_view,
    part,
    total,
    rows,
    format,
    partner,
    Grid,
    Signs,
    Codebook,
    Scale,
    HalfBits,
    Powers,
    single=False,
    scale_groups=1,
):
    if format == "Q4_K":
        x = T.load(activation_view[part])
        if single:
            partials.q4_k_single(weight, x, total)
        else:
            partials.q4_k(weight, x, total, rows)
    if format == "Q6_K":
        x = T.load(activation_view[part])
        if single:
            partials.q6_k_single(weight, x, total, scale_groups, part)
        else:
            partials.q6_k(weight, x, total, rows)
    if format in ("Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q8_1", "IQ4_NL", "MXFP4"):
        x = T.load(activation_view[part])
        integer_dot.block32(weight, x, total, rows, format, Codebook, Scale)
    if format in ("Q2_K", "Q3_K", "Q5_K", "Q8_K", "TQ2_0"):
        x = T.load(activation_view[part])
        integer_dot.block256(weight, x, total, rows, format)
    if format in ("IQ2_XS", "IQ2_XXS"):
        x = T.load(activation_view[part])
        integer_dot.iq2_indexed(weight, x, total, rows, format, Grid, Signs)
    if format in ("IQ1_S", "IQ1_M"):
        x = T.load(activation_view[part])
        integer_dot.iq1(weight, x, total, rows, format, Grid, HalfBits)
    if format in ("IQ2_S", "IQ3_S", "IQ3_XXS", "IQ4_XS"):
        x = T.load(activation_view[part])
        integer_dot.scaled_lookup(weight, x, total, rows, format, Grid, Signs, Codebook)
    if format == "TQ1_0":
        x = T.load(activation_view[part])
        integer_dot.ternary(weight, x, total, rows, Powers)
    if format in ("Q1_0", "NVFP4"):
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
                    total[r] += T.cast(weight_scale[r], "float32") * T.cast(x.d[()], "float32") * T.cast(partial[r, 0], "float32")
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
                        total[r] += Scale[T.cast(scale[r], "int32")] * T.cast(x.d[()], "float32") * T.cast(partial[r, 0], "float32")


def table_sizes(format):
    return (
        {"IQ1_S": 16384, "IQ1_M": 16384, "IQ2_S": 8192, "IQ2_XS": 4096, "IQ2_XXS": 2048, "IQ3_S": 2048, "IQ3_XXS": 1024}.get(format, 0),
        1024 if format in ("IQ2_XS", "IQ2_XXS", "IQ3_XXS") else 0,
        16 if format in ("IQ4_NL", "IQ4_XS", "MXFP4", "NVFP4") else 0,
        256 if format in ("MXFP4", "NVFP4") else 0,
        65536 if format == "IQ1_M" else 0,
        5 if format == "TQ1_0" else 0,
    )


@tilelang.jit(target="weft")
def vec_dot(M, K, format="Q4_0", activation="Q8_0", rows=1, scale_groups=1):
    assert scale_groups in (1, 2, 4, 8, 16)
    encoding, partner = getattr(F, format), getattr(F, activation)
    block = encoding.schema.elements
    weight_bytes = K // block * (encoding.schema.storage_bits // 8)
    activation_bytes = K // partner.schema.elements * (partner.schema.storage_bits // 8)
    assert K > 0 and K % block == 0 and K % partner.schema.elements == 0
    grid_size, signs_size, codebook_size, scale_size, half_size, powers_size = table_sizes(format)
    accumulator = "float32"

    @T.prim_func
    def main(
        Weights: T.Tensor((M, weight_bytes), "uint8"),
        Activation: T.Tensor((activation_bytes,), "uint8"),
        Output: T.Tensor((M,), "float32"),
        Grid: T.Tensor((grid_size,), "int8"),
        Signs: T.Tensor((signs_size,), "int8"),
        Codebook: T.Tensor((codebook_size,), "int8"),
        Scale: T.Tensor((scale_size,), "float32"),
        HalfBits: T.Tensor((half_size,), "float32"),
        Powers: T.Tensor((powers_size,), "uint32"),
    ):
        T.func_attr({"tir.noalias": True})
        weights = T.view(Weights, shape=(M, K), encoding=encoding, axes=("m", "k"), abi=True)
        activation_view = T.view(Activation, shape=(K,), encoding=partner, axes=("k",), abi=True)
        output = T.view(Output, shape=(M,), axes=("m",), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            grid, signs = Grid, Signs
            if format in ("IQ2_XS", "IQ2_XXS"):
                grid_table = T.view(Grid, shape=(grid_size // 8, 8), encoding=F.I8X8, axes=("grid_entry", "grid_payload"), abi=True)
                sign_table = T.view(Signs, shape=(128, 8), encoding=F.I8X8, axes=("sign_entry", "sign_payload"), abi=True)
                grid, signs = grid_table.values, sign_table.values
            codebook = Codebook
            if codebook_size:
                codebook = T.alloc_local((codebook_size,), "int8")
                T.copy(Codebook, codebook)
            with T.level.rows(M, group=rows) as row:
                total = T.state(() if format in ("Q4_K", "Q6_K") and M == rows == 1 else (rows,), accumulator, init=0)
                with T.level.blocks(K, extent=block) as part:
                    if format in ("Q4_K", "Q6_K") and M == rows == 1:
                        weight = T.load(weights[0, part])
                    else:
                        weight = T.load(weights[row, part])
                    accumulate(
                        weight,
                        activation_view,
                        part,
                        total,
                        rows,
                        format,
                        partner,
                        grid,
                        signs,
                        codebook,
                        Scale,
                        HalfBits,
                        Powers,
                        single=format in ("Q4_K", "Q6_K") and M == rows == 1,
                        scale_groups=scale_groups,
                    )
                T.copy(total, output[row])

    return main


@tilelang.jit(target="weft")
def projection(M, N, K, format="Q4_0", activation="Q8_0", rows=4, extreme_policy="first", inverse_policy="reciprocal", scale_groups=1):
    assert scale_groups in (1, 2, 4, 8, 16)
    encoding, partner = getattr(F, format), getattr(F, activation)
    block = encoding.schema.elements
    weight_bytes = K // block * (encoding.schema.storage_bits // 8)
    activation_bytes = K // partner.schema.elements * (partner.schema.storage_bits // 8)
    grid_size, signs_size, codebook_size, scale_size, half_size, powers_size = table_sizes(format)
    accumulator = "float32"
    single = rows == 1 and format in ("Q4_K", "Q6_K")

    @T.prim_func
    def main(
        Weights: T.Tensor((N, weight_bytes), "uint8"),
        Activation: T.Tensor((M, K), "float32"),
        Quantized: T.Tensor((M, activation_bytes), "uint8"),
        Output: T.Tensor((M, N), "float32"),
        Grid: T.Tensor((grid_size,), "int8"),
        Signs: T.Tensor((signs_size,), "int8"),
        Codebook: T.Tensor((codebook_size,), "int8"),
        Scale: T.Tensor((scale_size,), "float32"),
        HalfBits: T.Tensor((half_size,), "float32"),
        Powers: T.Tensor((powers_size,), "uint32"),
    ):
        T.func_attr({"tir.noalias": True})
        weights = T.view(Weights, shape=(N, K), encoding=encoding, axes=("n", "k"), abi=True)
        source = T.view(Activation, shape=(M, K), axes=("m", "k"), abi=True)
        quantized = T.view(Quantized, shape=(M, K), encoding=partner, axes=("m", "k"), abi=True)
        output = T.view(Output, shape=(M, N), axes=("m", "n"), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            grid, signs = Grid, Signs
            if format in ("IQ2_XS", "IQ2_XXS"):
                grid_table = T.view(Grid, shape=(grid_size // 8, 8), encoding=F.I8X8, axes=("grid_entry", "grid_payload"), abi=True)
                sign_table = T.view(Signs, shape=(128, 8), encoding=F.I8X8, axes=("sign_entry", "sign_payload"), abi=True)
                grid, signs = grid_table.values, sign_table.values
            codebook = Codebook
            if codebook_size:
                codebook = T.alloc_local((codebook_size,), "int8")
                T.copy(Codebook, codebook)
            quantize_rows(source, quantized, M, K, activation, partner.schema.elements, extreme_policy, inverse_policy, "rne")
            for m in T.serial(M):
                with T.level.cols(N, group=rows) as column:
                    total = T.state(() if single else (rows,), accumulator, init=0)
                    with T.level.blocks(K, extent=block) as part:
                        if single:
                            weight = T.load(weights[column, part][0, :])
                        else:
                            weight = T.load(weights[column, part])
                        accumulate(
                            weight,
                            quantized[m, :],
                            part,
                            total,
                            rows,
                            format,
                            partner,
                            grid,
                            signs,
                            codebook,
                            Scale,
                            HalfBits,
                            Powers,
                            single=single,
                            scale_groups=scale_groups,
                        )
                    T.copy(total, output[m, column])

    return main
