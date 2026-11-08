"""Packed tile programs; layouts and engine requirements belong to the author."""

import tilelang
from tilelang.weft import language as T
from . import layouts as L
from . import ggml_formats as F
from .preparation import quantize_rows


@T.macro
def k_layout_row(weights, activation, output, N, K, format, columns, cohort=32, groups_per_step=16):
    """Consume the explicitly prepared interleaved model layout by scale group."""
    width = 32 if format == "Q4_K" else 16
    cohort_size = min(columns, cohort)
    with T.level.cols(N, group=columns) as col:
        total = T.state((columns,), "float32", init=0)
        with T.level.blocks(K, extent=256) as block:
            x = T.load(activation[block])
            sums, ds = x.bsum, x.ds
            activation_codes = activation[block].q
            if format == "Q4_K":
                pairs = T.alloc_local((8,), "int16")
                for g in T.Parallel(8):
                    pairs[g] = T.cast(T.cast(sums[g * 2], "int32") + T.cast(sums[g * 2 + 1], "int32"), "int16")
            for cohort in T.unroll(columns // cohort_size):
                record = weights[col, block][cohort * cohort_size : (cohort + 1) * cohort_size, :]
                integer = T.alloc_local((cohort_size,), "int32")
                T.fill(integer, 0)
                if format == "Q4_K":
                    scales = T.load(record.scales)
                for first in T.serial((256 // width + groups_per_step - 1) // groups_per_step):
                    for offset in T.unroll(min(groups_per_step, 256 // width)):
                        group = first * groups_per_step + offset
                        low = T.load(record.q[:, group * width : (group + 1) * width] if format == "Q4_K" else record.low[:, group * width : (group + 1) * width])
                        inputs = T.load(activation_codes[group * width : (group + 1) * width])
                        codes = T.alloc_local((cohort_size, width), "uint4" if format == "Q4_K" else "uint6")
                        products = T.alloc_local((cohort_size,), "int32")
                        if format == "Q6_K":
                            high = T.load(record.high[:, group * width : (group + 1) * width])
                            scale_column = T.load(record.scales[:, group])
                        for c, k in T.Parallel(cohort_size, width):
                            codes[c, k] = (
                                low[c, k]
                                if format == "Q4_K"
                                else T.cast(low[c, k], "uint8") | (T.cast(high[c, k], "uint8") << T.uint8(4))
                            )
                        T.reduce_dot(codes, inputs, products, over="k", acc_dtype="int32", engine="vector")
                        for c in T.Parallel(cohort_size):
                            if format == "Q4_K":
                                integer[c] += products[c] * T.cast(scales[c, group], "int32")
                            else:
                                integer[c] += (products[c] - T.int32(32) * T.cast(sums[group], "int32")) * T.cast(scale_column[c], "int32")
                d = T.load(record.d)
                if format == "Q4_K":
                    dmin = T.load(record.dmin)
                    minimum = T.load(record.mins)
                    minima = T.alloc_local((cohort_size, 8), "int16")
                    correction = T.alloc_local((cohort_size,), "int32")
                    for c, g in T.Parallel(cohort_size, 8):
                        minima[c, g] = T.cast(minimum[c, g], "int16")
                    T.reduce_dot(minima, pairs, correction, over="k", acc_dtype="int32", engine="vector")
                for c in T.Parallel(cohort_size):
                    position = cohort * cohort_size + c
                    total[position] += (T.cast(d[c], "float32") * ds[()]) * T.cast(integer[c], "float32")
                    if format == "Q4_K":
                        total[position] += (T.cast(dmin[c], "float32") * ds[()]) * -T.cast(correction[c], "float32")
        T.copy(total, output[col])


@tilelang.jit(target="weft")
def k_layout_projection(M, N, K, format="Q4_K", columns=128):
    """Complete projection over caller-prepared GGML interleaved weights."""
    assert format in ("Q4_K", "Q6_K") and N % 16 == 0 and K % 256 == 0
    encoding = L.AMXQ4_KRows[16] if format == "Q4_K" else L.AMXQ6_KRows[16]
    record_bytes = 148 if format == "Q4_K" else 210

    @T.prim_func
    def main(
        Weights: T.Tensor((N * K // 256 * record_bytes,), "uint8"),
        Activation: T.Tensor((M, K), "float32"),
        Quantized: T.Tensor((M, K // 256 * 292), "uint8"),
        Output: T.Tensor((M, N), "float32"),
    ):
        T.func_attr({"tir.noalias": True})
        weights = T.view(Weights, shape=(N, K), encoding=encoding, axes=("n", "k"), abi=True)
        source = T.view(Activation, shape=(M, K), axes=("m", "k"), abi=True)
        prepared = T.view(Quantized, shape=(M, K), encoding=F.Q8_K, axes=("m", "k"), abi=True)
        output = T.view(Output, shape=(M, N), axes=("m", "n"), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            quantize_rows(source, prepared, M, K, "Q8_K", 256, "first", "direct", "rne", 256)
            for m in T.serial(M):
                k_layout_row(weights, prepared[m, :], output[m, :], N, K, format, columns)

    return main


@tilelang.jit(target="weft")
def ime(M, N, K, generation=1, affine=False, tile_rows=4, columns=16, engine="matrix"):
    assert generation in (1, 2)
    if generation == 1:
        weight = L.IMEQ4Affine_I[16] if affine else L.IMEQ4Centered_I[16]
        activation = L.IMEQ8_I[tile_rows]
        weight_bytes, activation_bytes = 19 if affine else 18, 36
    else:
        weight = L.IME2Weight4AffineRows[32] if affine else L.IME2Weight8Rows[32]
        activation = L.IME2Activation8Rows[1 if tile_rows == 1 else 4]
        weight_bytes, activation_bytes = 19 if affine else 34, 38

    @T.prim_func
    def main(
        W: T.Tensor((N * (K // 32) * weight_bytes,), "uint8"),
        Xq: T.Tensor((M * (K // 32) * activation_bytes,), "uint8"),
        Y: T.Tensor((M, N), "float32"),
    ):
        T.func_attr({"weft.alias_groups": {"W": "input", "Xq": "input", "Y": "output"}})
        wv = T.view(W, shape=(N, K), encoding=weight, axes=("n", "k"), abi=True)
        xv = T.view(Xq, shape=(M, K), encoding=activation, axes=("m", "k"), abi=True)
        yv = T.view(Y, shape=(M, N), axes=("m", "n"), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            with T.level.rows(M, group=tile_rows) as row:
                with T.level.cols(N, group=columns) as col:
                    total = T.state((tile_rows, columns), "float32", init=0)
                    with T.level.blocks(K, extent=32) as part:
                        w = T.load(wv[col, part])
                        x = T.load(xv[row, part])
                        q, xd, wd = w.q, x.d, w.d
                        if affine:
                            zero = w.zp
                        if generation == 2 and affine:
                            integer = T.alloc_local((tile_rows, columns), "int32")
                        else:
                            integer = T.state((tile_rows, columns), "int32", init=0)
                        if generation == 2 and affine:
                            xq = x.q
                            high = T.alloc_local((tile_rows, 32), "int4")
                            low = T.alloc_local((tile_rows, 32), "uint4")
                            high_product = T.alloc_local((tile_rows, columns), "int32")
                            low_product = T.alloc_local((tile_rows, columns), "int32")
                            negative = x.neg_sum
                            for m, k in T.Parallel(tile_rows, 32):
                                high[m, k] = T.cast(xq[m, k] >> 4, "int4")
                                low[m, k] = T.cast(xq[m, k], "uint4")
                            T.reduce_dot(high, q, high_product, over="k", acc_dtype="int32", engine=engine)
                            T.reduce_dot(low, q, low_product, over="k", acc_dtype="int32", engine=engine)
                            for m, n in T.Parallel(tile_rows, columns):
                                integer[m, n] = (
                                    high_product[m, n] * 16 + low_product[m, n] + T.cast(negative[m], "int32") * T.cast(zero[n], "int32")
                                )
                        elif generation == 1:
                            xq = x.q
                            with T.level.subtiles(part, extent=8) as sub:
                                left = T.load(xq, at=sub)
                                right = T.load(q, at=sub)
                                values = T.alloc_local((columns, 8), "int8")
                                for n, k in T.Parallel(columns, 8):
                                    values[n, k] = T.cast(right[n, k], "int8") - T.cast(zero[n] if affine else 8, "int8")
                                T.gemm(left, values, integer, transpose_B=True, engine=engine)
                        else:
                            xq = x.q
                            with T.level.subtiles(part, extent=16) as sub:
                                left = T.load(xq, at=sub)
                                right = T.load(q, at=sub)
                                partial = T.alloc_local((tile_rows, columns), "int32")
                                T.reduce_dot(left, right, partial, over="k", acc_dtype="int32", engine=engine)
                                for m, n in T.Parallel(tile_rows, columns):
                                    integer[m, n] += partial[m, n]
                        for m, n in T.Parallel(tile_rows, columns):
                            total[m, n] += (xd[m] * T.cast(wd[n], "float32")) * T.cast(integer[m, n], "float32")
                    T.copy(total, yv[row, col])

    return main


@T.macro
def q4_0_decode_tiles(wv, xv, yv, N, K, columns, sums=None):
    with T.level.cols(N, group=columns) as col:
        total = T.state((columns,), "float32", init=0)
        with T.level.blocks(K, extent=32) as part:
            weight = T.load(wv[col, part].q)
            activation = T.load(xv[part].q)
            wd = T.load(wv[col, part].d)
            xd = T.load(xv[part].d)
            products = T.alloc_local((columns,), "int32")
            T.reduce_dot(weight, activation, products, over="k", acc_dtype="int32", engine="vector")
            if sums is None:
                widened = T.alloc_local((32,), "int32")
                activation_sum = T.alloc_local((), "int32")
                for k in T.Parallel(32):
                    widened[k] = T.cast(activation[k], "int32")
                T.reduce_sum(widened, activation_sum, dim=0)
            else:
                activation_sum = T.load(sums[part].value)
            for c in T.Parallel(columns):
                centered = products[c] - 8 * T.cast(activation_sum[()], "int32")
                total[c] += T.cast(centered, "float32") * (T.cast(wd[c], "float32") * T.cast(xd[()], "float32"))
        T.copy(total, yv[col])



@tilelang.jit(target="weft")
def q4_0_decode(M, N, K, columns=64, reuse_sums=False):
    assert M == 1 and K % 32 == 0 and N % 16 == 0

    if not reuse_sums:
        @T.prim_func
        def main(
            W: T.Tensor((N * (K // 32) * 18,), "uint8"),
            Xq: T.Tensor((M, K // 32 * 34), "uint8"),
            Y: T.StridedTensor((N,), (1,), "float32"),
        ):
            T.func_attr({"tir.noalias": True})
            wv = T.view(W, shape=(N, K), encoding=L.AMXQ4_0Rows[16], axes=("n", "k"), abi=True)
            xv = T.view(Xq, shape=(K,), encoding=F.Q8_0, axes=("k",), abi=True)
            yv = T.view(Y, shape=(N,), axes=("n",), strides=(1,), abi=True)
            with T.Kernel(1, cpu_num_threads=1):
                q4_0_decode_tiles(wv, xv, yv, N, K, columns)

        return main

    @T.prim_func
    def main(
        W: T.Tensor((N * (K // 32) * 18,), "uint8"),
        Xq: T.Tensor((M, K // 32 * 34), "uint8"),
        Y: T.StridedTensor((N,), (1,), "float32"),
        Sums: T.Tensor((K // 32,), "int16"),
    ):
        T.func_attr({"tir.noalias": True})
        wv = T.view(W, shape=(N, K), encoding=L.AMXQ4_0Rows[16], axes=("n", "k"), abi=True)
        xv = T.view(Xq, shape=(K,), encoding=F.Q8_0, axes=("k",), abi=True)
        yv = T.view(Y, shape=(N,), axes=("n",), strides=(1,), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            sums = T.view(Sums, shape=(K,), encoding=L.BlockSum32, axes=("k",), abi=True)
            with T.level.blocks(K, extent=32) as part:
                activation = T.load(xv[part].q)
                widened = T.alloc_local((32,), "int16")
                total = T.alloc_local((), "int16")
                for k in T.Parallel(32):
                    widened[k] = T.cast(activation[k], "int16")
                T.reduce_sum(widened, total, dim=0)
                T.copy(total, sums[part].value)
            q4_0_decode_tiles(wv, xv, yv, N, K, columns, sums)

    return main


@T.macro
def q4_0_prefill_tiles(wv, xv, yv, M, N, K, rows, columns):
    with T.level.cols(N, group=columns) as col:
        with T.level.rows(M, group=rows) as row:
            total = T.state((rows, columns), "float32", init=0)
            with T.level.blocks(K, extent=32) as part:
                weight = T.load(wv[col, part].q)
                activation = T.load(xv[row, part].q)
                wd = T.load(wv[col, part].d)
                xd = T.load(xv[row, part].d)
                centered = T.alloc_local((columns, 32), "int8")
                products = T.alloc_local((rows, columns), "int32")
                for c, k in T.Parallel(columns, 32):
                    centered[c, k] = T.cast(weight[c, k], "int8") - T.int8(8)
                T.reduce_dot(activation, centered, products, over="k", acc_dtype="int32", engine="matrix")
                for r, c in T.Parallel(rows, columns):
                    total[r, c] += T.cast(products[r, c], "float32") * (T.cast(wd[c], "float32") * T.cast(xd[r], "float32"))
            T.copy(total, yv[row, col])



@tilelang.jit(target="weft")
def q4_0_prefill(M, N, K, LDC=None, rows=32, columns=32):
    LDC = N if LDC is None else LDC
    assert K % 32 == 0 and N % 16 == 0

    @T.prim_func
    def main(
        W: T.Tensor((N * (K // 32) * 18,), "uint8"),
        Xq: T.Tensor((M, K // 32 * 34), "uint8"),
        Y: T.StridedTensor((M, N), (LDC, 1), "float32"),
    ):
        T.func_attr({"tir.noalias": True})
        wv = T.view(W, shape=(N, K), encoding=L.AMXQ4_0Rows[16], axes=("n", "k"), abi=True)
        xv = T.view(Xq, shape=(M, K), encoding=F.Q8_0, axes=("m", "k"), abi=True)
        yv = T.view(Y, shape=(M, N), axes=("m", "n"), strides=(LDC, 1), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            q4_0_prefill_tiles(wv, xv, yv, M, N, K, rows, columns)

    return main


@tilelang.jit(target="weft")
def q4_0_projection(M, N, K, rows=4, columns=32, inverse_policy="direct"):
    """Prepare activations, then consume explicitly interleaved model weights."""
    assert M > 0 and N % 16 == 0 and K % 32 == 0

    @T.prim_func
    def main(
        Weights: T.Tensor((N * (K // 32) * 18,), "uint8"),
        Activation: T.Tensor((M, K), "float32"),
        Quantized: T.Tensor((M, K // 32 * 34), "uint8"),
        Output: T.Tensor((M, N), "float32"),
    ):
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Activation": "input", "Quantized": "workspace", "Output": "output"}})
        weights = T.view(Weights, shape=(N, K), encoding=L.AMXQ4_0Rows[16], axes=("n", "k"), abi=True)
        source = T.view(Activation, shape=(M, K), axes=("m", "k"), abi=True)
        prepared = T.view(Quantized, shape=(M, K), encoding=F.Q8_0, axes=("m", "k"), abi=True)
        output = T.view(Output, shape=(M, N), axes=("m", "n"), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            quantize_rows(source, prepared, M, K, "Q8_0", 32, "first", inverse_policy, "dynamic", 32)
            if M == 1:
                q4_0_decode_tiles(weights, prepared[0, :], output[0, :], N, K, columns)
            else:
                q4_0_prefill_tiles(weights, prepared, output, M, N, K, rows, columns)

    return main


@T.macro
def q6_k_scaled_tiles(weights, activation, output, M, N, K, rows, columns, reduction_window):
    """Scale signed six-bit values before bounded I16 x I8 contractions."""
    with T.level.cols(N, group=columns) as col:
        with T.level.rows(M, group=rows) as row:
            total = T.state((rows, columns), "float32", init=0)
            with T.level.blocks(K, extent=256) as part:
                weight, inputs = weights[col, part], activation[row, part]
                scales = T.load(weight.scales)
                products = T.alloc_local((rows, columns), "int32")
                T.fill(products, 0)
                for window in T.unroll(256 // reduction_window):
                    start = window * reduction_window
                    low = T.load(weight.low[:, start : start + reduction_window])
                    high = T.load(weight.high[:, start : start + reduction_window])
                    left = T.load(inputs.q[:, start : start + reduction_window])
                    right = T.alloc_local((columns, reduction_window), "int16")
                    partial = T.alloc_local((rows, columns), "int32")
                    for c, k in T.Parallel(columns, reduction_window):
                        code = T.cast(T.cast(low[c, k], "uint8") | (T.cast(high[c, k], "uint8") << T.uint8(4)), "int16") - T.int16(32)
                        group = T.cast(k, "uint32") // T.uint32(16) + T.uint32(window * reduction_window // 16)
                        right[c, k] = code * T.cast(scales[c, group], "int16")
                    T.reduce_dot(left, right, partial, over="k", acc_dtype="int32", engine="auto")
                    for r, c in T.Parallel(rows, columns):
                        products[r, c] += partial[r, c]
                wd, xd = T.load(weight.d), T.load(inputs.ds)
                for r, c in T.Parallel(rows, columns):
                    total[r, c] += T.cast(products[r, c], "float32") * (T.cast(wd[c], "float32") * xd[r])
            T.copy(total, output[row, col])


@tilelang.jit(target="weft")
def packed_x86(M, N, K, format="Q4_K", LDC=None, decode=False, rows=16, columns=32, reduction_window=64, groups_per_step=16):
    assert reduction_window in (32, 64, 128, 256)
    assert groups_per_step in (1, 2, 4, 8, 16)
    LDC = N if LDC is None else LDC
    block = 32 if format == "Q4_0" else 256
    record_bytes = {"Q4_0": 18, "Q4_K": 148, "Q6_K": 210}[format]
    weight_encoding = {"Q4_0": L.AMXQ4_0Rows[16], "Q4_K": L.AMXQ4_KRows[16], "Q6_K": L.AMXQ6_KRows[16]}[format]
    activation_encoding = F.Q8_0 if format == "Q4_0" else F.Q8_K
    activation_bytes = K // block * (34 if format == "Q4_0" else 292)
    rows = 1 if decode else rows

    @T.prim_func
    def main(
        W: T.Tensor((N * (K // block) * record_bytes,), "uint8"),
        Xq: T.Tensor((M, activation_bytes), "uint8"),
        Y: T.StridedTensor((M, N) if not decode else (N,), (LDC, 1) if not decode else (1,), "float32"),
    ):
        T.func_attr({"weft.alias_groups": {"W": "input", "Xq": "input", "Y": "output"}})
        wv = T.view(W, shape=(N, K), encoding=weight_encoding, axes=("n", "k"), abi=True)
        xv = T.view(Xq, shape=(M, K) if not decode else (K,), encoding=activation_encoding, axes=("m", "k") if not decode else ("k",), abi=True)
        yv = T.view(
            Y,
            shape=(M, N) if not decode else (N,),
            axes=("m", "n") if not decode else ("n",),
            strides=(LDC, 1) if not decode else (1,),
            abi=True,
        )
        with T.Kernel(1, cpu_num_threads=1):
            if format == "Q4_0":
                if decode:
                    q4_0_decode_tiles(wv, xv, yv, N, K, columns)
                else:
                    q4_0_prefill_tiles(wv, xv, yv, M, N, K, rows, columns)
            elif decode:
                k_layout_row(wv, xv, yv, N, K, format, columns, columns, groups_per_step)
            elif format == "Q6_K":
                q6_k_scaled_tiles(wv, xv, yv, M, N, K, rows, columns, reduction_window)
            else:
                with T.level.cols(N, group=columns) as col:
                    with T.level.rows(M, group=rows) as row:
                        total = T.state((rows, columns), "float32", init=0)
                        with T.level.blocks(K, extent=256) as part:
                            weight, activation = wv[col, part], xv[row, part]
                            scales = T.load(weight.scales)
                            scaled = T.alloc_local((rows, columns), "int32")
                            T.fill(scaled, 0)
                            for g in T.unroll(8):
                                left = T.load(activation.q[:, g * 32 : (g + 1) * 32])
                                low = T.load(weight.q[:, g * 32 : (g + 1) * 32])
                                right = T.alloc_local((columns, 32), "int8")
                                integer = T.alloc_local((rows, columns), "int32")
                                for c, k in T.Parallel(columns, 32):
                                    right[c, k] = T.cast(low[c, k], "int8")
                                T.reduce_dot(left, right, integer, over="k", acc_dtype="int32", engine="auto")
                                for r, c in T.Parallel(rows, columns):
                                    scaled[r, c] += integer[r, c] * T.cast(scales[c, g], "int32")
                            wd, xd = T.load(weight.d), T.load(activation.ds)
                            if format == "Q4_K":
                                sums = T.load(activation.bsum)
                                dmin, minima = T.load(weight.dmin), T.load(weight.mins)
                                pairs = T.alloc_local((rows, 8), "int16")
                                minimum = T.alloc_local((columns, 8), "int16")
                                correction = T.alloc_local((rows, columns), "int32")
                                paired_sums = T.sum_pairs(sums)
                                for r, g in T.Parallel(rows, 8):
                                    pairs[r, g] = T.cast(paired_sums[r, g], "int16")
                                for c, g in T.Parallel(columns, 8):
                                    minimum[c, g] = T.cast(minima[c, g], "int16")
                                T.reduce_dot(pairs, minimum, correction, over="k", acc_dtype="int32", engine="vector")
                            for r, c in T.Parallel(rows, columns):
                                total[r, c] += T.cast(scaled[r, c], "float32") * (T.cast(wd[c], "float32") * xd[r])
                                if format == "Q4_K":
                                    total[r, c] -= T.cast(correction[r, c], "float32") * (T.cast(dmin[c], "float32") * xd[r])
                        T.copy(total, yv[row, col])

    return main
