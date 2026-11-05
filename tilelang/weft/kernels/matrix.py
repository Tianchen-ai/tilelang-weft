"""Packed tile programs; layouts and engine requirements belong to the author."""

import tilelang
from tilelang.weft import language as T
from . import layouts as L
from . import ggml_formats as F


@tilelang.jit(target="weft")
def ime(M, N, K, generation=1, affine=False, tile_rows=4, columns=16, engine="matrix"):
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
                        integer = T.alloc_local((tile_rows, columns), "int32")
                        if generation == 2 and affine:
                            xq = x.q
                            high = T.alloc_local((tile_rows, 32), "int4")
                            low = T.alloc_local((tile_rows, 32), "uint4")
                            high_product = T.alloc_local((tile_rows, columns), "int32")
                            low_product = T.alloc_local((tile_rows, columns), "int32")
                            negative = x.neg_sum
                            for r, k in T.Parallel(tile_rows, 32):
                                high[r, k] = T.cast(xq[r, k] >> 4, "int4")
                                low[r, k] = T.cast(xq[r, k], "uint4")
                            T.reduce_dot(high, q, high_product, over="k", acc_dtype="int32", engine=engine)
                            T.reduce_dot(low, q, low_product, over="k", acc_dtype="int32", engine=engine)
                            for r, c in T.Parallel(tile_rows, columns):
                                integer[r, c] = (
                                    high_product[r, c] * 16 + low_product[r, c] + T.cast(negative[r], "int32") * T.cast(zero[c], "int32")
                                )
                        elif generation == 1:
                            values = T.alloc_local((columns, 32), "int8")
                            for c, k in T.Parallel(columns, 32):
                                values[c, k] = T.cast(q[c, k], "int8") - T.cast(zero[c] if affine else 8, "int8")
                            T.gemm(x.q, values, integer, transpose_B=True, clear_accum=True, engine=engine)
                        else:
                            T.fill(integer, 0)
                            xq = x.q
                            with T.level.subtiles(part, extent=16) as sub:
                                left = T.load(xq, at=sub)
                                right = T.load(q, at=sub)
                                partial = T.alloc_local((tile_rows, columns), "int32")
                                T.reduce_dot(left, right, partial, over="k", acc_dtype="int32", engine=engine)
                                for r, c in T.Parallel(tile_rows, columns):
                                    integer[r, c] += partial[r, c]
                        for r, c in T.Parallel(tile_rows, columns):
                            total[r, c] += (T.cast(integer[r, c], "float32") * T.cast(wd[c], "float32")) * xd[r]
                    T.copy(total, yv[row, col])

    return main


@tilelang.jit(target="weft")
def q4_0_decode(M, N, K, columns=64):
    assert M == 1 and K % 32 == 0 and N % 16 == 0

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
            with T.level.cols(N, group=columns) as col:
                total = T.state((columns,), "float32", init=0)
                with T.level.blocks(K, extent=32) as part:
                    weight = T.load(wv[col, part].q)
                    activation = T.load(xv[part].q)
                    wd = T.load(wv[col, part].d)
                    xd = T.load(xv[part].d)
                    products = T.alloc_local((columns,), "int32")
                    widened = T.alloc_local((32,), "int32")
                    activation_sum = T.alloc_local((1,), "int32")
                    T.reduce_dot(weight, activation, products, over="k", acc_dtype="int32", engine="vector")
                    for k in T.Parallel(32):
                        widened[k] = T.cast(activation[k], "int32")
                    T.reduce_sum(widened, activation_sum, dim=0)
                    for c in T.Parallel(columns):
                        centered = products[c] - 8 * activation_sum[0]
                        total[c] += T.cast(centered, "float32") * (T.cast(wd[c], "float32") * T.cast(xd[()], "float32"))
                T.copy(total, yv[col])

    return main


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

    return main


@tilelang.jit(target="weft")
def packed_x86(M, N, K, format="Q4_K", LDC=None, decode=False, rows=16, columns=32):
    LDC = N if LDC is None else LDC
    block = 32 if format == "Q4_0" else 256
    group = 16 if format == "Q6_K" else 32
    record_bytes = {"Q4_0": 18, "Q4_K": 148, "Q6_K": 210}[format]
    weight = {"Q4_0": L.AMXQ4_0Rows[16], "Q4_K": L.AMXQ4_KRows[16], "Q6_K": L.AMXQ6_KRows[16]}[format]
    activation = F.Q8_0 if format == "Q4_0" else F.Q8_K
    activation_bytes = K // block * (34 if format == "Q4_0" else 292)
    rows = 1 if decode else rows
    engine = "vector" if decode else "matrix"

    @T.prim_func
    def main(
        W: T.Tensor((N * (K // block) * record_bytes,), "uint8"),
        Xq: T.Tensor((M, activation_bytes), "uint8"),
        Y: T.StridedTensor((M, N) if not decode else (N,), (LDC, 1) if not decode else (1,), "float32"),
    ):
        wv = T.view(W, shape=(N, K), encoding=weight, axes=("n", "k"), abi=True)
        xv = T.view(Xq, shape=(M, K) if not decode else (K,), encoding=activation, axes=("m", "k") if not decode else ("k",), abi=True)
        yv = T.view(
            Y,
            shape=(M, N) if not decode else (N,),
            axes=("m", "n") if not decode else ("n",),
            strides=(LDC, 1) if not decode else (1,),
            abi=True,
        )
        with T.Kernel(1, cpu_num_threads=1):
            with T.level.rows(M, group=rows) as row:
                with T.level.cols(N, group=columns) as col:
                    total = T.state((rows, columns), "float32", init=0)
                    with T.level.blocks(K, extent=block) as part:
                        w = T.stage(T.load(wv[col, part]))
                        x = T.stage(T.load(xv[part] if decode else xv[row, part]))
                        q = w.low if format == "Q6_K" else w.q
                        if format == "Q6_K":
                            high, scales = w.high, w.scales
                        if format == "Q4_K":
                            scales, minimum, dmin = w.scales, w.mins, w.dmin
                            sums = x.bsum
                        xd = x.d if format == "Q4_0" else x.ds
                        wd, xq = w.d, x.q
                        for g in T.serial(block // group):
                            left = T.alloc_local((rows, 64), "int8")
                            right = T.alloc_local((columns, 64), "int8")
                            integer = T.alloc_local((rows, columns), "int32")
                            for r, k in T.Parallel(rows, 64):
                                if k < group:
                                    left[r, k] = xq[g * group + k] if decode else xq[r, g * group + k]
                                else:
                                    left[r, k] = 0
                            for c, k in T.Parallel(columns, 64):
                                if k < group:
                                    if format == "Q6_K":
                                        right[c, k] = (
                                            T.cast(
                                                T.cast(q[c, g * group + k], "int32") | (T.cast(high[c, g * group + k], "int32") << 4),
                                                "int8",
                                            )
                                            - 32
                                        )
                                    elif format == "Q4_0":
                                        right[c, k] = T.cast(q[c, g * group + k], "int8") - 8
                                    else:
                                        right[c, k] = T.cast(q[c, g * group + k], "int8")
                                else:
                                    right[c, k] = 0
                            T.gemm(left, right, integer, transpose_B=True, clear_accum=True, engine=engine)
                            for r, c in T.Parallel(rows, columns):
                                scale = T.cast(wd[c], "float32")
                                if format != "Q4_0":
                                    scale = scale * T.cast(scales[c, g], "float32")
                                contribution = T.cast(integer[r, c], "float32") * scale
                                if format == "Q4_K":
                                    activation_sum = (
                                        T.cast(sums[2 * g], "int32") + T.cast(sums[2 * g + 1], "int32")
                                        if decode
                                        else T.cast(sums[r, 2 * g], "int32") + T.cast(sums[r, 2 * g + 1], "int32")
                                    )
                                    contribution -= (
                                        T.cast(dmin[c], "float32") * T.cast(minimum[c, g], "float32") * T.cast(activation_sum, "float32")
                                    )
                                total[r, c] += contribution * (xd[()] if decode else xd[r])
                    if decode:
                        result = T.alloc_local((columns,), "float32")
                        for c in T.Parallel(columns):
                            result[c] = total[0, c]
                        T.copy(result, yv[col])
                    else:
                        T.copy(total, yv[row, col])

    return main
