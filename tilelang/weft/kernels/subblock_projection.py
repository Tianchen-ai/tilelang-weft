"""Invocation-local subblock decoding shared across activation rows."""

import tilelang
from tilelang.weft import language as T
from . import ggml_formats as F
from .preparation import quantize_rows


@T.macro(noinline=True)
def staged_tiles(weights, prepared, output, table, scales, M, N, K, format,
                 batch_rows, staged_columns, rows, columns, elements):
    """Retain each output tile across K within explicit encoded reuse scopes."""
    with T.level.tiles(N, extent=staged_columns, axis="n") as nc:
        weight_stage = T.stage(T.load(weights[nc, :]))
        with T.level.tiles(M, extent=batch_rows, axis="m") as mc:
            activation_stage = T.stage(T.load(prepared[mc, :]))
            with T.level.cols(nc, group=columns) as col:
                with T.level.rows(mc, group=rows) as row:
                    total = T.state((rows, columns), "float32", init=0)
                    with T.level.blocks(K, extent=elements) as block:
                        weight = weight_stage[col, block]
                        record_tiles(weight, activation_stage, table, scales, row, block,
                                     total, rows, columns, format)
                    T.copy(total, output[row, col])


@T.macro
def record_tiles(weight, activation_stage, table, scales, row, block, total, rows, columns, format):
    if format == "TQ2_0":
        w, x = T.load(weight), T.load(activation_stage[row, block])
        payload = w.q
        codes = T.alloc_local((columns, 256), "int8")
        for c, k in T.Parallel(columns, 256):
            codes[c, k] = T.cast(payload[c, k], "int8") - T.int8(1)
        products = T.alloc_local((rows, columns), "int32")
        T.reduce_dot(x.q, codes, products, over="k", acc_dtype="int32", engine="vector")
        for r, c in T.Parallel(rows, columns):
            total[r, c] += (T.cast(w.d[c], "float32") * x.ds[r]) * T.cast(products[r, c], "float32")
    elif format == "IQ4_XS":
        w, x = T.load(weight), T.load(activation_stage[row, block])
        low, high, wd, xd = w.scales_l, w.scales_h, w.d, x.ds
        group_index = T.alloc_var("uint16", init=0)
        with T.level.subtiles(block, extent=32) as sub:
            factors = T.alloc_local((columns,), "int32")
            for c in T.Parallel(columns):
                lower = (T.cast(low[c, group_index // T.uint16(2)], "uint16") >> (group_index % T.uint16(2) * T.uint16(4))) & T.uint16(15)
                upper = (high[c] >> (group_index * T.uint16(2))) & T.uint16(3)
                factors[c] = T.cast(lower | (upper << T.uint16(4)), "int32") - T.int32(32)
            codes = T.load(table, at=T.load(w.q, at=sub, axis=1), bounds="in_bounds")
            products = T.alloc_local((rows, columns), "int32")
            T.reduce_dot(T.load(x.q, at=sub, axis=1), codes, products,
                         over="k", acc_dtype="int32", engine="vector")
            for r, c in T.Parallel(rows, columns):
                total[r, c] += ((T.cast(wd[c], "float32") * xd[r]) * T.cast(factors[c], "float32")) * T.cast(products[r, c], "float32")
            group_index += T.uint16(1)
    else:
        with T.level.subtiles(block, extent=32) as activation_block:
            activation = activation_stage[row, activation_block]
            xd = T.load(activation.d)
            if format == "NVFP4":
                with T.level.subtiles(activation_block, extent=16) as sub:
                    codes = T.load(table, at=T.load(weight.q, at=sub, axis=1), bounds="in_bounds")
                    factors = T.load(scales, at=T.load(weight.d, at=sub, axis=1), bounds="in_bounds")
                    factors = T.reshape(factors, (columns,))
                    products = T.alloc_local((rows, columns), "int32")
                    T.reduce_dot(T.load(activation.q, at=sub, axis=1), codes, products,
                                 over="k", acc_dtype="int32", engine="vector")
                    for r, c in T.Parallel(rows, columns):
                        total[r, c] += (T.cast(xd[r], "float32") * factors[c]) * T.cast(products[r, c], "float32")
            else:
                bits = T.load(weight.q, at=activation_block, axis=1)
                wd = T.load(weight.d)
                codes = T.alloc_local((columns, 32), "int8")
                for c, k in T.Parallel(columns, 32):
                    codes[c, k] = T.cast(bits[c, k], "int8") * T.int8(2) - T.int8(1)
                products = T.alloc_local((rows, columns), "int32")
                T.reduce_dot(T.load(activation.q), codes, products,
                             over="k", acc_dtype="int32", engine="vector")
                for r, c in T.Parallel(rows, columns):
                    total[r, c] += (T.cast(wd[c], "float32") * T.cast(xd[r], "float32")) * T.cast(products[r, c], "float32")


@tilelang.jit(target="weft")
def projection(M, N, K, format="NVFP4", columns=8,
               extreme_policy="first", inverse_policy="direct", rows=1,
               batch_rows=None, staged_columns=None):
    assert format in ("Q1_0", "NVFP4", "TQ2_0", "IQ4_XS")
    encoding = getattr(F, format)
    elements = encoding.schema.elements
    batch_rows = M if batch_rows is None else batch_rows
    staged_columns = columns if staged_columns is None else staged_columns
    assert min(M, N, columns, rows, batch_rows, staged_columns) > 0
    assert M % batch_rows == N % staged_columns == 0
    assert batch_rows % rows == staged_columns % columns == 0
    assert K > 0 and K % elements == 0
    weight_bytes = K // elements * (encoding.schema.storage_bits // 8)
    superblock = format in ("TQ2_0", "IQ4_XS")
    lookup = format in ("NVFP4", "IQ4_XS")
    activation_elements = 256 if superblock else 32
    activation_bytes = 292 if superblock else 34
    activation_encoding = F.Q8_K if superblock else F.Q8_0

    @T.prim_func
    def main(
        Weights: T.Tensor((N, weight_bytes), "uint8"),
        Activation: T.Tensor((M, K), "float32"),
        Quantized: T.Tensor((M, K // activation_elements * activation_bytes), "uint8"),
        Output: T.Tensor((M, N), "float32"),
        Codebook: T.Tensor((16 if lookup else 0,), "int8"),
        Scale: T.Tensor((256 if format == "NVFP4" else 0,), "float32"),
    ):
        T.func_attr({"weft.alias_groups": {"Weights": "input", "Activation": "input",
            "Codebook": "input", "Scale": "input", "Quantized": "workspace", "Output": "output"}})
        weights = T.view(Weights, shape=(N, K), encoding=encoding, axes=("n", "k"), abi=True)
        source = T.view(Activation, shape=(M, K), axes=("m", "k"), abi=True)
        prepared = T.view(Quantized, shape=(M, K), encoding=activation_encoding, axes=("m", "k"), abi=True)
        output = T.view(Output, shape=(M, N), axes=("m", "n"), abi=True)
        if lookup:
            codebook = T.view(Codebook, shape=(16,), axes=("code",), abi=True)
        if format == "NVFP4":
            scales = T.view(Scale, shape=(256,), axes=("scale",), abi=True)
        with T.Kernel(1, cpu_num_threads=1):
            if lookup:
                table = T.stage(T.load(codebook))
            quantize_rows(source, prepared, M, K, "Q8_K" if superblock else "Q8_0", activation_elements,
                          extreme_policy, inverse_policy, "rne" if superblock else "dynamic", 32)
            if format == "NVFP4":
                staged_tiles(weights, prepared, output, table, scales, M, N, K, format,
                             batch_rows, staged_columns, rows, columns, elements)
            elif lookup:
                staged_tiles(weights, prepared, output, table, Scale, M, N, K, format,
                             batch_rows, staged_columns, rows, columns, elements)
            else:
                staged_tiles(weights, prepared, output, Codebook, Scale, M, N, K, format,
                             batch_rows, staged_columns, rows, columns, elements)

    return main
