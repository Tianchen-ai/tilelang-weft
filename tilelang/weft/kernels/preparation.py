"""Explicit caller-owned activation preparation with declared packed fields."""

import tilelang
from tilelang.weft import language as T
from . import ggml_formats as F


@T.macro
def quantize_k_record(value, source, output, part, extreme_policy, chunk):
    extreme = T.alloc_var("float32", init=0)
    positive = T.alloc_var("float32", init=0)
    negative = T.alloc_var("float32", init=0)
    maximum = T.alloc_local((1,), "float32")
    minimum = T.alloc_local((1,), "float32")
    T.reduce_max(value, maximum, dim=0)
    T.reduce_min(value, minimum, dim=0)
    positive = maximum[0]
    negative = minimum[0]
    extreme = T.if_then_else(T.abs(positive) > T.abs(negative), positive, negative)
    # Different magnitudes already determine the sign. Only an equal pair of
    # opposite extrema needs source positions to retain the first occurrence.
    if extreme_policy == "first" and positive > 0 and positive == -negative:
        candidates = T.alloc_local((256,), "int32")
        first = T.alloc_local((1,), "int32")
        for i in T.Parallel(256):
            candidates[i] = T.if_then_else(T.abs(value[i]) == positive, i, 256)
        T.reduce_min(candidates, first, dim=0)
        tied = T.load(source[part][first[0]])
        extreme = tied[()]
    inverse = T.alloc_var("float32", init=0)
    d = T.alloc_var("float32", init=0)
    if extreme != 0:
        inverse = -127.0 / extreme
        d = 1.0 / inverse
    if chunk == 256:
        q = T.alloc_local((256,), "int8")
        wide = T.alloc_local((16, 16), "int16")
        totals = T.alloc_local((16,), "int16")
        for i in T.Parallel(256):
            q[i] = T.narrow(value[i] * inverse, "int8", rounding="rne", saturation=True)
        grouped = T.reshape(q, (16, 16))
        for group, lane in T.Parallel(16, 16):
            wide[group, lane] = T.cast(grouped[group, lane], "int16")
        T.reduce_sum(wide, totals, dim=1)
        T.copy(q, output.q)
        T.copy(totals, output.bsum)
    else:
        with T.level.subtiles(part, extent=chunk) as window:
            values = T.load(source[window])
            q = T.alloc_local((chunk,), "int8")
            for i in T.Parallel(chunk):
                q[i] = T.narrow(values[i] * inverse, "int8", rounding="rne", saturation=True)
            T.copy(q, output.q[window])
            with T.level.subtiles(window, extent=16) as group:
                selected = T.load(q, at=group)
                wide = T.alloc_local((16,), "int16")
                total = T.alloc_local((), "int16")
                for i in T.Parallel(16):
                    wide[i] = T.cast(selected[i], "int16")
                T.reduce_sum(wide, total, dim=0)
                T.copy(total, output.bsum[group])
    T.copy(d, output.ds)


@T.macro
def quantize_rows(xv, yv, M, K, format, elements, extreme_policy, inverse_policy, rounding, chunk=64):
    for row in T.serial(M):
        with T.level.blocks(K, extent=elements) as part:
            value = T.load(xv[row, part])
            if format == "Q8_K":
                quantize_k_record(value, xv[row, :], yv[row, part], part, extreme_policy, chunk)
            else:
                absolute = T.alloc_local((elements,), "float32")
                maximum = T.alloc_local((1,), "float32")
                d = T.alloc_local((1,), "float32")
                inverse = T.alloc_var("float32", init=0)
                q = T.alloc_local((elements,), "int8")
                for i in T.Parallel(elements):
                    absolute[i] = T.abs(value[i])
                T.reduce_max(absolute, maximum, dim=0)
                d[0] = maximum[0] / 127.0
                if maximum[0] != 0:
                    inverse = 127.0 / maximum[0] if inverse_policy == "direct" else 1.0 / d[0]
                for i in T.Parallel(elements):
                    q[i] = T.narrow(value[i] * inverse, "int8", rounding=rounding, saturation=False)
                T.copy(q, yv[row, part].q)
                T.copy(d, yv[row, part].d)
                if format == "Q8_1":
                    wide = T.alloc_local((elements,), "int16")
                    total = T.alloc_local((1,), "int16")
                    scaled = T.alloc_local((1,), "float32")
                    for i in T.Parallel(elements):
                        wide[i] = T.cast(q[i], "int16")
                    T.reduce_sum(wide, total, dim=0)
                    scaled[0] = T.cast(total[0], "float32") * d[0]
                    T.copy(scaled, yv[row, part].s)


@tilelang.jit(target="weft")
def quantize(M, K, format="Q8_0", extreme_policy="first", inverse_policy="reciprocal", rounding="dynamic", chunk=64, records=1):
    assert format in ("Q8_0", "Q8_1", "Q8_K")
    assert chunk in (32, 64, 128, 256)
    assert records in (1, 2, 4, 8) and (records == 1 or format != "Q8_K")
    encoding = getattr(F, format)
    elements = encoding.schema.elements
    record_bytes = encoding.schema.storage_bits // 8
    assert K > 0 and K % (elements * records) == 0

    @T.prim_func
    def main(X: T.Tensor((M, K), "float32"), Output: T.Tensor((M, K // elements * record_bytes), "uint8")):
        T.func_attr({"tir.noalias": True})
        if records == 1:
            xv = T.view(X, shape=(M, K), axes=("m", "k"), abi=True)
            yv = T.view(Output, shape=(M, K), encoding=encoding, axes=("m", "k"), abi=True)
        else:
            xv = T.view(X, shape=(M, K // elements, elements), axes=("m", "record", "k"))
            yv = T.view(Output, shape=(M, K // elements, elements), encoding=encoding, axes=("m", "record", "k"))
        with T.Kernel(1):
            if records == 1:
                quantize_rows(xv, yv, M, K, format, elements, extreme_policy, inverse_policy, rounding, chunk)
            else:
                for row in T.serial(M):
                    with T.level.rows(K // elements, group=records, axis="record") as cohort:
                        values = T.load(xv[row, cohort, :])
                        absolute = T.alloc_local((records, elements), "float32")
                        maximum = T.alloc_local((records,), "float32")
                        scale = T.alloc_local((records,), "float32")
                        inverse = T.alloc_local((records,), "float32")
                        q = T.alloc_local((records, elements), "int8")
                        for record, lane in T.Parallel(records, elements):
                            absolute[record, lane] = T.abs(values[record, lane])
                        T.reduce_max(absolute, maximum, dim=1)
                        for record in T.Parallel(records):
                            scale[record] = maximum[record] / 127.0
                            denominator = maximum[record] if inverse_policy == "direct" else scale[record]
                            safe = T.if_then_else(denominator != 0, denominator, 1.0)
                            inverse[record] = T.if_then_else(denominator != 0, (127.0 if inverse_policy == "direct" else 1.0) / safe, 0.0)
                        for record, lane in T.Parallel(records, elements):
                            q[record, lane] = T.narrow(values[record, lane] * inverse[record], "int8", rounding=rounding, saturation=False)
                        for record in T.unroll(records):
                            T.copy(q[record, :], yv[row, cohort, :][record, :].q)
                            T.copy(scale[record], yv[row, cohort, :][record, :].d)
                        if format == "Q8_1":
                            wide = T.alloc_local((records, elements), "int32")
                            total = T.alloc_local((records,), "int32")
                            scaled = T.alloc_local((records,), "float32")
                            for record, lane in T.Parallel(records, elements):
                                wide[record, lane] = T.cast(q[record, lane], "int32")
                            T.reduce_sum(wide, total, dim=1)
                            for record in T.Parallel(records):
                                scaled[record] = T.cast(total[record], "float32") * scale[record]
                            for record in T.unroll(records):
                                T.copy(scaled[record], yv[row, cohort, :][record, :].s)

    return main
