import tilelang.testing
import tilelang
import tilelang.language as T
from itertools import product
import torch
import pytest


def test_weft_native_modules_own_independent_contexts():
    pytest.importorskip("weft_core")
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import asdict
    from weft_core import compiler
    from weft_core.cpu import target
    from weft_core.options import X86Options, WeftOptions
    from tilelang.weft.kernels.activation import affine

    source = affine(17, block=8)
    text = source.artifact.semantic_ir
    machine = target()
    options = X86Options() if machine.arch == "x86_64" else WeftOptions(matrix_extension=machine.matrix_extension)
    def compile_one(_index):
        module = compiler.parse_module(text)
        lowered, _ = compiler.lower(module, {**asdict(options), **asdict(machine)})
        del module
        return compiler.emit(lowered)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(compile_one, range(4)))
    assert all("llvm_ir" in result if machine.arch == "x86_64" else "intrinsic_c" in result for result in results)


def test_weft_snapshot_keeps_the_prewrite_value_with_alias_inputs():
    pytest.importorskip("weft_core")
    from tilelang.weft import language as W

    @tilelang.jit(target="weft")
    def capture(N):
        @W.prim_func
        def main(X: W.Tensor((N,), "int32"), Alias: W.Tensor((N,), "int32"), Output: W.Tensor((N,), "int32")):
            with W.Kernel(1):
                view = W.view(X, shape=(N,), axes=("k",))
                output = W.view(Output, shape=(N,), axes=("k",))
                snapshot = W.stage(W.load(view))
                for i in W.Parallel(N):
                    Alias[i] += 7
                W.copy(snapshot, output)

        return main

    source = torch.arange(17, dtype=torch.int32)
    previous = source.clone()
    output = torch.empty_like(source)
    capture(17)(source, source, output)
    torch.testing.assert_close(output, previous, atol=0, rtol=0)
    torch.testing.assert_close(source, previous + 7, atol=0, rtol=0)


def test_weft_stage_rejects_tile_writes_to_snapshot():
    pytest.importorskip("weft_core")
    from tilelang.weft import language as W

    @tilelang.jit(target="weft")
    def invalid():
        @W.prim_func
        def main(X: W.Tensor((8,), "int32")):
            with W.Kernel(1):
                snapshot = W.stage(W.load(X))
                W.fill(snapshot, 0)

        return main

    with pytest.raises(Exception, match="immutable snapshot"):
        invalid()


def test_weft_stage_rejects_descendant_rebinding():
    pytest.importorskip("weft_core")
    from tilelang.weft import language as W

    with pytest.raises(Exception, match="immutable snapshot and cannot be rebound"):

        @W.prim_func
        def invalid(X: W.Tensor((8,), "int32")):
            with W.Kernel(1):
                snapshot = W.stage(W.load(X))
                with W.level.blocks(8, extent=8) as block:
                    snapshot = W.load(X)


def test_weft_level_initializer_does_not_capture_body_value():
    pytest.importorskip("weft_core")
    from tilelang.weft import language as W

    with pytest.raises(Exception, match="initializer cannot capture 'body_value'"):

        @W.prim_func
        def invalid(X: W.Tensor((8,), "int32")):
            with W.Kernel(1):
                with W.level.blocks(8, extent=8) as block:
                    body_value = W.alloc_local((8,), "int32")
                    snapshot = W.stage(body_value)


def test_weft_tile_copy_masks_tail_without_an_explicit_element_if():
    pytest.importorskip("weft_core")
    from tilelang.weft import language as W

    @tilelang.jit(target="weft")
    def tiled(N):
        @W.prim_func
        def main(X: W.Tensor((N,), "float32"), Y: W.Tensor((N,), "float32")):
            with W.Kernel(W.ceildiv(N, 16)) as block:
                local = W.alloc_local((16,), "float32")
                W.copy(X[block * 16], local)
                for i in W.Parallel(16):
                    local[i] = local[i] * 2
                W.copy(local, Y[block * 16])

        return main

    storage = torch.full((37,), 123.0)
    output = storage[2:35]
    source = torch.arange(33, dtype=torch.float32)
    tiled(33)(source, output)
    torch.testing.assert_close(output, source * 2)
    torch.testing.assert_close(storage[:2], torch.full((2,), 123.0))
    torch.testing.assert_close(storage[35:], torch.full((2,), 123.0))


def test_weft_level_initializers_capture_incoming_values():
    pytest.importorskip("weft_core")
    from tilelang.weft import language as W

    @tilelang.jit(target="weft")
    def incoming():
        @W.prim_func
        def main(X: W.Tensor((8,), "int32"), Y: W.Tensor((8,), "int32")):
            with W.Kernel(1):
                local = W.load(X)
                with W.level.blocks(8, extent=8) as block:
                    snapshot = W.stage(local)
                    local = W.alloc_local((8,), "int32")
                    W.fill(local, 42)
                    W.copy(snapshot, Y)

        return main

    source, output = torch.arange(8, dtype=torch.int32), torch.empty(8, dtype=torch.int32)
    incoming()(source, output)
    torch.testing.assert_close(output, source, atol=0, rtol=0)


@pytest.mark.parametrize("reverse", [False, True])
def test_weft_native_tile_scan(reverse):
    pytest.importorskip("weft_core")
    from tilelang.weft import language as W

    @tilelang.jit(target="weft")
    def scan(N):
        @W.prim_func
        def main(X: W.Tensor((N,), "int32"), Y: W.Tensor((N,), "int32")):
            with W.Kernel(1):
                local = W.alloc_local((N,), "int32")
                W.copy(X, local)
                W.cumsum(local, dim=0, reverse=reverse)
                W.copy(local, Y)

        return main

    x = torch.arange(-8, 9, dtype=torch.int32)
    y = torch.empty_like(x)
    scan(17)(x, y)
    expected = x.flip(0).cumsum(0).flip(0) if reverse else x.cumsum(0)
    torch.testing.assert_close(y, expected.to(torch.int32), atol=0, rtol=0)


def test_weft_grid_atomic_returns_and_seq_cst_ordering():
    pytest.importorskip("weft_core")
    from tilelang.weft import language as W

    @tilelang.jit(target="weft")
    def histogram(N):
        @W.prim_func
        def main(X: W.Tensor((N,), "int32"), H: W.Tensor((4,), "int32"), Previous: W.Tensor((N,), "int32")):
            with W.Kernel(W.ceildiv(N, 16), cpu_num_threads=4) as block:
                for i in W.Parallel(16):
                    index = block * 16 + i
                    if index < N:
                        Previous[index] = W.atomic_add(H[X[index]], 1, memory_order="seq_cst", return_prev=True)

        return main

    x = torch.arange(65, dtype=torch.int32) % 4
    h, previous = torch.zeros(4, dtype=torch.int32), torch.full_like(x, -1)
    histogram(65)(x, h, previous)
    torch.testing.assert_close(h, torch.bincount(x.to(torch.int64), minlength=4).to(torch.int32), atol=0, rtol=0)
    for bucket in range(4):
        torch.testing.assert_close(previous[x == bucket].sort().values, torch.arange(int(h[bucket]), dtype=torch.int32), atol=0, rtol=0)


def test_weft_atomic_load_store_cas_and_bitwise():
    pytest.importorskip("weft_core")
    from tilelang.weft import language as W

    @tilelang.jit(target="weft")
    def operations():
        @W.prim_func
        def main(State: W.Tensor((1,), "int32"), Previous: W.Tensor((6,), "int32")):
            with W.Kernel(1):
                W.atomic_store(State[0], 7, memory_order="release")
                Previous[0] = W.atomic_load(State[0], memory_order="acquire")
                Previous[1] = W.atomic_cas(State[0], 7, 13)
                Previous[2] = W.atomic_cas(State[0], 7, 42)
                Previous[3] = W.atomic_and(State[0], 11)
                Previous[4] = W.atomic_xor(State[0], 3)
                Previous[5] = W.atomic_xchg(State[0], -1)

        return main

    state, previous = torch.zeros(1, dtype=torch.int32), torch.empty(6, dtype=torch.int32)
    operations()(state, previous)
    torch.testing.assert_close(previous, torch.tensor([7, 7, 13, 13, 9, 10], dtype=torch.int32), atol=0, rtol=0)
    assert state.item() == -1


def test_weft_autotune_restores_readwrite_alias_storage():
    pytest.importorskip("weft_core")
    from tilelang.weft import language as W

    source = torch.arange(17, dtype=torch.float32)
    initial = source.clone()

    def reference(x, alias):
        x.add_(alias)

    @tilelang.autotune(
        configs=[{"B": 8}, {"B": 16}],
        warmup=1,
        rep=2,
        ref_prog=reference,
        supply_prog=lambda params: [source, source],
        atol=0,
        rtol=0,
        max_mismatched_ratio=0,
    )
    @tilelang.jit(target="weft")
    def double_in_place(N, B=8):
        @W.prim_func
        def main(X: W.Tensor((N,), "float32"), Alias: W.Tensor((N,), "float32")):
            with W.Kernel(W.ceildiv(N, B)) as block:
                for i in W.Parallel(B):
                    index = block * B + i
                    if index < N:
                        X[index] = X[index] + Alias[index]

        return main

    kernel = double_in_place(17)
    torch.testing.assert_close(source, initial, atol=0, rtol=0)
    kernel(source, source)
    torch.testing.assert_close(source, initial * 2, atol=0, rtol=0)
    assert kernel.execution_backend == "weft_native"
    assert "tilelang" in kernel.get_kernel_source()


@pytest.mark.parametrize("noinline", [False, True])
def test_weft_macro_returns_through_condition_loop_and_level(noinline):
    pytest.importorskip("weft_core")
    from tilelang.weft import language as W

    @W.macro(noinline=noinline)
    def absolute(x):
        for k in W.serial(3):
            if x < 0:
                return -x, x
        with W.level.blocks(1, extent=1) as block:
            if x > 0:
                return x, -x
        return x, x

    @tilelang.jit(target="weft")
    def application(N):
        @W.prim_func
        def main(X: W.Tensor((N,), "int32"), Y: W.Tensor((N,), "int32")):
            with W.Kernel(1):
                for i in W.serial(N):
                    a, b = absolute(X[i])
                    Y[i] = a - b

        return main

    x = torch.arange(-4, 5, dtype=torch.int32)
    y = torch.empty_like(x)
    kernel = application(9)
    kernel(x, y)
    torch.testing.assert_close(y, x.abs() * 2, atol=0, rtol=0)
    assert "weft_kernel.helper_return" in kernel.artifact.semantic_ir
    if noinline:
        assert "noinline" in kernel.get_kernel_source()


def test_weft_macro_rejects_value_return_with_fallthrough():
    pytest.importorskip("weft_core")
    from tilelang.weft import language as W

    @W.macro
    def incomplete(x):
        if x > 0:
            return x

    with pytest.raises(Exception, match="return paths must preserve"):

        @W.prim_func
        def invalid(X: W.Tensor((1,), "int32"), Y: W.Tensor((1,), "int32")):
            with W.Kernel(1):
                Y[0] = incomplete(X[0])


def test_weft_macro_rejects_incompatible_named_axes():
    pytest.importorskip("weft_core")
    from tilelang.weft import language as W

    @W.macro
    def choose(condition, left, right):
        if condition:
            return left
        return right

    @tilelang.jit(target="weft")
    def invalid():
        @W.prim_func
        def main(X: W.Tensor((8,), "int32"), Y: W.Tensor((8,), "int32")):
            with W.Kernel(1):
                left = W.view(X, shape=(8,), axes=("m",))
                right = W.view(Y, shape=(8,), axes=("n",))
                result = choose(X[0] > 0, left, right)
                W.copy(W.load(result), Y)

        return main

    with pytest.raises(Exception, match="preserve Encoding and logical axis identity"):
        invalid()


@pytest.mark.parametrize("depth", [0, 1, 2, 3, 4])
def test_weft_loop_pipeline_uses_shared_expander(depth):
    pytest.importorskip("weft_core")
    from tilelang.weft import language as W

    @tilelang.jit(target="weft")
    def sum_blocks(N):
        @W.prim_func
        def main(X: W.Tensor((N,), "int32"), Y: W.Tensor((8,), "int32")):
            with W.Kernel(1):
                total = W.alloc_local((8,), "int32")
                W.fill(total, 0)
                for block in W.Pipelined(W.ceildiv(N, 8), num_stages=depth):
                    data = W.alloc_local((8,), "int32")
                    W.copy(X[block * 8], data)
                    for lane in W.Parallel(8):
                        total[lane] += data[lane]
                W.copy(total, Y)

        return main

    source, output = torch.arange(39, dtype=torch.int32), torch.empty(8, dtype=torch.int32)
    compiled = sum_blocks(39)
    compiled(source, output)
    expected = torch.nn.functional.pad(source, (0, 1)).reshape(5, 8).sum(0).to(torch.int32)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    assert f"weft.pipeline.depth = {depth}" in compiled.artifact.semantic_ir
    if depth > 1:
        assert f"weft.pipeline.expanded_depth = {depth}" in compiled.artifact.physical_ir
        from dataclasses import asdict
        from weft_core import compiler
        from weft_core.cpu import target
        from weft_core.options import WeftOptions

        for profile in ("sg2044", "k1", "k3-a100"):
            canonical = compiler.parse_module(compiled.artifact.semantic_ir)
            lowered, _ = compiler.lower(canonical, {**asdict(WeftOptions()), **asdict(target(profile))})
            assert compiler.emit(lowered)["intrinsic_c"]


@pytest.mark.parametrize("operation", ["silu", "softplus", "exp", "log", "copy"])
def test_weft_product_activations(operation):
    pytest.importorskip("weft_core")
    from tilelang.weft.kernels.activation import elementwise

    x = torch.linspace(0.01, 8, 37)
    y = torch.empty_like(x)
    elementwise(37, operation=operation, block=16, cpu_num_threads=2)(x, y)
    reference = {
        "silu": torch.nn.functional.silu,
        "softplus": torch.nn.functional.softplus,
        "exp": torch.exp,
        "log": torch.log,
        "copy": lambda x: x,
    }[operation](x)
    torch.testing.assert_close(y, reference)


@pytest.mark.parametrize("operation", ["rms", "layer", "residual", "softmax"])
def test_weft_product_normalization(operation):
    pytest.importorskip("weft_core")
    from tilelang.weft.kernels.normalization import rms_norm, layer_norm, residual_rms_norm
    from tilelang.weft.kernels.attention import softmax

    torch.manual_seed(21)
    x, weight, bias, residual = torch.randn(3, 17), torch.randn(17), torch.randn(17), torch.randn(3, 17)
    y, epsilon = torch.empty_like(x), 1e-5
    if operation == "rms":
        rms_norm(3, 17)(x, weight, y, epsilon)
        expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + epsilon) * weight
    elif operation == "layer":
        layer_norm(3, 17)(x, weight, bias, y, epsilon)
        expected = torch.nn.functional.layer_norm(x, (17,), weight, bias, epsilon)
    elif operation == "residual":
        residual_rms_norm(3, 17)(x, residual, weight, y, epsilon)
        value = x + residual
        expected = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + epsilon) * weight
    else:
        softmax(3, 17)(x, y)
        expected = x.softmax(-1)
    torch.testing.assert_close(y, expected)


def test_weft_product_gemm_tail():
    pytest.importorskip("weft_core")
    from tilelang.weft.kernels.linear import matmul

    torch.manual_seed(22)
    a, b, output = torch.randn(5, 37), torch.randn(37, 19), torch.empty(5, 19)
    matmul(5, 19, 37, block_m=4, block_n=16, block_k=16, engine="vector")(a, b, output)
    torch.testing.assert_close(output, a @ b)


def test_weft_matrix_engine_executes_native_extension_when_available():
    pytest.importorskip("weft_core")
    from weft_core import compiler
    from tilelang.weft.kernels.linear import matmul

    available = compiler.native_target().get("native_matrix_extensions", {})
    selected = next((name for name in ("amx-int8", "spacemit-ime1", "spacemit-ime2") if name in available), None)
    if selected is None:
        pytest.skip("native target does not provide an integer matrix extension")
    torch.manual_seed(22)
    m, n, k = (16, 16, 64) if selected == "amx-int8" else (4, 8, 32)
    a = torch.randint(-8, 8, (m, k), dtype=torch.int8)
    b = torch.randint(-8, 8, (k, n), dtype=torch.int8)
    output = torch.empty(m, n, dtype=torch.int32)
    kernel = matmul(m, n, k, block_m=m, block_n=n, block_k=k, dtype="int8", acc_dtype="int32", engine="matrix")
    kernel(a, b, output)
    torch.testing.assert_close(output, a.to(torch.int32) @ b.to(torch.int32), atol=0, rtol=0)
    assert selected in kernel.artifact.native_artifact.metadata.kernel_abi["used_extensions"]


@pytest.mark.parametrize("interleaved", [False, True])
def test_weft_product_rope(interleaved):
    pytest.importorskip("weft_core")
    from tilelang.weft.kernels.attention import rope

    torch.manual_seed(23)
    x, cosine, sine = torch.randn(3, 2, 34), torch.randn(3, 17), torch.randn(3, 17)
    output = torch.empty_like(x)
    rope(3, 2, 17, interleaved=interleaved)(x, cosine, sine, output)
    a, b = (x[..., ::2], x[..., 1::2]) if interleaved else x.chunk(2, -1)
    first = a * cosine[:, None, :] - b * sine[:, None, :]
    second = a * sine[:, None, :] + b * cosine[:, None, :]
    expected = torch.stack((first, second), -1).flatten(-2) if interleaved else torch.cat((first, second), -1)
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("format", ["Q4_K", "Q6_K"])
def test_weft_product_quantized_partials_and_scale_branches(format):
    pytest.importorskip("weft_core")
    import numpy as np
    import struct
    from tilelang.weft.kernels.quantized import q4_k_q8_k, q6_k_q8_k, dequantize

    rng = np.random.default_rng(17)
    m, blocks = 3, 2
    x = rng.integers(-128, 128, (blocks, 256), dtype=np.int16).astype(np.int8)
    xbytes = b"".join(struct.pack("<f", 0.125) + values.tobytes() + values.reshape(16, 16).sum(1).astype("<i2").tobytes() for values in x)
    expected = np.zeros(m, dtype=np.float32)
    rows = []
    decoded_rows = []
    for row in range(m):
        encoded = []
        decoded = []
        for block in range(blocks):
            if format == "Q4_K":
                q = rng.integers(0, 16, 256, dtype=np.uint8)
                scales, minimum = rng.integers(0, 64, (2, 8), dtype=np.uint8)
                joined = np.concatenate(
                    (
                        scales[:4] | ((scales[4:] >> 4) << 6),
                        minimum[:4] | ((minimum[4:] >> 4) << 6),
                        (scales[4:] & 15) | ((minimum[4:] & 15) << 4),
                    )
                )
                packed = (q.reshape(4, 64)[:, :32] | (q.reshape(4, 64)[:, 32:] << 4)).tobytes()
                encoded.append(struct.pack("<ee", 0.25, 0.125) + joined.tobytes() + packed)
                product = (q.astype(np.int32) * x[block].astype(np.int32)).reshape(8, 32).sum(1)
                correction = x[block].reshape(8, 32).sum(1)
                expected[row] -= np.float32(0.125 * 0.125 * (minimum.astype(np.int32) * correction).sum())
                expected[row] += np.float32(0.125 * 0.25 * (scales.astype(np.int32) * product).sum())
                decoded.append(0.25 * scales.repeat(32).astype(np.float32) * q - 0.125 * minimum.repeat(32).astype(np.float32))
            else:
                q = rng.integers(0, 64, 256, dtype=np.uint8)
                halves = q.reshape(2, 128)
                low = ((halves[:, :64] & 15) | ((halves[:, 64:] & 15) << 4)).tobytes()
                planes = (halves >> 4).reshape(2, 4, 32)
                high = (planes[:, 0] | (planes[:, 1] << 2) | (planes[:, 2] << 4) | (planes[:, 3] << 6)).tobytes()
                scales = rng.integers(-8, 8, 16, dtype=np.int8)
                encoded.append(low + high + scales.tobytes() + struct.pack("<e", 0.25))
                product = ((q.astype(np.int32) - 32) * x[block].astype(np.int32)).reshape(16, 16).sum(1)
                expected[row] += np.float32(0.25 * 0.125 * (product * scales.astype(np.int32)).sum())
                decoded.append(0.25 * scales.repeat(16).astype(np.float32) * (q.astype(np.int32) - 32))
        rows.append(b"".join(encoded))
        decoded_rows.append(np.concatenate(decoded))
    weights = torch.from_numpy(np.frombuffer(b"".join(rows), dtype=np.uint8).copy().reshape(m, -1))
    activation = torch.from_numpy(np.frombuffer(xbytes, dtype=np.uint8).copy())
    output = torch.empty(m, dtype=torch.float32)
    kernel = (q4_k_q8_k if format == "Q4_K" else q6_k_q8_k)(m, blocks * 256)
    kernel(weights, activation, output)
    torch.testing.assert_close(output, torch.from_numpy(expected))
    assert "weft_kernel.mac_groups" in kernel.artifact.semantic_ir
    assert "weft_kernel.materialize" in kernel.artifact.semantic_ir
    restored = torch.empty(m, blocks * 256)
    dequantize(m, blocks * 256, format=format)(weights, restored)
    torch.testing.assert_close(restored, torch.from_numpy(np.stack(decoded_rows)).to(torch.float32))
    from tilelang.weft.kernels.composition import packed_projection

    dense_input = torch.randn(2, blocks * 256)
    projected = torch.empty(2, m)
    composed = packed_projection(2, m, blocks * 256, format=format)
    composed(weights, dense_input, restored, projected)
    torch.testing.assert_close(projected, dense_input @ restored.T)
    from tilelang.weft.kernels.deployment import vec_dot
    from weft_core.integrations.llama_cpp import validate_vec_dot

    native_row = vec_dot(format=format)
    validate_vec_dot(native_row.adapter.native, format.lower().replace("_k", "_K"))
    row_output = torch.empty(1)
    for row in range(m):
        native_row(weights[row], activation, row_output, blocks * 256)
        torch.testing.assert_close(row_output[0], torch.tensor(expected[row]))
    assert native_row.adapter.native.metadata.single_instance_abi["shape_parameters"] == ["K"]


@pytest.mark.parametrize("format", ["Q1_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q8_1", "IQ4_NL"])
def test_weft_affine_and_lookup_dequantization(format):
    pytest.importorskip("weft_core")
    import numpy as np
    import struct
    from tilelang.weft.kernels.quantized import dequantize, dequantize_iq4_nl

    rng = np.random.default_rng(45)
    m, blocks, elements = 3, 2, 128 if format == "Q1_0" else 32
    codebook = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=np.int8)
    rows, expected = [], []
    for _row in range(m):
        records, values = [], []
        for _block in range(blocks):
            q = (
                rng.integers(-128, 128, elements, dtype=np.int16).astype(np.int8)
                if format.startswith("Q8")
                else rng.integers(0, 2 if format == "Q1_0" else 32 if format.startswith("Q5") else 16, elements, dtype=np.uint8)
            )
            prefix = struct.pack("<e", 0.25)
            if format in ("Q4_1", "Q5_1", "Q8_1"):
                prefix += struct.pack("<e", 0.125)
            if format.startswith("Q8"):
                payload, restored = q.tobytes(), q.astype(np.float32)
            elif format == "Q1_0":
                payload, restored = np.packbits(q, bitorder="little").tobytes(), q.astype(np.float32) * 2 - 1
            else:
                payload = ((q[:16] & 15) | ((q[16:] & 15) << 4)).tobytes()
                if format.startswith("Q5"):
                    payload = np.packbits(q >> 4, bitorder="little").tobytes() + payload
                restored = codebook[q].astype(np.float32) if format == "IQ4_NL" else q.astype(np.float32)
                if format in ("Q4_0", "Q5_0"):
                    restored -= 8 if format == "Q4_0" else 16
            records.append(prefix + payload)
            values.append(0.25 * restored + (0.125 if format in ("Q4_1", "Q5_1") else 0))
        rows.append(b"".join(records))
        expected.append(np.concatenate(values))
    storage = torch.from_numpy(np.frombuffer(b"".join(rows), np.uint8).copy().reshape(m, -1))
    output = torch.empty(m, blocks * elements)
    if format == "IQ4_NL":
        dequantize_iq4_nl(m, blocks * elements)(storage, torch.from_numpy(codebook), output)
    else:
        dequantize(m, blocks * elements, format=format)(storage, output)
    torch.testing.assert_close(output, torch.from_numpy(np.stack(expected)))


@pytest.mark.parametrize("prefix", [False, True])
def test_weft_tuple_argmax_combiner(prefix):
    pytest.importorskip("weft_core")
    from tilelang.weft import language as W

    @W.macro
    def maximum(left, left_index, right, right_index):
        choose_right = (right > left) | ((right == left) & (right_index < left_index))
        return W.if_then_else(choose_right, right, left), W.if_then_else(choose_right, right_index, left_index)

    @tilelang.jit(target="weft")
    def application(N):
        @W.prim_func
        def main(
            X: W.Tensor((N,), "float32"),
            Values: W.Tensor((N if prefix else 1,), "float32"),
            Indices: W.Tensor((N if prefix else 1,), "int32"),
        ):
            with W.Kernel(1):
                values = W.alloc_local((N,), "float32")
                indices = W.alloc_local((N,), "int32")
                output_values = W.alloc_local((N if prefix else 1,), "float32")
                output_indices = W.alloc_local((N if prefix else 1,), "int32")
                W.copy(X, values)
                for i in W.Parallel(N):
                    indices[i] = i
                if prefix:
                    W.scan((values, indices), (output_values, output_indices), combine=maximum)
                else:
                    W.reduce((values, indices), (output_values, output_indices), combine=maximum, dim=0)
                W.copy(output_values, Values)
                W.copy(output_indices, Indices)

        return main

    x = torch.tensor([1.0, 3.0, -1.0, 3.0, 2.0, 8.0, 7.0])
    values = torch.empty(7 if prefix else 1)
    indices = torch.empty_like(values, dtype=torch.int32)
    application(7)(x, values, indices)
    if prefix:
        expected_values = x.cummax(0).values
        expected_indices = torch.tensor([int(x[:end].argmax()) for end in range(1, 8)], dtype=torch.int32)
    else:
        expected_values, expected_indices = x.max().reshape(1), x.argmax().to(torch.int32).reshape(1)
    torch.testing.assert_close(values, expected_values)
    torch.testing.assert_close(indices, expected_indices, atol=0, rtol=0)


def test_weft_online_attention_multiple_state_handoffs():
    pytest.importorskip("weft_core")
    from tilelang.weft.kernels.attention import online_attention

    torch.manual_seed(37)
    q, k, v, y = torch.randn(3, 16), torch.randn(19, 16), torch.randn(19, 16), torch.empty(3, 16)
    online_attention(3, 19, 16, rows=2, keys=8, engine="vector")(q, k, v, y, 0.25)
    torch.testing.assert_close(y, (q @ k.T * 0.25).softmax(-1) @ v)


def test_weft_product_welford_and_topk():
    pytest.importorskip("weft_core")
    from tilelang.weft.kernels.state import welford, top_k

    torch.manual_seed(38)
    x, moments = torch.randn(3, 17), torch.empty(3, 3)
    welford(3, 17)(x, moments)
    expected = torch.stack((torch.full((3,), 17.0), x.mean(-1), x.var(-1, unbiased=False) * 17), -1)
    torch.testing.assert_close(moments, expected)
    values, indices = torch.empty(3, 4), torch.empty(3, 4, dtype=torch.int32)
    top_k(3, 17, 4)(x, values, indices)
    reference = x.topk(4)
    torch.testing.assert_close(values, reference.values)
    torch.testing.assert_close(indices, reference.indices.to(torch.int32), atol=0, rtol=0)


def test_weft_product_selective_scan_updates_caller_state():
    pytest.importorskip("weft_core")
    from tilelang.weft.kernels.state import selective_scan

    torch.manual_seed(39)
    b, l, d, n = 1, 3, 4, 7
    x, delta, gate = (torch.randn(b, l, d) for _ in range(3))
    bc, cc = torch.randn(b, l, n), torch.randn(b, l, n)
    alog, skip, bias, state = torch.randn(d, n), torch.randn(d), torch.randn(d), torch.randn(b, d, n)
    expected_state, expected_output, output = state.clone(), torch.empty_like(x), torch.empty_like(x)
    for token in range(l):
        step = torch.nn.functional.softplus(delta[:, token] + bias)
        expected_state = expected_state * torch.exp(-alog.exp() * step[..., None]) + bc[:, token, None, :] * (step * x[:, token])[..., None]
        expected_output[:, token] = ((expected_state * cc[:, token, None, :]).sum(-1) + skip * x[:, token]) * torch.nn.functional.silu(
            gate[:, token]
        )
    selective_scan(b, l, d, n)(x, delta, bc, cc, alog, skip, gate, bias, state, output)
    torch.testing.assert_close(state, expected_state)
    torch.testing.assert_close(output, expected_output)


@pytest.mark.parametrize("transpose", [False, True])
def test_weft_batched_projection_epilogue(transpose):
    pytest.importorskip("weft_core")
    from tilelang.weft.kernels.linear import projection

    torch.manual_seed(40)
    x, weight, bias = torch.randn(2, 3, 19), torch.randn(2, 5, 19), torch.randn(5)
    output = torch.empty(2, 3, 5)
    expected = torch.nn.functional.silu(torch.bmm(x, weight.transpose(1, 2)) + bias)
    operand = weight if transpose else weight.transpose(1, 2).contiguous()
    projection(2, 3, 5, 19, block_m=2, block_n=4, block_k=8, transpose_weight=transpose, activation="silu", engine="vector")(
        x, operand, bias, output
    )
    torch.testing.assert_close(output, expected)


def test_weft_moe_routing_grouping_empty_expert_and_combine():
    pytest.importorskip("weft_core")
    from tilelang.weft.kernels.moe import router, group_tokens, expert_linear, combine

    torch.manual_seed(41)
    tokens, experts, k, d, n = 3, 4, 2, 7, 5
    logits = torch.tensor([[3.0, 3.0, -2.0, -9.0], [2.0, 5.0, 1.0, -9.0], [4.0, 0.0, 3.0, -9.0]])
    indices, weights = torch.empty(tokens, k, dtype=torch.int32), torch.empty(tokens, k)
    router(tokens, experts, k)(logits, indices, weights)
    expected_indices = logits.argsort(dim=-1, descending=True, stable=True)[:, :k]
    torch.testing.assert_close(indices, expected_indices.to(torch.int32), atol=0, rtol=0)
    torch.testing.assert_close(weights, logits.gather(1, expected_indices).softmax(-1))
    x, w = torch.randn(tokens, d), torch.randn(experts, n, d)
    offsets, cursor = torch.empty(experts + 1, dtype=torch.int32), torch.empty(experts, dtype=torch.int32)
    positions, grouped = torch.empty(tokens, k, dtype=torch.int32), torch.empty(tokens * k, d)
    group_tokens(tokens, d, experts, k)(x, indices, offsets, cursor, positions, grouped)
    histogram = indices.flatten().bincount(minlength=experts).to(torch.int32)
    torch.testing.assert_close(offsets, torch.cat((torch.zeros(1, dtype=torch.int32), histogram.cumsum(0).to(torch.int32))), atol=0, rtol=0)
    torch.testing.assert_close(grouped[positions.long()], x[:, None, :].expand(-1, k, -1))
    projected, output = torch.empty(tokens * k, n), torch.empty(tokens, n)
    expert_linear(tokens * k, d, n, experts, block_n=4, block_k=4, engine="vector")(grouped, w, offsets, projected)
    combine(tokens, n, k)(projected, positions, weights, output)
    expected = torch.stack([sum((x[t] @ w[int(indices[t, s])].T) * weights[t, s] for s in range(k)) for t in range(tokens)])
    torch.testing.assert_close(output, expected)


def test_weft_causal_convolution_updates_history():
    pytest.importorskip("weft_core")
    from tilelang.weft.kernels.state import causal_conv1d

    torch.manual_seed(42)
    b, l, d, w = 2, 3, 7, 4
    x, weight, bias, history = torch.randn(b, l, d), torch.randn(d, w), torch.randn(d), torch.randn(b, w, d)
    expected_history, output = history.clone(), torch.empty_like(x)
    expected = torch.empty_like(x)
    for token in range(l):
        expected_history = torch.cat((expected_history[:, 1:], x[:, token : token + 1]), 1)
        expected[:, token] = torch.nn.functional.silu((expected_history * weight.T).sum(1) + bias)
    causal_conv1d(b, l, d, w)(x, weight, bias, history, output)
    torch.testing.assert_close(history, expected_history)
    torch.testing.assert_close(output, expected)


def test_weft_native_min_max_preserve_nan_and_signed_zero():
    pytest.importorskip("weft_core")
    from tilelang.weft import language as W

    @tilelang.jit(target="weft")
    def select(N):
        @W.prim_func
        def main(
            A: W.Tensor((N,), "float32"), B: W.Tensor((N,), "float32"), Min: W.Tensor((N,), "float32"), Max: W.Tensor((N,), "float32")
        ):
            with W.Kernel(1):
                for i in W.Parallel(N):
                    Min[i] = W.min(A[i], B[i])
                    Max[i] = W.max(A[i], B[i])

        return main

    a = torch.tensor([float("nan"), 2.0, -0.0, 0.0, -float("inf"), 4.0])
    b = torch.tensor([3.0, float("nan"), 0.0, -0.0, float("inf"), 4.0])
    lo, hi = torch.empty_like(a), torch.empty_like(a)
    select(6)(a, b, lo, hi)
    for actual, expected in ((lo, torch.where(a < b, a, b)), (hi, torch.where(a > b, a, b))):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        assert torch.equal(torch.signbit(actual[2:4]), torch.signbit(expected[2:4]))


def test_weft_parallel_scalar_temporaries_keep_independent_coordinates():
    pytest.importorskip("weft_core")
    from tilelang.weft import language as W

    @tilelang.jit(target="weft")
    def scalar_temporaries(M, N):
        @W.prim_func
        def main(Output: W.Tensor((M, N), "int32")):
            with W.Kernel(1):
                for row, column in W.Parallel(M, N):
                    digit = W.alloc_var("int32", init=0)
                    value = W.alloc_var("int32", init=1)
                    if column < 5:
                        digit = column // 2
                        value = row + 2
                    elif column < 11:
                        digit = column - 5
                        value = row - 3
                    else:
                        digit = column % 3
                        value = row + 7
                    Output[row, column] = digit * value

        return main

    output = torch.empty((3, 17), dtype=torch.int32)
    scalar_temporaries(3, 17)(output)
    row, column = torch.arange(3)[:, None], torch.arange(17)[None, :]
    expected = torch.where(column < 5, column // 2 * (row + 2), torch.where(column < 11, (column - 5) * (row - 3), column % 3 * (row + 7)))
    torch.testing.assert_close(output, expected.to(torch.int32), atol=0, rtol=0)


@pytest.mark.parametrize("format", ["Q8_0", "Q8_1", "Q8_K"])
def test_weft_quantization_keeps_packed_field_and_rounding_contract(format):
    pytest.importorskip("weft_core")
    import numpy as np
    import struct
    from tilelang.weft.kernels.preparation import quantize
    from tilelang.weft.kernels import ggml_formats as F

    schema = getattr(F, format).schema
    block = schema.elements
    x = torch.linspace(-1.25, 2.0, 2 * block).reshape(2, block)
    x[1] = 0
    output = torch.empty(2, schema.storage_bits // 8, dtype=torch.uint8)
    quantize(2, block, format=format)(x, output)
    for row in range(2):
        values = x[row].numpy()
        extreme = values[np.argmax(np.abs(values))]
        inv = (
            np.float32(0) if extreme == 0 else np.float32(-127) / extreme if format == "Q8_K" else np.float32(127) / np.max(np.abs(values))
        )
        d = np.float32(0) if inv == 0 else np.float32(1) / inv
        q = np.rint(values * inv).astype(np.int8)
        data = bytes(output[row].tolist())
        if format == "Q8_K":
            assert np.frombuffer(data[4:260], np.int8).tolist() == q.tolist()
            assert np.frombuffer(data[260:], "<i2").tolist() == q.reshape(16, 16).astype(np.int16).sum(1).tolist()
            assert struct.unpack("<f", data[:4])[0] == float(d)
        else:
            offset = 4 if format == "Q8_1" else 2
            assert np.frombuffer(data[offset:], np.int8).tolist() == q.tolist()
            assert struct.unpack("<e", data[:2])[0] == float(np.float16(d))
            if format == "Q8_1":
                assert struct.unpack("<e", data[2:4])[0] == float(np.float16(np.float32(q.astype(np.int16).sum()) * d))


def test_weft_address_observable_local_atomic_keeps_storage_identity():
    pytest.importorskip("weft_core")
    from tilelang.weft import language as T

    @tilelang.jit(target="weft")
    def factory():
        @T.prim_func
        def main(Output: T.Tensor((8,), "int32")):
            with T.Kernel(1):
                storage = T.alloc_local((1,), "int32")
                storage[0] = 0
                for i in T.serial(8):
                    previous = T.atomic_add(storage[0], 1, return_prev=True)
                    Output[i] = previous

        return main

    output = torch.empty(8, dtype=torch.int32)
    kernel = factory()
    kernel(output)
    torch.testing.assert_close(output, torch.arange(8, dtype=torch.int32), rtol=0, atol=0)
    assert "weft_kernel.local_alloc" in kernel.artifact.semantic_ir


def test_weft_causal_grouped_attention_and_kv_update():
    pytest.importorskip("weft_core")
    from tilelang.weft.kernels.attention import kv_update, causal_attention

    torch.manual_seed(47)
    q = torch.randn(1, 2, 4, 8)
    key, value = torch.randn(1, 5, 2, 8), torch.randn(1, 5, 2, 8)
    new_key, new_value = torch.randn(1, 2, 2, 8), torch.randn(1, 2, 2, 8)
    kv_update(1, 2, 2, 8, 5)(new_key, new_value, key, value, 2)
    torch.testing.assert_close(key[:, 2:4], new_key)
    torch.testing.assert_close(value[:, 2:4], new_value)
    output = torch.empty_like(q)
    causal_attention(1, 2, 4, 2, 8, 5)(q, key, value, output, 2, 0.5)
    expected = torch.empty_like(output)
    for token in range(2):
        for head in range(4):
            scores = (key[0, : 3 + token, head // 2] @ q[0, token, head]) * 0.5
            expected[0, token, head] = scores.softmax(0) @ value[0, : 3 + token, head // 2]
    torch.testing.assert_close(output, expected)


def test_weft_eager_runtime_extent_reuses_the_same_kernel():
    pytest.importorskip("weft_core")
    from tilelang.weft.kernels.activation import affine_eager

    for size in (17, 39):
        x = torch.linspace(-1, 1, size)
        output = torch.empty_like(x)
        affine_eager(x, output, 2.0, 0.25, block=16, cpu_num_threads=1)
        torch.testing.assert_close(output, x * 2 + 0.25)
    assert len(affine_eager._kernel_cache) == 1


def test_jit2_gemm():
    @tilelang.jit(verbose=True)
    def gemm(
        A,
        B,
        C,
        dtype: T.dtype = T.float16,
        accum_dtype: T.dtype = T.float32,
        block_M: int = 64,
        block_N: int = 64,
        block_K: int = 64,
    ):
        M, N, K = T.const("M N K")

        A: T.Tensor[[M, K], dtype]
        B: T.Tensor[[K, N], dtype]
        C: T.Tensor[[M, N], dtype]

        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N)) as (by, bx):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    A = torch.randn(1024, 1024, dtype=torch.float16, device="cuda")
    B = torch.randn(1024, 1024, dtype=torch.float16, device="cuda")
    C = torch.randn(1024, 1024, dtype=torch.float16, device="cuda")
    gemm(A, B, C)
    C_ref = A @ B
    torch.testing.assert_close(C, C_ref, atol=1e-2, rtol=1e-2)


@tilelang.testing.requires_cuda
def test_jit2_gemm_ptr():
    @tilelang.jit
    def gemm_ptr(
        A: T.ptr,
        B: T.ptr,
        C: T.ptr,
        M: int,
        N: int,
        K: int,
        dtype: T.dtype,
        out_dtype: T.dtype,
        block_M: int = 64,
        block_N: int = 64,
        block_K: int = 32,
    ):
        A = T.make_tensor(A, (M, K), dtype)
        B = T.make_tensor(B, (K, N), dtype)
        C = T.make_tensor(C, (M, N), out_dtype)
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), out_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[bx * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, by * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[bx * block_M, by * block_N])

    prod = product([T.float16, T.tfloat32], [T.float32])
    gemm_ptr.par_compile(
        [
            {"A": T.ptr(), "B": T.ptr(), "C": T.ptr(), "M": 1024, "N": 1024, "K": 1024, "dtype": in_dtype, "out_dtype": out_dtype}
            for in_dtype, out_dtype in prod
        ]
    )
    for in_dtype, out_dtype in prod:
        in_dtype = in_dtype.as_torch()
        out_dtype = out_dtype.as_torch()
        A = torch.randn(1024, 1024, dtype=in_dtype, device="cuda")
        B = torch.randn(1024, 1024, dtype=in_dtype, device="cuda")
        C_ref = out_dtype(A @ B)
        C = torch.empty(1024, 1024, dtype=out_dtype, device="cuda")
        gemm_ptr(A, B, C, 1024, 1024, 1024, in_dtype, out_dtype)
        torch.testing.assert_close(C, C_ref, atol=1e-2, rtol=1e-2)


def test_jit2_many_annot():
    @T.macro
    def copy_impl(A, B):
        M, N = A.shape
        M_, N_ = B.shape
        assert M == M_, f"M mismatch {M} {M_}"
        assert N == N_, f"N mismatch {N} {N_}"
        # assert tuple(A.shape) == tuple(B.shape), f"Invalid tensor shape: {A.shape}, {B.shape}"
        with T.Kernel(T.ceildiv(M, 128), T.ceildiv(N, 128), threads=128) as (bx, by):
            T.copy(A[bx * 128 : bx * 128 + 128, by * 128 : by * 128 + 128], B[bx * 128 : bx * 128 + 128, by * 128 : by * 128 + 128])

    @tilelang.jit
    def copy1(A, B):
        N, M = T.const("N, M")
        A: T.Tensor[[N, M], T.float32]
        B: T.Tensor[[N, M], T.float32]
        copy_impl(A, B)

    @tilelang.jit
    def copy2(
        A: T.Tensor[[128, 128], T.float32],
        B: T.Tensor[[128, 128], T.float32],
    ):
        copy_impl(A, B)

    @tilelang.jit
    def copy3(A, B):
        N = T.const("N")
        A: T.Tensor[[N, 128], T.float32]
        B: T.Tensor[[N, 128], T.float32]
        copy_impl(A, B)

    @tilelang.jit
    def copy4(A, B):
        N = T.dynamic("N")
        M = T.const("M")
        A: T.Tensor[[N, M], T.float32]
        B: T.Tensor[[N, M], T.float32]
        copy_impl(A, B)

    @tilelang.jit
    def copy5(A, B):
        N, M, N_, M_ = T.const("N, M, N_, M_")
        A: T.StridedTensor[[N, M], [N_, M_], T.float32]
        B: T.StridedTensor[[N, M], [N_, M_], T.float32]
        copy_impl(A, B)

    @tilelang.jit
    def copy6(A, B):
        N = T.dynamic("N")
        M, N_, M_ = T.const("M, N_, M_")
        A: T.StridedTensor[[N, M], [N_, M_], T.float32]
        B: T.StridedTensor[[N, M], [N_, M_], T.float32]
        copy_impl(A, B)

    tilelang.par_compile([copy.get_tir(T.Tensor((128, 128)), T.Tensor((128, 128))) for copy in [copy1, copy2, copy3, copy4]])

    for copy in [copy1, copy2, copy3, copy4]:
        A = torch.randn(128, 128, device="cuda")
        B = torch.empty(128, 128, device="cuda")
        copy(A, B)
        assert torch.equal(B, A)

    for copy in [copy5, copy6]:
        A = torch.randn(128, 2, 128, 2, device="cuda")
        B = torch.randn(128, 2, 128, 2, device="cuda")
        copy(A[:, 0, :, 0], B[:, 0, :, 0])
        assert torch.equal(A[:, 0, :, 0], B[:, 0, :, 0])


def test_jit2_return():
    @T.macro
    def copy_impl(A):
        M, N = A.shape
        B = T.empty(M, N, dtype=A.dtype)
        with T.Kernel(T.ceildiv(M, 128), T.ceildiv(N, 128), threads=128) as (bx, by):
            T.copy(A[bx * 128 : bx * 128 + 128, by * 128 : by * 128 + 128], B[bx * 128 : bx * 128 + 128, by * 128 : by * 128 + 128])
        return B

    @tilelang.jit
    def copy1(A):
        M, N = T.const("M, N")
        A: T.Tensor[[M, N], T.float32]
        return copy_impl(A)

    @tilelang.jit
    def copy2(A):
        A: T.Tensor[[128, 128], T.float32]
        return copy_impl(A)

    @tilelang.jit
    def copy3(A):
        N = T.const("N")
        A: T.Tensor[[N, 128], T.float32]
        return copy_impl(A)

    @tilelang.jit
    def copy4(A):
        N = T.dynamic("N")
        M = T.const("M")
        A: T.Tensor[[N, M], T.float32]
        return copy_impl(A)

    @tilelang.jit
    def copy5(A):
        N, M, N_, M_ = T.const("N, M, N_, M_")
        A: T.StridedTensor[[N, M], [N_, M_], T.float32]
        return copy_impl(A)

    @tilelang.jit
    def copy6(A):
        N = T.dynamic("N")
        M, N_, M_ = T.const("M, N_, M_")
        A: T.StridedTensor[[N, M], [N_, M_], T.float32]
        return copy_impl(A)

    for copy in [copy1, copy2, copy3, copy4]:
        A = torch.randn(128, 128, device="cuda")
        B = copy(A)
        assert torch.equal(B, A)

    for copy in [copy5, copy6]:
        A = torch.randn(128, 2, 128, 2, device="cuda")
        B = copy(A[:, 0, :, 0])
        assert torch.equal(A[:, 0, :, 0], B)


def test_jit2_compile_with_consts():
    @tilelang.jit
    def transpose(X, Y, block_M, block_N):
        M, N = T.const("M N")
        X: T.Tensor[[M, N], T.float32]
        Y: T.Tensor[[N, M], T.float32]

        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=128) as (bx, by):
            X_tile = T.alloc_shared((block_M, block_N), T.float32)
            Y_tile = T.alloc_shared((block_N, block_M), T.float32)

            T.copy(X[bx * block_M, by * block_N], X_tile)
            for i, j in T.Parallel(block_M, block_N):
                Y_tile[j, i] = X_tile[i, j]
            T.copy(Y_tile, Y[by * block_N, bx * block_M])

    transpose.compile(M=1024, N=1024, block_M=64, block_N=64)


def _kernel_body_is_skipped_in_phase1(func) -> bool:
    """Whether the eager rewriter guards the launch body with skip_kernel_ctx,
    which is what keeps phase-1 signature inference from executing it."""
    from tilelang.language.eager.ast import mutate

    return "skip_kernel_ctx" in mutate(func).source


def test_jit2_recognizes_launch_from_every_dialect():
    from tilelang.cpu import language as Tcpu
    from tilelang.cuda import language as Tcuda
    from tilelang.rocm import language as Trocm

    Launch = T.Kernel

    def default_facade(A):
        with T.Kernel(1):
            pass

    def cuda_dialect(A):
        with Tcuda.Kernel(1, threads=128):
            pass

    def rocm_dialect(A):
        with Trocm.Kernel(1, threads=64):
            pass

    def cpu_dialect(A):
        with Tcpu.Kernel(1):
            pass

    def cluster(A):
        with T.ClusterKernel(2, cluster_dims=2):
            pass

    def aliased(A):
        with Launch(1):
            pass

    def not_a_launch(A):
        with T.ws(0):
            pass

    for func in (default_facade, cuda_dialect, rocm_dialect, cpu_dialect, cluster, aliased):
        assert _kernel_body_is_skipped_in_phase1(func), func.__name__
    assert not _kernel_body_is_skipped_in_phase1(not_a_launch)


@tilelang.testing.requires_cuda
def test_jit2_phase1_does_not_execute_kernel_body():
    """Phase 1 infers the signature with symbolic T.const values, so the launch
    body must not run then: a gemm tile that depends on a const is only valid
    once the values are bound in phase 2."""

    @tilelang.jit
    def gemm_full_n(A, B, block_M, block_K):
        M, N, K = T.const("M, N, K")
        A: T.Tensor[[M, K], T.float16]
        B: T.Tensor[[K, N], T.float16]
        C = T.empty((M, N), T.float16)
        with T.Kernel(T.ceildiv(M, block_M), threads=128) as bx:
            A_shared = T.alloc_shared((block_M, block_K), T.float16)
            B_shared = T.alloc_shared((block_K, N), T.float16)
            C_local = T.alloc_fragment((block_M, N), T.float32)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[bx * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, 0], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[bx * block_M, 0])
        return C

    a = torch.randn(256, 128, device="cuda", dtype=torch.float16)
    b = torch.randn(128, 64, device="cuda", dtype=torch.float16)
    c = gemm_full_n(a, b, 64, 32)
    torch.testing.assert_close(c, (a.float() @ b.float()).half(), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    tilelang.testing.main()
