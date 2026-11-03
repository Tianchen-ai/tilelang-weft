import tilelang.testing
import tilelang
import tilelang.language as T
from itertools import product
import torch
import pytest


def test_weft_snapshot_keeps_the_prewrite_value_with_alias_inputs():
    pytest.importorskip("weft_core")
    from tilelang.weft import language as W

    @tilelang.jit(target="weft")
    def capture(N):
        @W.prim_func
        def main(X: W.Tensor((N,), "int32"), Alias: W.Tensor((N,), "int32"),
                 Output: W.Tensor((N,), "int32")):
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
        def main(X: W.Tensor((N,), "int32"), H: W.Tensor((4,), "int32"),
                 Previous: W.Tensor((N,), "int32")):
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


def test_weft_autotune_restores_readwrite_alias_storage():
    pytest.importorskip("weft_core")
    from tilelang.weft import language as W

    source = torch.arange(17, dtype=torch.float32)
    initial = source.clone()

    def reference(x, alias):
        x.add_(alias)

    @tilelang.autotune(configs=[{"B": 8}, {"B": 16}], warmup=1, rep=2,
                       ref_prog=reference, supply_prog=lambda params: [source, source],
                       atol=0, rtol=0, max_mismatched_ratio=0)
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
