"""Call a packed GGML-layout dot through the installed TileLang JIT."""

import numpy as np

from tilelang.weft.kernels.packed import vec_dot


def main():
    rows, k = 3, 64
    weight_record = np.dtype([("d", "<f2"), ("q", "u1", (16,))])
    activation_record = np.dtype([("d", "<f2"), ("q", "i1", (32,))])
    weights = np.zeros((rows, k // 32), dtype=weight_record)
    weights["d"] = np.arange(1, rows + 1, dtype=np.float16)[:, None] / 4
    weights["q"] = 0x99  # Q4_0: both nibbles are 9, hence centered code 1.
    activation = np.zeros(k // 32, dtype=activation_record)
    activation["d"] = 0.5
    activation["q"] = np.arange(-16, 16, dtype=np.int8)
    output = np.full(rows, np.nan, dtype=np.float32)
    i8 = np.empty(0, dtype=np.int8)
    f32 = np.empty(0, dtype=np.float32)
    u32 = np.empty(0, dtype=np.uint32)

    kernel = vec_dot(M=rows, K=k, format="Q4_0", activation="Q8_0", rows=1)
    kernel(weights.view(np.uint8).reshape(rows, -1), activation.view(np.uint8),
           output, i8, i8, i8, f32, f32, u32)
    np.testing.assert_allclose(output, -4 * np.arange(1, rows + 1), rtol=0, atol=0)
    print(output)


if __name__ == "__main__":
    main()
