// Link the SDK adapter and an external GGML build. Kernels are loaded from the package.
#include <weft/ggml.h>
#include "ggml-quants.h"
#include "quants.h"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    ggml_cpu_init();
    constexpr int k = 512;
    std::vector<float> weights(k), activation(k);
    for (int i = 0; i < k; ++i) {
        weights[i] = std::sin(float(i) * 0.13f);
        activation[i] = std::cos(float(i) * 0.21f);
    }
    block_q8_K x[k / QK_K];
    block_q4_K q4[k / QK_K];
    block_q6_K q6[k / QK_K];
    quantize_row_q8_K_ref(activation.data(), x, k);
    quantize_row_q4_K_ref(weights.data(), q4, k);
    quantize_row_q6_K_ref(weights.data(), q6, k);
    ggml_weft_initialize();
    auto *context = ggml_init({1024 * 1024, nullptr, false});
    if (!context) return 2;
    for (auto type : {GGML_TYPE_Q4_K, GGML_TYPE_Q6_K}) {
        const auto original = type == GGML_TYPE_Q4_K ? ggml_vec_dot_q4_K_q8_K : ggml_vec_dot_q6_K_q8_K;
        const void *w = type == GGML_TYPE_Q4_K ? static_cast<void *>(q4) : static_cast<void *>(q6);
        float reference = 0, actual = 0;
        original(k, &reference, 0, w, 0, x, 0, 1);
        auto *matrix = ggml_new_tensor_2d(context, type, k, 1);
        auto *input = ggml_new_tensor_2d(context, GGML_TYPE_F32, k, 1);
        auto *node = ggml_mul_mat(context, matrix, input);
        ggml_weft_begin_node(node, 0, false);
        const auto binding = ggml_weft_select_vec_dot(type, original);
        if (!binding.function) return 2;
        ggml_weft_call_dot(binding, k, &actual, 0, w, 0, x, 0, 1);
        ggml_weft_end_node();
        std::cout << ggml_type_name(type) << ": GGML=" << reference << ", TileLang=" << actual << '\n';
        if (std::abs(actual - reference) > 1e-4f + 1e-4f * std::abs(reference)) return 1;
        ggml_weft_counters counters{};
        if (!ggml_weft_counters_get(type, &counters) || !counters.vec_dot_calls) return 3;
        std::cout << ggml_type_name(type) << ": correct, native replacement calls=" << counters.vec_dot_calls << '\n';
    }
    ggml_free(context);
}
