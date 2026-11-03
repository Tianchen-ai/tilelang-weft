// Compile after exporting example 06; only the native runtime is linked.
#include <weft/runtime.hpp>
#include "kernel.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    const std::string root = argv[1];
    std::ifstream stream(root + "/manifest.json");
    if (!stream) return 2;
    std::string manifest((std::istreambuf_iterator<char>(stream)), {});
    weft::context context;
    weft_error error{};
    weft_kernel *handle = nullptr;
    weft::check(weft_kernel_open_native(context.get(), (root + "/kernel.so").c_str(),
                                       manifest.c_str(), &handle, &error), error);
    std::unique_ptr<weft_kernel, decltype(&weft_kernel_release)> lifetime(handle, weft_kernel_release);
    weft::check(weft_kernel_check_thread(handle, &error), error);
    weft_native_entry entry;
    weft::check(weft_kernel_get_entry(handle, &entry, &error), error);
    auto invoke = reinterpret_cast<decltype(&tilelang_main_instance)>(entry);
    float input[16], output[16];
    for (int i = 0; i < 16; ++i) input[i] = float(i);
    // Both source configurations in example 06 cover the complete 16 elements.
    invoke(input, output, 2.0f, 1.0f);
    for (int i = 0; i < 16; ++i)
        if (std::fabs(output[i] - (2.0f * input[i] + 1.0f)) > 1e-6f) return 1;
    std::cout << "native C++ deployment: correct\n";
}
