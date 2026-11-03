"""Compile dynamic packed rows and bind their exact GGML native roles."""

import argparse
from pathlib import Path

from tilelang.weft.kernels.deployment import vec_dot
from tilelang.weft.integrations.llama_cpp import create_package, install_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llama-source", type=Path, help="apply the SDK call-site integration to this external llama.cpp source")
    args = parser.parse_args()
    package = create_package(
        args.output,
        {"q4": vec_dot(format="Q4_K"), "q6": vec_dot(format="Q6_K")},
        bindings={"ggml.vec_dot": {"q4_K": "q4", "q6_K": "q6"}},
    )
    if args.llama_source:
        install_source(args.llama_source)
    print(package.path)


if __name__ == "__main__":
    main()
