/*!
 * \file weft/target.cc
 * \brief CPU target descriptor for the shared Weft compiler.
 */
#include <tvm/target/target.h>
#include <tvm/target/target_kind.h>
#include <tvm/ir/transform.h>

namespace tvm {
TVM_REGISTER_PASS_CONFIG_OPTION("tl.weft.backend_options", DictAttrs);

TVM_REGISTER_TARGET_KIND("weft", kDLCPU)
    .add_attr_option<ffi::String>("arch")
    .add_attr_option<ffi::String>("march")
    .add_attr_option<ffi::String>("abi")
    .add_attr_option<int64_t>("vlen_bits")
    .add_attr_option<ffi::String>("matrix_extension")
    .add_attr_option<ffi::String>("toolchain")
    .add_attr_option<ffi::String>("options")
    .add_attr_option<ffi::String>("core")
    .set_default_keys({"cpu"});
}  // namespace tvm
