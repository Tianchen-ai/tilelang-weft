/*!
 * \file weft/language/operations.cc
 * \brief Native TIRx operations retaining Weft source-model contracts.
 */
#include <tvm/ir/transform.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm::tl {
using tirx::CallEffectKind;
using tirx::TCallEffectKind;
using tirx::TScriptPrinterName;

#define WEFT_OP(Name, Effect)                                                   \
  TVM_REGISTER_OP("tl.weft." Name)                                             \
      .set_num_inputs(-1)                                                      \
      .set_attr<TScriptPrinterName>("TScriptPrinterName", "weft." Name)         \
      .set_attr<TCallEffectKind>("TCallEffectKind",                             \
                                 Integer(CallEffectKind::Effect))

WEFT_OP("view", kPure);
WEFT_OP("slice", kPure);
WEFT_OP("reshape", kPure);
WEFT_OP("extent", kPure);
WEFT_OP("field_view", kPure);
WEFT_OP("narrow", kPure);
WEFT_OP("load_record", kReadState);
WEFT_OP("load_value", kReadState);
WEFT_OP("field", kOpaque);
WEFT_OP("stage", kOpaque);
WEFT_OP("state", kOpaque);
WEFT_OP("mac_groups", kOpaque);
WEFT_OP("store_value", kOpaque);
WEFT_OP("helper_return", kOpaque);
WEFT_OP("helper_result", kPure);
WEFT_OP("helper_result_tile", kOpaque);
WEFT_OP("derive_argument", kPure);
WEFT_OP("derive_yield", kOpaque);
WEFT_OP("interleave", kPure);
WEFT_OP("atomic_rmw", kOpaque);
WEFT_OP("combine_argument", kPure);
WEFT_OP("combine_yield", kOpaque);
WEFT_OP("reduce_dot", kOpaque);
#undef WEFT_OP
}  // namespace tvm::tl
