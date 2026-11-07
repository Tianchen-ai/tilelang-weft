"""Normal TileLang programs; names select author source, never backend rules."""

from dataclasses import dataclass
from types import MappingProxyType
from . import activation, attention, linear, normalization, moe, state, quantized, codebook, preparation, deployment
from . import packed
from . import projection


@dataclass(frozen=True)
class Program:
    name: str
    family: str
    definition: object
    defaults: tuple = ()

    def compile(self, **source_parameters):
        return self.definition(**{**dict(self.defaults), **source_parameters})


_entries = [
    Program("rms_norm", "normalization", normalization.rms_norm),
    Program("layer_norm", "normalization", normalization.layer_norm),
    Program("residual_rms_norm", "normalization", normalization.residual_rms_norm),
    *(
        Program(name, "activation", activation.elementwise, (("operation", operation),))
        for name, operation in (("silu", "silu"), ("softplus", "softplus"), ("exponential", "exp"), ("logarithm", "log"), ("copy", "copy"))
    ),
    Program("swiglu", "activation", activation.swiglu),
    Program("add", "activation", activation.binary),
    Program("where", "selection", activation.where),
    Program("fill", "memory", activation.fill),
    Program("matmul", "projection", linear.matmul),
    Program("blocked_matmul", "projection", linear.blocked_matmul),
    Program("staged_f16_projection", "projection", projection.staged_f16_projection),
    Program("batched_matmul", "projection", linear.projection, (("transpose_weight", False), ("bias", False))),
    *(
        Program(name, "projection", linear.projection, (("transpose_weight", transpose), ("bias", bias), ("activation", act)))
        for name, transpose, bias, act in (
            ("linear", True, False, "none"),
            ("matmul_bias", False, True, "none"),
            ("linear_bias", True, True, "none"),
            ("linear_silu", True, False, "silu"),
            ("linear_bias_silu", True, True, "silu"),
        )
    ),
    Program("softmax", "attention", attention.softmax),
    Program("rope_split", "attention", attention.rope),
    Program("rope_interleaved", "attention", attention.rope, (("interleaved", True),)),
    Program("kv_update", "attention", attention.kv_update),
    Program("causal_attention", "attention", attention.causal_attention),
    Program("online_attention", "attention", attention.online_attention),
    Program("router", "moe", moe.router),
    Program("group_tokens", "moe", moe.group_tokens),
    Program("expert_linear", "moe", moe.expert_linear),
    Program("combine", "moe", moe.combine),
    Program("causal_conv1d", "mamba", state.causal_conv1d),
    Program("selective_scan", "mamba", state.selective_scan),
    Program("welford", "state", state.welford),
    Program("top_k", "state", state.top_k),
    Program("q4_k_q8_k", "packed-dot", quantized.q4_k_q8_k),
    Program("q6_k_q8_k", "packed-dot", quantized.q6_k_q8_k),
    Program("ggml_q4_k_row", "deployment", deployment.vec_dot),
    Program("ggml_q6_k_row", "deployment", deployment.vec_dot, (("format", "Q6_K"),)),
    Program("field_window_dequantize", "dequantization", quantized.field_windows),
]
for _format in ("Q1_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q8_1", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K", "TQ2_0"):
    _entries.append(Program(_format.lower() + "_dequantize", "dequantization", quantized.dequantize, (("format", _format),)))
_entries.append(Program("iq4_nl_dequantize", "dequantization", quantized.dequantize_iq4_nl))
for _format, _function in (
    ("IQ1_S", codebook.iq1_s),
    ("IQ1_M", codebook.iq1_m),
    ("IQ2_S", codebook.iq2_s),
    ("IQ3_S", codebook.iq3_s),
    ("IQ3_XXS", codebook.iq3_xxs),
    ("IQ4_XS", codebook.iq4_xs),
    ("TQ1_0", codebook.tq1_0),
):
    _entries.append(Program(_format.lower() + "_dequantize", "dequantization", _function))
for _format in ("IQ2_XS", "IQ2_XXS"):
    _entries.append(Program(_format.lower() + "_dequantize", "dequantization", codebook.iq2_indexed, (("format", _format),)))
for _format in ("MXFP4", "NVFP4"):
    _entries.append(Program(_format.lower() + "_dequantize", "dequantization", codebook.fp4, (("format", _format),)))

for _format in ("Q8_0", "Q8_1", "Q8_K"):
    _entries.append(Program(_format.lower() + "_quantize", "preparation", preparation.quantize, (("format", _format),)))

for _format in (
    "Q1_0",
    "Q4_0",
    "Q4_1",
    "Q5_0",
    "Q5_1",
    "Q8_0",
    "Q2_K",
    "Q3_K",
    "Q4_K",
    "Q5_K",
    "Q6_K",
    "IQ1_S",
    "IQ1_M",
    "IQ2_S",
    "IQ2_XS",
    "IQ2_XXS",
    "IQ3_S",
    "IQ3_XXS",
    "IQ4_NL",
    "IQ4_XS",
    "TQ1_0",
    "TQ2_0",
    "MXFP4",
    "NVFP4",
):
    _activation = (
        "Q8_1"
        if _format in ("Q4_1", "Q5_1")
        else "Q8_0"
        if _format in ("Q1_0", "Q4_0", "Q5_0", "Q8_0", "IQ4_NL", "MXFP4", "NVFP4")
        else "Q8_K"
    )
    _defaults = (("format", _format), ("activation", _activation))
    _entries.append(Program(_format.lower() + "_dot", "packed-dot", packed.vec_dot, _defaults))
    _entries.append(Program(_format.lower() + "_projection", "packed-projection", packed.projection, _defaults))

catalog = MappingProxyType({entry.name: entry for entry in _entries})


def get(name):
    return catalog[name]


def programs(*, family=None):
    return tuple(program for program in catalog.values() if family is None or program.family == family)
