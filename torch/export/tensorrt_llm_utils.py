# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Utils for TensorRT-LLM checkpoint export.

Some of the logics in this file are empirical and needs constant update if exceptions occur.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from modelopt import __version__

from .model_config import (
    LAYERNORM_DEFAULT,
    LAYERNORM_RMS,
    QUANTIZATION_NONE,
    ModelConfig,
)

MODEL_NAME_TO_HF_ARCH_MAP = {
    "bloom": "BloomForCausalLM",
    "baichuan": "BaichuanForCausalLM",
    "chatglm": "ChatGLMForCausalLM",
    "falcon": "FalconForCausalLM",
    "gptj": "GPTJForCausalLM",
    "llama": "LlamaForCausalLM",
    "mpt": "MPTForCausalLM",
    "qwen": "QWenForCausalLM",
    "gemma": "GemmaForCausalLM",
    "phi": "PhiForCausalLM",
    "phi3": "Phi3ForCausalLM",
    "phi3small": "Phi3SmallForCausalLM",
    "gpt2": "GPTForCausalLM",
    "gptnext": "GPTForCausalLM",
    "recurrentgemma": "RecurrentGemmaForCausalLM",
    "glm": "ChatGLMForCausalLM",
}


def is_tensorrt_llm_0_8_or_9():
    """Returns true if tensorrt_llm version is 0.8 or 0.9."""
    try:
        import tensorrt_llm

        return tensorrt_llm.__version__.startswith(("0.8", "0.9"))
    except Exception:
        return False


def _find_layernorm_type(model_config: ModelConfig):
    if model_config.ln_f:
        return model_config.ln_f.layernorm_type
    for layer in model_config.layers:
        if layer.input_layernorm:
            return layer.input_layernorm.layernorm_type
        if layer.post_layernorm:
            return layer.post_layernorm.layernorm_type
    return LAYERNORM_DEFAULT


def convert_to_tensorrt_llm_config(
    model_config: ModelConfig, tp_size_overwrite: Optional[int] = None
):
    """Convert to TensorRT-LLM checkpoint config.

    `tp_size_overwrite` overwrites the tp_size in config.mapping, set only only for phi with TP.
    This is because the TRT-LLM builder expects its checkpoint to be unsharded.
    """
    decoder_type = model_config.layers[0].decoder_type
    tp_size = model_config.tensor_parallel
    pp_size = model_config.pipeline_parallel

    first_attention_config = None
    first_attention_decoder_config = None
    for decoder_layer in model_config.layers:
        if decoder_layer.attention:
            first_attention_config = decoder_layer.attention
            first_attention_decoder_config = decoder_layer
            break

    assert (
        first_attention_config is not None and first_attention_decoder_config is not None
    ), "Model must have at least one attention block"

    config = {
        "producer": {
            "name": "modelopt",
            "version": __version__,
        },
        "architecture": MODEL_NAME_TO_HF_ARCH_MAP[decoder_type],
        "dtype": model_config.dtype,
        "logits_dtype": "float16" if model_config.dtype == "bfloat16" else model_config.dtype,
        "num_hidden_layers": len(model_config.layers) * pp_size,
        "num_attention_heads": model_config.num_attention_heads,
        "num_key_value_heads": model_config.num_kv_heads,
        "hidden_size": model_config.hidden_size,
        "norm_epsilon": first_attention_decoder_config.input_layernorm.eps,
        "vocab_size": model_config.vocab_size,
        "max_position_embeddings": model_config.max_position_embeddings,
        "hidden_act": model_config.hidden_act,
        "use_parallel_embedding": True,
        "embedding_sharding_dim": 0,
        "quantization": {"quant_algo": None, "kv_cache_quant_algo": None},
        "mapping": {
            "world_size": tp_size_overwrite * pp_size if tp_size_overwrite else tp_size * pp_size,
            "tp_size": tp_size_overwrite if tp_size_overwrite else tp_size,
            "pp_size": pp_size,
        },
        "head_size": first_attention_decoder_config.attention_head_size,
        "intermediate_size": first_attention_decoder_config.ffn_hidden_size_local * tp_size,
        "position_embedding_type": (
            "alibi" if first_attention_decoder_config.use_alibi else "rope_gpt_neox"
        ),
        "share_embedding_table": True if (model_config.lm_head is None and pp_size == 1) else False,
        "residual_mlp": first_attention_decoder_config.residual_mlp is not None,
        # Model Optimizer customized fields
        "bias": first_attention_config.dense.bias is not None,
        "rotary_pct": first_attention_decoder_config.rotary_pct,
        "rank": model_config.rank,
        "decoder": first_attention_decoder_config.decoder_type,
        "rmsnorm": _find_layernorm_type(model_config) == LAYERNORM_RMS,
        "lm_head_bias": model_config.lm_head is not None and model_config.lm_head.bias is not None,
    }

    if first_attention_decoder_config.rotary_base:
        config["rotary_base"] = first_attention_decoder_config.rotary_base

    if model_config.quantization == "fp8":
        config["quantization"].update({"quant_algo": "FP8"})
    elif model_config.quantization == "int4_awq":
        config["quantization"].update(
            {
                "quant_algo": "W4A16_AWQ",
                "group_size": first_attention_config.qkv.awq_block_size,
                "has_zero_point": False,
                "pre_quant_scale": True,
                "exclude_modules": ["lm_head"],
            }
        )
    elif model_config.quantization == "w4a8_awq":
        config["quantization"].update(
            {
                "quant_algo": "W4A8_AWQ",
                "group_size": first_attention_config.qkv.awq_block_size,
                "has_zero_point": False,
                "pre_quant_scale": True,
                "exclude_modules": ["lm_head"],
            }
        )
        # W4A8 only supports float32 logits_dtype now.
        config["logits_dtype"] = "float32"
    elif model_config.quantization == "int8_sq":
        config["quantization"].update(
            {
                "quant_algo": "W8A8_SQ_PER_CHANNEL",
            }
        )
    elif model_config.quantization == QUANTIZATION_NONE:
        config["quantization"].update(
            {
                "quant_algo": None,
            }
        )
    else:
        config["quantization"].update(
            {
                "quant_algo": model_config.quantization,
            }
        )

    if first_attention_config.kv_cache_dtype is not None:
        config["quantization"].update(
            {
                "kv_cache_quant_algo": first_attention_config.kv_cache_dtype,
            }
        )

    if decoder_type == "gpt2":
        config["position_embedding_type"] = "learned_absolute"
    elif decoder_type == "chatglm":
        config.update(
            {
                "position_embedding_type": "rope_gptj",
                "intermediate_size": model_config.layers[0].ffn_hidden_size_local * tp_size // 2,
                "max_position_embeddings": model_config.layers[0].seq_length,  # 32768
                "chatglm_version": model_config.layers[0].model_name.split("_")[0],
                "add_bias_linear": first_attention_config.dense.bias is not None,  # False
                "add_qkv_bias": first_attention_config.qkv.bias is not None,  # True
                "apply_query_key_layer_scaling": False,
                "apply_residual_connection_post_layernorm": model_config.layers[
                    0
                ].apply_residual_connection_post_layernorm,  # False
                "rope_ratio": model_config.layers[0].rope_ratio,
            }
        )
    elif decoder_type == "glm":
        config.update(
            {
                "max_position_embeddings": 1024,
                "chatglm_version": "glm",
                "add_bias_linear": model_config.layers[0].attention.dense.bias is not None,  # True
                "add_qkv_bias": model_config.layers[0].attention.qkv.bias is not None,  # True
                "apply_query_key_layer_scaling": False,
                "apply_residual_connection_post_layernorm": model_config.layers[
                    0
                ].apply_residual_connection_post_layernorm,  # False
                "position_embedding_type": "learned_absolute",
                "rope_ratio": model_config.layers[0].rope_ratio,
                "use_parallel_embedding": "False",
            }
        )
    elif decoder_type == "falcon":
        config.update(
            {
                "position_embedding_type": (
                    "alibi_with_scale" if model_config.layers[0].use_alibi else "rope_gpt_neox"
                ),
                "parallel_attention": model_config.layers[0].parallel_attention,
                "new_decoder_architecture": model_config.layers[0].new_decoder_architecture,
            }
        )
    elif decoder_type == "gptj":
        config.update(
            {
                "position_embedding_type": "rope_gptj",
                "rotary_dim": first_attention_config.rotary_dim,
            }
        )
    elif (
        decoder_type == "llama" and model_config.layers[0].moe_num_experts
    ):  # For Mixtral and Arctic
        config.update(
            {
                "moe_num_experts": model_config.layers[0].moe_num_experts,
                "moe_top_k": model_config.layers[0].moe_top_k,
            }
        )
    elif decoder_type == "mpt":
        config.update(
            {
                "clip_qkv": first_attention_config.clip_qkv,
                "alibi_bias_max": model_config.layers[0].alibi_bias_max,
            }
        )
    elif decoder_type == "qwen":
        intermediate_size = model_config.layers[0].ffn_hidden_size_local * tp_size
        qwen_type = "qwen"
        if model_config.layers[0].qwen_type:
            qwen_type = model_config.layers[0].qwen_type  # "qwen" or "qwen2"
        # Qwen version 1 has actual intermediate_size one half of what's in hf_config
        if qwen_type == "qwen":
            intermediate_size *= 2
        config.update(
            {
                "intermediate_size": intermediate_size,
                "seq_length": model_config.layers[0].seq_length,
                "qwen_type": (
                    model_config.layers[0].qwen_type if model_config.layers[0].qwen_type else "qwen"
                ),
            }
        )
    elif decoder_type == "phi":
        config["partial_rotary_factor"] = model_config.layers[0].partial_rotary_factor
    elif decoder_type == "recurrentgemma":
        config["conv_kernel"] = 4
        config["state_size"] = 1
        config["state_dtype"] = "float32"
        config["rnn_hidden_size"] = model_config.layers[0].rnn_hidden_size
        config["logits_soft_cap"] = model_config.layers[0].logits_soft_cap
        config["emb_scale_by_sqrt_dim"] = model_config.layers[0].emb_scale_by_sqrt_dim
        config["layer_types"] = model_config.layers[0].layer_types
    elif "phi3" in decoder_type:
        config["intermediate_size"] = config["intermediate_size"] // 2  # fc and gate are merged
        config["original_max_position_embeddings"] = (
            first_attention_decoder_config.original_max_position_embeddings
        )
        if (
            model_config.layers[0].longrope_scaling_short_factors is not None
            and model_config.layers[0].longrope_scaling_short_factors is not None
        ):
            config.update(
                {
                    "longrope_scaling_short_factors": model_config.layers[
                        0
                    ].longrope_scaling_short_factors,
                    "longrope_scaling_long_factors": model_config.layers[
                        0
                    ].longrope_scaling_long_factors,
                }
            )
    if decoder_type == "phi3small":
        config["mup_attn_multiplier"] = model_config.layers[0].mup_attn_multiplier
        config["mup_embedding_multiplier"] = model_config.layers[0].mup_embedding_multiplier
        config["mup_use_scaling"] = model_config.layers[0].mup_use_scaling
        config["mup_width_multiplier"] = model_config.layers[0].mup_width_multiplier
        config["blocksparse_block_size"] = model_config.layers[0].blocksparse_block_size
        config["blocksparse_homo_head_pattern"] = model_config.layers[
            0
        ].blocksparse_homo_head_pattern
        config["blocksparse_num_local_blocks"] = model_config.layers[0].blocksparse_num_local_blocks
        config["blocksparse_vertical_stride"] = model_config.layers[0].blocksparse_vertical_stride
        config["dense_attention_every_n_layers"] = model_config.layers[
            0
        ].dense_attention_every_n_layers
        config["gegelu_limit"] = model_config.layers[0].gegelu_limit
        if (
            model_config.layers[0].longrope_scaling_short_factors is not None
            and model_config.layers[0].longrope_scaling_short_factors is not None
        ):
            config.update(
                {
                    "longrope_short_mscale": model_config.layers[0].longrope_short_mscale,
                    "longrope_long_mscale": model_config.layers[0].longrope_long_mscale,
                }
            )

    # Handle Medusa decoding
    # TODO (chenhany): when inference pp > 1; only last pp has medusa heads
    if model_config.medusa_heads is not None:
        config["base_architecture"] = config["architecture"]
        config["architecture"] = "MedusaForCausalLM"
        # NOTE: max_draft_len is related to the medusa tree len. Currently it is hardcoded to 63.
        config["max_draft_len"] = 63
        config["num_medusa_heads"] = len(model_config.medusa_heads)
        config["num_medusa_layers"] = len(model_config.medusa_heads[0].medusa_layers)
        config["quantization"]["exclude_modules"] = ["lm_head", "medusa_heads"]

    return config


def weights_to_npz(
    weights: Dict[str, np.ndarray], tensorrt_llm_config: Dict[str, Any], export_dir: Path
):
    """Export the model_config and the weights in the backward-compatible npz forward."""
    print("Warning: this is an old NPZ format and will be deprecated soon.")

    # step 1: rename key
    def get_npz_key(k):
        key_mapping = {
            "transformer.position_embedding": "_np:position_embedding:weight",
            "transformer.vocab_embedding": "_np:vocab_embedding:weight",
            "transformer.ln_f.weight": "_np:final_layernorm:weight",
            "transformer.ln_f.bias": "_np:final_layernorm:bias",
        }
        if k in key_mapping:
            return key_mapping[k]

        if "lm_head" in k:
            # src: lm_head.weight
            # dst: _np:lm_head:weight
            ns = k.split(".")
            return ":".join(["_np"] + ns)
        else:
            # src: transformers.layers.0.attention.q.weight
            # dst: _np:layers:20:attention:qkv:q:weight
            ns = k.split(".")
            return ":".join(["_np"] + ns[1:])

    # numpy doesn't know bfloat16, define abstract binary type instead

    np_bfloat16 = np.dtype("V2", metadata={"dtype": "bfloat16"})

    def _torch_to_numpy(x):
        if x.dtype != torch.bfloat16:
            return x.detach().cpu().numpy()
        return x.detach().view(torch.int16).cpu().numpy().view(np_bfloat16)

    np_weights = {}
    for k in list(weights):
        np_weights[get_npz_key(k)] = _torch_to_numpy(weights.pop(k))
    weights = np_weights

    # step 2: awq post process
    if "AWQ" in tensorrt_llm_config.get("quantization", {}).get("quant_algo", ""):
        for k in list(weights):
            if k.endswith("weights_scaling_factor"):
                if "qkv" in k:
                    weights[k] = np.transpose(weights[k])
                else:
                    weights[k] = weights[k].flatten()

    decoder = tensorrt_llm_config["decoder"]
    tp_size = tensorrt_llm_config["mapping"]["tp_size"]
    pp_size = tensorrt_llm_config["mapping"]["pp_size"]

    weights_path = export_dir / f"{decoder}_tp{tp_size}_rank{pp_size}.npz"
    np.savez(weights_path, **weights)
