# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""The API convert the TensorRT-LLM checkpoint to the engines."""

import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from tensorrt_llm._common import check_max_num_tokens
from tensorrt_llm.builder import BuildConfig
from tensorrt_llm.commands.build import build_and_save
from tensorrt_llm.models import PretrainedConfig
from tensorrt_llm.models.modeling_utils import WEIGHT_LOADER_MODELS
from tensorrt_llm.plugin import PluginConfig


# TODO: re-enable the refit_engine_dir flag.
def build_tensorrt_llm(
    pretrained_config: Union[str, Path],
    engine_dir: Union[str, Path],
    max_input_len: int = 200,
    max_output_len: int = 200,
    max_batch_size: int = 1,
    max_beam_width: int = 1,
    max_num_tokens: Optional[int] = None,
    num_build_workers: int = 1,
    enable_sparsity: bool = False,
    max_prompt_embedding_table_size: int = 0,
    opt_batch_size: int = 0,
):
    """The API to convert the TensorRT-LLM checkpoint to engines.

    Args:
        pretrained_config: The pretrained_config (file path) exported by
            ``modelopt.torch.export.export_tensorrt_llm_checkpoint``.
        engine_dir: The target output directory to save the built tensorrt_llm engines.
        max_input_len: The max input sequence length.
        max_output_len: The max output sequence length.
        max_batch_size: The max batch size.
        max_beam_width: The max beam search width.
        max_num_tokens: The max number of tokens that can be processed at the same time.
            For the context phase, the max_num_tokens counts the full sequence length.
            For the generation phase, the max_num_tokens counts only the ones under generation
            as the input sequence has been processed as cached.
            max_num_tokens should fall between [max_batch_size * max_beam_width, max_batch_size * max_input_len].
            when inflight batching is enabled.
            Higher max_num_tokens means more GPU memory will be used for resource allocation.
            If not specified the max_num_tokens will be set to the max bound.
            Details: https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/perf_best_practices.md
        num_build_workers: The number of workers to use for the building process.
            If build time is a concern, you can increase this worker count to num of GPUs.
            At a lost of higer CPU memory usage footprint.
            If CPU memory is limited, num_build_workers should be set to 1 to conserve memory.
        enable_sparsity: The switch to enable sparsity for TRT compiler.
            With this flag, the TRT compiler will search tactics of sparse kernels for each node of which
            weight tensors are sparsified. This increases engine building time significantly.
        max_prompt_embedding_table_size: Length of the prepended/concatenated embeddings (either multimodal
            feature embeddings or prompt tuning embeddings) to the LLM input embeddings.
        opt_batch_size: The batch size that TensorRT-LLM kernels are optimized for.
            If not specified, it's set to the max_batch_size assuming batched inference.
    """
    engine_dir = Path(engine_dir)
    engine_dir.mkdir(parents=True, exist_ok=True)

    pretrained_config_json = Path(pretrained_config)
    assert pretrained_config_json.exists()
    ckpt_dir = pretrained_config_json.parent

    config = PretrainedConfig.from_json_file(pretrained_config_json)

    if opt_batch_size == 0:
        opt_batch_size = max_batch_size

    build_config = _get_build_config(
        config,
        max_input_len=max_input_len,
        max_output_len=max_output_len,
        max_batch_size=max_batch_size,
        opt_batch_size=opt_batch_size,
        max_beam_width=max_beam_width,
        max_num_tokens=max_num_tokens,
        weight_sparsity=enable_sparsity,
        max_prompt_embedding_table_size=max_prompt_embedding_table_size,
    )

    if num_build_workers == 1:
        for rank in range(config.mapping.world_size):
            passed = _build_tensorrt_llm_rank(
                rank,
                rank % num_build_workers,
                config,
                build_config,
                engine_dir,
                ckpt_dir=ckpt_dir,
            )
            assert passed, "Engine building failed, please check error log."
    else:
        with ProcessPoolExecutor(
            mp_context=get_context("spawn"), max_workers=num_build_workers
        ) as p:
            futures = [
                p.submit(
                    _build_tensorrt_llm_rank,
                    rank,
                    rank % num_build_workers,
                    config,
                    build_config,
                    engine_dir,
                    ckpt_dir=ckpt_dir,
                )
                for rank in range(config.mapping.world_size)
            ]
            exceptions = []
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    traceback.print_exc()
                    exceptions.append(e)
            assert len(exceptions) == 0, "Engine building failed, please check error log."


def build_tensorrt_llm_rank(
    pretrained_config: Dict[str, Any],
    weights: Dict[str, torch.Tensor],
    rank: int,
    engine_dir: Union[str, Path],
    max_input_len: int = 200,
    max_output_len: int = 200,
    max_batch_size: int = 1,
    max_beam_width: int = 1,
    max_num_tokens: Optional[int] = None,
    enable_sparsity: bool = False,
    max_prompt_embedding_table_size: int = 0,
    opt_batch_size: int = 0,
):
    """The API to convert the TensorRT-LLM checkpoint to the engine for a single rank.

    Args:
        pretrained_config: The pretrained_config (dict) exported by
            ``modelopt.torch.export.torch_to_tensorrt_llm_checkpoint``.
        weights: a dict of model weights and scaling factors.
            If not provided, the weights will be loaded from the directory of the pretrained_config.
        rank: the GPU rank of the engine to build.
        engine_dir: The target output directory to save the built tensorrt_llm engines.
        max_input_len: The max input sequence length.
        max_output_len: The max output sequence length.
        max_batch_size: The max batch size.
        max_beam_width: The max beam search width.
        max_num_tokens: The max number of tokens that can be processed at the same time.
            For the context phase, the max_num_tokens counts the full sequence length.
            For the generation phase, the max_num_tokens counts only the ones under generation
            as the input sequence has been processed as cached.
            max_num_tokens should fall between [max_batch_size * max_beam_width, max_batch_size * max_input_len].
            when inflight batching is enabled.
            Higher max_num_tokens means more GPU memory will be used for resource allocation.
            If not specified the max_num_tokens will be set to the max bound.
            Details: https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/perf_best_practices.md
        enable_sparsity: The switch to enable sparsity for TRT compiler.
            With this flag, the TRT compiler will search tactics of sparse kernels for each node of which
            weight tensors are sparsified. This increases engine building time significantly.
        max_prompt_embedding_table_size: Length of the prepended/concatenated embeddings (either multimodal
            feature embeddings or prompt tuning embeddings) to the LLM input embeddings.
        opt_batch_size: The batch size that TensorRT-LLM kernels are optimized for.
            If not specified, it's set to the max_batch_size assuming batched inference.
    """
    engine_dir = Path(engine_dir)
    engine_dir.mkdir(parents=True, exist_ok=True)

    config = PretrainedConfig.from_dict(pretrained_config)
    assert weights, "Please specify the weights dict"

    if opt_batch_size == 0:
        opt_batch_size = max_batch_size

    build_config = _get_build_config(
        config,
        max_input_len=max_input_len,
        max_output_len=max_output_len,
        max_batch_size=max_batch_size,
        opt_batch_size=opt_batch_size,
        max_beam_width=max_beam_width,
        max_num_tokens=max_num_tokens,
        weight_sparsity=enable_sparsity,
        max_prompt_embedding_table_size=max_prompt_embedding_table_size,
    )

    passed = _build_tensorrt_llm_rank(
        rank,
        rank % torch.cuda.device_count(),
        config,
        build_config,
        engine_dir,
        weights=weights,
    )
    assert passed, "Engine building failed, please check error log."


def _get_build_config(
    pretrained_config: PretrainedConfig,
    max_input_len: int,
    max_output_len: int,
    max_batch_size: int,
    opt_batch_size: int,
    max_beam_width: int,
    max_num_tokens: Optional[int],
    weight_sparsity: Optional[bool] = False,
    max_prompt_embedding_table_size: int = 0,
):
    quant_algo = pretrained_config.quantization.quant_algo
    is_awq = quant_algo is not None and "AWQ" in quant_algo

    use_qdq = pretrained_config.quantization.quant_algo in [
        "FP8",
        "W8A8_SQ_PER_CHANNEL",
    ]

    plugin_config = PluginConfig.from_dict(
        {
            # Plugins
            "gpt_attention_plugin": pretrained_config.dtype,
            "gemm_plugin": pretrained_config.dtype if not use_qdq else None,
            "nccl_plugin": pretrained_config.dtype,
            "weight_only_groupwise_quant_matmul_plugin": (
                pretrained_config.dtype if is_awq else None
            ),
        }
    )

    max_num_tokens, opt_num_tokens = check_max_num_tokens(
        max_num_tokens=max_num_tokens,
        opt_num_tokens=None,  # equal to max_batch_size*max_beam_width by default
        max_batch_size=max_batch_size,
        max_input_len=max_input_len,
        max_beam_width=max_beam_width,
        remove_input_padding=plugin_config.remove_input_padding,
        enable_context_fmha=plugin_config.context_fmha,
        tokens_per_block=plugin_config.tokens_per_block,
    )

    if max_batch_size < 4:
        print(
            "Warning: TensorRT LLM may hit a runtime issue with batch size is smaller than 4 on some models."
            " Force set to 4"
        )
        max_batch_size = 4

    build_config = BuildConfig.from_dict(
        {
            "max_input_len": max_input_len,
            "max_output_len": max_output_len,
            "max_batch_size": max_batch_size,
            "opt_batch_size": opt_batch_size,
            "max_beam_width": max_beam_width,
            "max_num_tokens": max_num_tokens,
            "opt_num_tokens": opt_num_tokens,
            "max_prompt_embedding_table_size": max_prompt_embedding_table_size,
            "gather_context_logits": False,
            "gather_generation_logits": False,
            "strongly_typed": use_qdq,
            "builder_opt": 4 if "RecurrentGemma" not in pretrained_config.architecture else None,
            "weight_sparsity": weight_sparsity,
            "profiling_verbosity": "layer_names_only",
            "enable_debug_output": False,
            "speculative_decoding_mode": (
                "medusa" if "Medusa" in pretrained_config.architecture else None
            ),
        },
        plugin_config=plugin_config,
    )

    build_config.use_fused_mlp = pretrained_config.quantization.quant_algo in [
        "FP8",
        None,
    ] and pretrained_config.hidden_act in ["silu", "swiglu", "fast-swiglu", "gelu", "geglu"]

    return build_config


def _build_tensorrt_llm_rank(
    rank: int,
    gpu_id: int,
    pretrained_config: PretrainedConfig,
    build_config: BuildConfig,
    engine_dir: Union[str, Path],
    ckpt_dir: Optional[Path] = None,
    log_level="warning",
):
    kwargs = {}
    if pretrained_config.architecture in WEIGHT_LOADER_MODELS:
        kwargs["tp_size"] = pretrained_config.mapping.tp_size
        kwargs["pp_size"] = pretrained_config.mapping.pp_size
        kwargs["world_size"] = kwargs["tp_size"] * kwargs["pp_size"]

    success = build_and_save(
        rank,
        gpu_id,
        ckpt_dir,
        build_config,
        engine_dir,
        log_level,
        pretrained_config,
        model_cls=None,
        **kwargs,
    )

    return success
