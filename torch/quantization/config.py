# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""This document lists the quantization formats supported by Model Optimizer and example quantization configs.

.. _quantization-formats:

Quantization Formats
==========================================

The following table lists the quantization formats supported by Model Optimizer and the corresponding quantization
config. See :ref:`Quantization Configs <example-quantization-configs>` for the
specific quantization config definitions.

Please see :doc:`choosing the right quantization formats <../../guides/_choosing_quant_methods>` to
learn more about the formats and their use-cases.

.. note::

    The recommended configs given below are for LLM models. For CNN models, only INT8 quantization
    is supported. Please use quantization config ``INT8_DEFAULT_CFG`` for CNN models.

=================================   =======================================================
Quantization  Format                Model Optimizer config
=================================   =======================================================
INT8                                ``INT8_SMOOTHQUANT_CFG``

FP8                                 ``FP8_DEFAULT_CFG``

INT4 Weights only AWQ (W4A16)       ``INT4_AWQ_CFG``

INT4-FP8 AWQ (W4A8)                 ``W4A8_AWQ_BETA_CFG``

=================================   =======================================================

.. _quantization-configs:

Quantization Configs
================================

Quantization config is dictionary specifying the values for keys ``"quant_cfg"`` and
``"algorithm"``. The ``"quant_cfg"`` key specifies the quantization configurations. The
``"algorithm"`` key specifies the ``algorithm`` argument to
:meth:`calibrate <modelopt.torch.quantization.model_calib.calibrate>`.

Quantization configurations is a dictionary mapping wildcards or filter functions
to its quantizer attributes. The wildcards or filter functions  are matched
against the quantizer module names. The quantizer modules have names ending with
``weight_quantizer`` and ``input_quantizer`` and they perform weight quantization and
input quantization (or activation quantization) respectively. The quantizer modules are generally
instances of
:class:`TensorQuantizer <modelopt.torch.quantization.nn.modules.tensor_quantizer.TensorQuantizer>` and
the specified quantizer attributes describe its quantization behavior. Quantizer attributes is a
dictionary mapping quantizer attribute names to their values.

Quantizer attributes can also be a list of dictionaries. In this case, the matched quantizer module
is replaced with a
:class:`SequentialQuantizer <modelopt.torch.quantization.nn.modules.tensor_quantizer.SequentialQuantizer>`
module which is used to quantize a tensor in multiple formats sequentially. Each quantizer attribute
dictionary in the list specifies the quantization formats for each quantization step of the
sequential quantizer. For example, `SequentialQuantizer` is used in 'INT4 Weights, FP8 Activations'
quantization in which the weights are quantized in INT4 followed by FP8.

.. _example-quantization-configs:

Here are examples quantization configs from Model Optimizer:

.. code-block::

    INT8_DEFAULT_CFG = {
        "quant_cfg": {
        "*weight_quantizer": {"num_bits": 8, "axis": 0},
        "*input_quantizer": {"num_bits": 8, "axis": None},
        "*lm_head*": {"enable": False},
        "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
        "default": {"num_bits": 8, "axis": None},
        },
        "algorithm": "max",
    }

    INT8_SMOOTHQUANT_CFG = {
        "quant_cfg": {
        "*weight_quantizer": {"num_bits": 8, "axis": 0},
        "*input_quantizer": {"num_bits": 8, "axis": -1},
        "*lm_head*": {"enable": False},
        "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
        "default": {"num_bits": 8, "axis": None},
        },
        "algorithm": "smoothquant",
    }

    FP8_DEFAULT_CFG = {
        "quant_cfg": {
        "*weight_quantizer": {"num_bits": (4, 3), "axis": None},
        "*input_quantizer": {"num_bits": (4, 3), "axis": None},
        "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
        "default": {"num_bits": (4, 3), "axis": None},
        },
        "algorithm": "max",
    }

    INT4_BLOCKWISE_WEIGHT_ONLY_CFG = {
        "quant_cfg": {
        "*weight_quantizer": {"num_bits": 4, "block_sizes": {-1: 128}, "enable": True},
        "*input_quantizer": {"enable": False},
        "*lm_head*": {"enable": False},
        "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
        "default": {"enable": False},
        },
        "algorithm": "max",
    }

    INT4_AWQ_CFG = {
        "quant_cfg": {
        "*weight_quantizer": {"num_bits": 4, "block_sizes": {-1: 128}, "enable": True},
        "*input_quantizer": {"enable": False},
        "*lm_head*": {"enable": False},
        "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
        "default": {"enable": False},
        },
        "algorithm": {"method": "awq_lite", "alpha_step": 0.1},
        # "algorithm": {"method": "awq_full", "alpha_step": 0.1, "max_co_batch_size": 1024},
        # "algorithm": {"method": "awq_clip", "max_co_batch_size": 2048},
    }

    W4A8_AWQ_BETA_CFG = {
    "quant_cfg": {
        "*weight_quantizer": [
            {"num_bits": 4, "block_sizes": {-1: 128}, "enable": True},
            {"num_bits": (4, 3), "axis": None, "enable": True},
        ],
        "*input_quantizer": {"num_bits": (4, 3), "axis": None, "enable": True},
        "*lm_head*": {"enable": False},
        "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
        "default": {"enable": False},
    },
    "algorithm": "awq_lite",
    }

These config can be accessed as attributes of ``modelopt.torch.quantization`` and can be given as
input to :meth:`mtq.quantize() <modelopt.torch.quantization.model_quant.quantize>`. For example:

.. code-block::

    import modelopt.torch.quantization as mtq
    model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop)

You can also create your own config by following these examples.
For instance, if you want to quantize a model with int4 AWQ algorithm, but need to skip quantizing
the layer named ``lm_head``,  you can create a custom config and quantize your model as following:

.. code-block::

    # Create custom config
    CUSTOM_INT4_AWQ_CFG = copy.deepcopy(mtq.INT4_AWQ_CFG)
    CUSTOM_INT4_AWQ_CFG["quant_cfg"]["*lm_head*"] = {"enable": False}

    # quantize model
    model = mtq.quantize(model, CUSTOM_INT4_AWQ_CFG, forward_loop)

"""

from typing import Any, Callable, Dict, Set, Union

from modelopt.torch.opt.config import ModeloptBaseConfig, ModeloptField

INT8_DEFAULT_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": 8, "axis": 0},
        "*input_quantizer": {"num_bits": 8, "axis": None},
        "*lm_head*": {"enable": False},
        "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
        "*output_layer*": {"enable": False},
        "default": {"num_bits": 8, "axis": None},
    },
    "algorithm": "max",
}

INT8_SMOOTHQUANT_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": 8, "axis": 0},
        "*input_quantizer": {"num_bits": 8, "axis": -1},
        "*lm_head*": {"enable": False},
        "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
        "*output_layer*": {"enable": False},
        "default": {"num_bits": 8, "axis": None},
    },
    "algorithm": "smoothquant",
}

FP8_DEFAULT_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": (4, 3), "axis": None},
        "*input_quantizer": {"num_bits": (4, 3), "axis": None},
        "*lm_head*": {"enable": False},
        "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
        "*output_layer*": {"enable": False},
        "default": {"num_bits": (4, 3), "axis": None},
    },
    "algorithm": "max",
}

FP8_DYNAMIC_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": (4, 3), "axis": -1, "type": "dynamic", "enable": True},
        "*input_quantizer": {"num_bits": (4, 3), "axis": -1, "type": "dynamic", "enable": True},
        "*lm_head*": {"enable": False},
        "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
        "*output_layer*": {"enable": False},
        "default": {"enable": False},
    },
    "algorithm": "max",
}

FP8_WA_FP8_KV_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": (4, 3), "axis": None},
        "*input_quantizer": {"num_bits": (4, 3), "axis": None},
        "*lm_head*": {"enable": False},
        "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
        "*bmm_quantizer": {"num_bits": (4, 3), "axis": None},
        "*output_layer*": {"enable": False},
        "default": {"num_bits": (4, 3), "axis": None},
    },
    "algorithm": "max",
}

INT4_BLOCKWISE_WEIGHT_ONLY_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": 4, "block_sizes": {-1: 128}, "enable": True},
        "*input_quantizer": {"enable": False},
        "*lm_head*": {"enable": False},
        "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
        "*output_layer*": {"enable": False},
        "default": {"enable": False},
    },
    "algorithm": "max",
}

INT4_AWQ_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": 4,
            "block_sizes": {-1: 128, "type": "static"},
            "enable": True,
        },
        "*input_quantizer": {"enable": False},
        "*lm_head*": {"enable": False},
        "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
        "*output_layer*": {"enable": False},
        "default": {"enable": False},
    },
    "algorithm": {"method": "awq_lite", "alpha_step": 0.1},
    # "algorithm": {"method": "awq_full", "alpha_step": 0.1, "max_co_batch_size": 1024},
    # "algorithm": {"method": "awq_clip", "max_co_batch_size": 2048},
}

# W4A8 currently uses INT4 blockwise quantization (block size = 128) followed by FP8 quantization
# for weights. This could change in the future
W4A8_AWQ_BETA_CFG = {
    "quant_cfg": {
        "*weight_quantizer": [
            {"num_bits": 4, "block_sizes": {-1: 128, "type": "static"}, "enable": True},
            {"num_bits": (4, 3), "axis": None, "enable": True},
        ],
        "*input_quantizer": {"num_bits": (4, 3), "axis": -1, "enable": True},
        "*lm_head*": {"enable": False},
        "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
        "*output_layer*": {"enable": False},
        "default": {"enable": False},
    },
    "algorithm": "awq_lite",
}

choices: Set[str] = {
    "INT8_DEFAULT_CFG",
    "INT8_SMOOTHQUANT_CFG",
    "FP8_DEFAULT_CFG",
    "INT4_BLOCKWISE_WEIGHT_ONLY_CFG",
    "INT4_AWQ_CFG",
    "W4A8_AWQ_BETA_CFG",
}


# TODO: [OMNIML-823] refine modes here.
class QuantizeConfig(ModeloptBaseConfig):
    """Default configuration for ``quantize`` mode."""

    quant_cfg: Dict[Union[str, Callable], Any] = ModeloptField(
        default={"default": {"num_bits": 8, "axis": None}}, title="Quantization configuration"
    )
    algorithm: Union[str, Dict[str, Any]] = ModeloptField(
        default="max", title="Calibration algorithm"
    )


class _QuantizeExportConfig(ModeloptBaseConfig):
    """An empty config."""
