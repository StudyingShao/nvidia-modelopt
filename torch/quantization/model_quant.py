# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""User-facing quantization API."""
from typing import Any, Callable, Dict, Optional, Union

import torch.nn as nn

from modelopt.torch.opt import apply_mode
from modelopt.torch.opt.searcher import ForwardLoop

from .conversion import set_quantizer_attribute
from .mode import QuantizeModeRegistry
from .model_calib import calibrate
from .nn import TensorQuantizer

__all__ = [
    "quantize",
    "disable_quantizer",
    "enable_quantizer",
    "print_quant_summary",
    "fold_weight",
]


def quantize(
    model: nn.Module,
    config: Dict[str, Any],
    forward_loop: Optional[ForwardLoop] = None,
) -> nn.Module:
    """Quantizes and calibrates the model.

    This method performs replacement of modules with their quantized counterparts and
    performs calibration as specified by ``quant_cfg``.
    ``forward_loop`` is used to forward data through the model and gather statistics for calibration.

    Args:
        model: A pytorch model
        config: A dictionary specifying the values for keys ``"quant_cfg"`` and ``"algorithm"``.
            The ``"quant_cfg"`` key specifies the quantization configurations.
            The ``"algorithm"`` key specifies the ``algorithm`` argument to
            :meth:`calibrate <modelopt.torch.quantization.model_calib.calibrate>`.

            Quantization configurations is a dictionary mapping wildcards or filter functions
            to its quantizer attributes. The wildcards or filter functions  are matched
            against the quantizer module names. The quantizer modules have names ending with
            ``weight_quantizer`` and ``input_quantizer`` and they perform weight quantization and
            input quantization (or activation quantization) respectively. The quantizer modules
            are instances of
            :class:`TensorQuantizer <modelopt.torch.quantization.nn.modules.tensor_quantizer.TensorQuantizer>` and the
            specified quantizer attributes describe its quantization behavior. See
            :meth:`set_quantizer_by_cfg <modelopt.torch.quantization.conversion.set_quantizer_by_cfg>` for more details
            on ``"quant_cfg"`` dictionary.

            An example ``config`` dictionary is given below:

            .. code-block::python

                config = {

                    "quant_cfg": {
                        # "num_bits" specifies the number of bits for quantization
                        # "axis" specifies the axis for quantization
                        "*weight_quantizer": {"num_bits": 8, "axis": 0},
                        "*input_quantizer": {"num_bits": 8, "axis": -1},

                        # Default quantization settings
                        "default": {"num_bits": 8, "axis": None},
                    }
                    "algorithm": "max"
                }

            Please see :mod:`config <modelopt.torch.quantization.config>` for more examples.

        forward_loop: A callable that forwards all calibration data through the model. This is used
            to gather statistics for calibration. It should take model as the argument. It does not need
            to return anything. Here are a few examples for correct ``forward_loop`` definitions:
            Example 1:

            .. code-block::

                    def forward_loop(model) -> None:
                        # iterate over the data loader and forward data through the model
                        for batch in data_loader:
                            model(batch)

            Example 2:

            .. code-block::

                    def forward_loop(model) -> float:
                        # evaluate the model on the task
                        return evaluate(model, task, ....)

            Example 3:

            .. code-block::

                    def forward_loop(model) -> None:
                        # run evaluation pipeline
                        evaluator.model = model
                        evaluator.evaluate()

            .. note::

                Calibration does not require forwarding the entire dataset through the model.
                Please subsample the dataset or reduce the number of batches if needed.

    Returns: A pytorch model which has been quantized and calibrated.
    """
    model = apply_mode(model, mode=[("quantize", config)], registry=QuantizeModeRegistry)
    return calibrate(model, config["algorithm"], forward_loop=forward_loop)


def disable_quantizer(model: nn.Module, wildcard_or_filter_func: Union[str, Callable]):
    """Disable quantizer by wildcard or filter function."""
    set_quantizer_attribute(model, wildcard_or_filter_func, {"enable": False})


def enable_quantizer(model: nn.Module, wildcard_or_filter_func: Union[str, Callable]):
    """Enable quantizer by wildcard or filter function."""
    set_quantizer_attribute(model, wildcard_or_filter_func, {"enable": True})


def print_quant_summary(model: nn.Module):
    """Print summary of all quantizer modules in the model."""
    count = 0
    for name, mod in model.named_modules():
        if isinstance(mod, TensorQuantizer):
            print(f"{name:80} {mod}")
            count += 1
    print(f"{count} TensorQuantizers found in model")


def fold_weight(model: nn.Module):
    """Fold weight quantizer for fast evaluation."""
    for name, module in model.named_modules():
        if hasattr(module, "weight_quantizer") and hasattr(module, "weight"):
            module.weight.data.copy_(
                (module.weight_quantizer(module.weight.float())).to(module.weight.dtype)
            )
            module.weight_quantizer.disable()
