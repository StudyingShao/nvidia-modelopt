# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Quantized batch normalization module."""

import torch.nn as nn

from .quant_module import QuantInputBase, QuantModuleRegistry
from .tensor_quantizer import TensorQuantizer


class _QuantBatchNormNd(QuantInputBase):

    @property
    def input_quantizer(self):
        return self._input_bn_quantizer

    @property
    def output_quantizer(self):
        return self._output_bn_quantizer

    def _setup(self):
        self._register_temp_attribute(
            "_input_bn_quantizer", TensorQuantizer(self.default_quant_desc_input)
        )
        self._register_temp_attribute(
            "_output_bn_quantizer", TensorQuantizer(self.default_quant_desc_output)
        )
        self._output_bn_quantizer.disable()


QuantModuleRegistry.register({nn.BatchNorm1d: "nn.BatchNorm1d"})(_QuantBatchNormNd)
QuantModuleRegistry.register({nn.BatchNorm2d: "nn.BatchNorm2d"})(_QuantBatchNormNd)
QuantModuleRegistry.register({nn.BatchNorm3d: "nn.BatchNorm3d"})(_QuantBatchNormNd)
