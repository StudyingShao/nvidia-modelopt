# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""LLM deployment package with tensorrt_llm.

Model Optimizer supports automatic conversion of Model Optimizer exported LLM to TensorRT-LLM
engines for accelerated inferencing.

Convert to TensorRT-LLM:

Model Optimizer offers a single API to build the exported model from the quantization stage on top
of the TensorRT-LLM build API.

.. code-block:: python

    from modelopt.deploy.llm import build_tensorrt_llm

    build_tensorrt_llm(
        pretrained_config=pretrained_config_json_path,
        engine_dir=engine_dir,
        max_input_len=max_input_len,
        max_output_len=max_output_len,
        max_batch_size=max_batch_size,
        max_beam_width=max_num_beams,
        num_build_workers=num_build_workers,
    )

Batched Inference with TensorRT-LLM:

Model Optimizer offers an easy-to-use python API to run batched offline inferences to test the TensorRT-LLM
engine(s) built.

For example:

.. code-block:: python

    from modelopt.deploy.llm import generate, load

    # The host_context loading (called once).
    host_context = load(tokenizer=tokenizer, engine_dir=engine_dir, num_beams=num_beams)
    # generate could be called multiple times as long as the host_context is present.
    outputs = generate(input_texts, max_output_len, host_context)
    print(outputs)

"""

from mpi4py import MPI

# Pre load MPI libs to avoid tensorrt_llm importing failures.
print(f"Loaded mpi lib {MPI.__file__} successfully")

# Pre import tensorrt_llm
try:
    import tensorrt_llm
except Exception as e:
    print(
        "tensorrt_llm package is not installed. Please build or install tensorrt_llm package"
        " properly before calling the llm deployment API."
    )
    raise (e)

from .model_config_trt import *  # noqa
from .generate import *  # noqa
