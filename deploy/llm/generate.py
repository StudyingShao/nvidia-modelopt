# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""A wrapper over the TensorRT-LLM high level API runner."""


import json
from pathlib import Path
from typing import Dict, Iterable, List, Union

from tensorrt_llm.hlapi import KvCacheConfig as TRT_KvCacheConfig
from tensorrt_llm.hlapi.llm import LLM as TRT_LLM
from tensorrt_llm.hlapi.llm import ModelConfig as TRT_ModelConfig
from tensorrt_llm.hlapi.tokenizer import TokenizerBase, TransformersTokenizer


class LLM(TRT_LLM):
    """A wrapper over the ``tensorrt_llm.hlapi.llm.LLM`` for LLM profiling and validation."""

    def __init__(
        self,
        engine_dir: Union[str, Path],
        tokenizer: TokenizerBase,
        kv_cache_config: Dict[str, Union[int, float]] = {},
        cpp_executor: bool = False,
    ):
        """Initializes the LLM runner class.

        Args:
            engine_dir: the directory path of the TensorRT-LLM engine.
            tokenizer: the tokenizer. For example, a tokenizer from the Huggingface model.
            kv_cache_config: the kv cache config as a dict. Please refer to
                https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/performance/perf-best-practices.md
            cpp_executor: Whether we run generate with cpp_executor.
        """
        trt_llm_config = TRT_ModelConfig(model_dir=engine_dir)

        with open(Path(engine_dir) / "config.json", "r") as engine_config_file:
            engine_config = json.load(engine_config_file)
            build_config = engine_config["build_config"]
            world_size = (
                engine_config.get("pretrained_config", {}).get("mapping", {}).get("world_size", 1)
            )
            max_tokens_in_paged_kv_cache = (
                (
                    build_config["max_input_len"]
                    + build_config["max_output_len"] * build_config["max_beam_width"]
                )
                * build_config["max_batch_size"]
                // world_size
            )

        trt_kv_cache_config = TRT_KvCacheConfig()

        # If not specified, free_gpu_memory_fraction is set to the default TRT LLM value 0.9
        trt_kv_cache_config.free_gpu_memory_fraction = kv_cache_config.get(
            "free_gpu_memory_fraction", 0.9
        )

        # If not specified, max_tokens is set to the max value calculated above.
        if "max_tokens" in kv_cache_config:
            trt_kv_cache_config.max_tokens = kv_cache_config.get(
                "max_tokens", max_tokens_in_paged_kv_cache
            )

        self.cpp_executor = cpp_executor

        if "RecurrentGemma" in trt_llm_config._pretrined_config.architecture:
            # Override cpp_executor for recurrent gemma.
            print("Override cpp_executor to be True for recurrent gemma")
            self.cpp_executor = True

        super().__init__(
            trt_llm_config,
            tokenizer=TransformersTokenizer(tokenizer),
            kv_cache_config=trt_kv_cache_config,
            enable_executor=self.cpp_executor,
        )

    @property
    def max_input_len(self):
        """Get the max input length from the LLM instance."""
        return self.config.max_input_len

    @property
    def max_beam_width(self):
        """Get the max beam width from the LLM instance."""
        return self.config.max_beam_width

    def generate_tokens(
        self,
        prompts: Union[Iterable[str], Iterable[List[int]]],
        max_new_tokens: int,
        temperature: float = 1.0,
        keep_input_prompt: bool = True,
    ) -> Union[List[List[int]], List[List[List[int]]]]:
        """Generates the tokens based on the input prompts.

        Args:
            prompts: The input prompts. Could be a list of strings or token lists.
            max_new_tokens: The max output token length.
            temperature: The sampling temperature
            keep_input_prompt: Set to include input prommpts in the outputs.

        Returns:
            a list of output token lists if max_beam_width is 1 or a 3D list with shape [batch, beam, sequence_len].
        """
        assert temperature >= 0.0, "Temperature must be greater than 0.0."

        # TRT LLM acccepts temperature values only greater than 0.0
        temperature = max(temperature, 0.01)

        beam_width = self.max_beam_width
        sampling_config = self.get_default_sampling_config()
        sampling_config.max_new_tokens = max_new_tokens
        sampling_config.beam_width = beam_width
        sampling_config.temperature = [temperature]

        prompt_ids = [
            self.tokenizer.encode(prompt) if isinstance(prompt, str) else prompt
            for prompt in prompts
        ]
        outputs = self.generate(prompt_ids, sampling_config=sampling_config)

        def _process_output_token_id(output_token_id, prompt_id, with_input, keep_input_prompt):
            if with_input == keep_input_prompt:
                return output_token_id

            elif with_input:  # and not keep_input_prompt
                return output_token_id[len(prompt_id) :]

            else:  # not with_input and keep_input_prompt:
                return prompt_id + output_token_id

        with_input = not self.cpp_executor
        output_tokens = []
        for prompt_id, output in zip(prompt_ids, outputs):
            if isinstance(output.token_ids[0], list):
                # beam search
                output_token_ids = output.token_ids
            else:
                output_token_ids = [output.token_ids]

            for output_token_id in output_token_ids:
                output_tokens.append(
                    _process_output_token_id(
                        output_token_id, prompt_id, with_input, keep_input_prompt
                    )
                )

        return (
            output_tokens
            if beam_width == 1
            else [
                output_tokens[i : i + beam_width] for i in range(0, len(output_tokens), beam_width)
            ]
        )

    def generate_text(
        self,
        prompts: Union[Iterable[str], Iterable[List[int]]],
        max_new_tokens: int,
        temperature: float = 1.0,
        keep_input_prompt: bool = True,
    ) -> Union[List[str], List[List[str]]]:
        """Generates the text based on the input prompts.

        Args:
            prompts: The input prompts. Could be a list of strings or token lists.
            max_new_tokens: The max output token length.
            temperature: The sampling temperature
            keep_input_prompt: Set to include input prommpts in the outputs.

        Returns:
            a list of output text strings if max_beam_width is 1 or a 2D list with shape [batch, beam].
        """
        assert temperature >= 0.0, "Temperature must be greater than 0.0."

        # TRT LLM acccepts temperature values only greater than 0.0
        temperature = max(temperature, 0.01)

        beam_width = self.max_beam_width
        output_tokens = self.generate_tokens(
            prompts, max_new_tokens, temperature, keep_input_prompt
        )
        if beam_width == 1:
            output_text = [self.tokenizer.decode(batch) for batch in output_tokens]
        else:
            output_text = [
                [self.tokenizer.decode(beam) for beam in batch] for batch in output_tokens
            ]
        return output_text
