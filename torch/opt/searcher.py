# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Standard interface to implement a searcher algorithm.

A searcher is useful whenever we want to search/optimize over a set of hyperparameters in the model.
Searchers are usually used in conjunction with a mode, which can define a search space via its
entrypoints, i.e., convert the model into a search space. The searcher then optimizes over this
search space.
"""

import copy
import os
from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Tuple,
    Union,
    final,
)

import torch
import torch.nn as nn

from modelopt.torch.utils import distributed as dist
from modelopt.torch.utils import (
    no_stdout,
    run_forward_loop,
)

LimitsTuple = Tuple[float, float]
ConstraintsDict = Dict[str, Union[str, float, None]]
Deployment = Dict[str, str]
ForwardLoop = Callable[[nn.Module], None]
ScoreFunc = Callable[[nn.Module], float]

SearchConfig = Dict[str, Any]  # config dict for searcher
SearchStateDict = Dict[str, Any]  # state dict for searcher

__all__ = ["BaseSearcher"]


class BaseSearcher(ABC):
    """A basic search interface that can be used to search/optimize a model.

    The base interface supports basic features like setting up a search, checkpointing, and
    loading logic and defines a minimal workflow to follow.
    """

    model: nn.Module
    config: SearchConfig
    constraints: ConstraintsDict
    deployment: Optional[Deployment]
    dummy_input: Union[Any, Tuple]
    forward_loop: Optional[ForwardLoop]

    @final
    def __init__(self) -> None:
        """We don't allow to override __init__ method."""
        super().__init__()

    # TODO: see if we really want to keep all the config here.
    @property
    def default_search_config(self) -> SearchConfig:
        """Get the default config for the searcher."""
        return {
            "checkpoint": None,
            "verbose": dist.is_master(),
            "forward_loop": None,
            "data_loader": None,
            "collect_func": None,
            "max_iter_data_loader": None,
            "score_func": None,
            "loss_func": None,
            "deployment": None,
        }

    @property
    @abstractmethod
    def default_state_dict(self) -> SearchStateDict:
        """Return default state dict."""

    def sanitize_search_config(self, config: Optional[SearchConfig]) -> SearchConfig:
        """Sanitize the search config dict."""
        # supply with defaults (for verbose we wanna make sure it's on master only)
        config = {**self.default_search_config, **(config or {})}
        config["verbose"] = config["verbose"] and self.default_search_config["verbose"]

        # sanity checks
        assert (
            config.keys() == self.default_search_config.keys()
        ), f"Unexpected config keys: {config.keys() - self.default_search_config.keys()}"
        assert config["score_func"] is not None, "Please provide `score_func`!"

        # return
        return config

    # TODO: double-check if we want all these args here.
    @final
    def search(
        self,
        model: nn.Module,
        constraints: ConstraintsDict,
        dummy_input: Union[Any, Tuple],
        config: Optional[SearchConfig] = None,
    ) -> SearchStateDict:
        """Search a given prunable model for the best sub-net and return the search model.

        The best sub-net maximizes the score given by ``score_func`` while satisfying the
        ``constraints``.

        Args:
            model: The converted model to be searched.
            constraints: The dictionary from constraint name to upper bound the searched model has
                to satisfy.
            dummy_input: Arguments of ``model.forward()``. This is used for exporting and
                calculating inference-based metrics, such as latency/FLOPs. The format of
                ``dummy_inputs`` follows the convention of the ``args`` argument in
                `torch.onnx.export <https://pytorch.org/docs/stable/onnx.html#torch.onnx.export>`_.
            config: Additional optional arguments to configure the search.

        Returns: A tuple (subnet, state_dict) where
            subnet is the searched subnet (nn.Module), which can be used for subsequent tasks like
            fine-tuning, state_dict contains the history and detailed stats of the search procedure.
        """
        # check model train state
        is_training = model.training

        # reset the search
        self.reset_search()

        # update and initialize searcher
        self.model = model
        self.config = self.sanitize_search_config(config)
        self.deployment = self.config["deployment"]

        self.constraints = constraints
        self.dummy_input = dummy_input
        self.forward_loop = self.construct_forward_loop(silent=not self.config["verbose"])

        # load checkpoint if it exists
        self.load_search_checkpoint()

        # run initial step and sanity checks before the search
        self.before_search()

        # run actual search
        self.run_search()

        # run clean-up steps after search
        self.after_search()

        # make sure model is in original state
        model.train(is_training)

        # return the config for the best result
        return self.best

    def reset_search(self) -> None:
        """Reset search at the beginning."""
        # reset self.best where we store results
        self.best: SearchStateDict = {}

        # reset state dict (do it afterwards in case best is in state_dict)
        for key, val in self.default_state_dict.items():
            setattr(self, key, copy.deepcopy(val))

    def before_search(self) -> None:
        """Optional pre-processing steps before the search."""
        pass

    @abstractmethod
    def run_search(self) -> None:
        """Run actual search."""

    def after_search(self) -> None:
        """Optional post-processing steps after the search."""
        pass

    @property
    def has_score(self) -> bool:
        """Check if the model has a score function."""
        return self.config["score_func"] is not None

    def eval_score(self, silent=True) -> float:
        """Optionally silent evaluation of the score function."""
        assert self.has_score, "Please provide `score_func`!"
        score_func: ScoreFunc = self.config["score_func"]
        with no_stdout() if silent else nullcontext():
            return float(score_func(self.model))

    def construct_forward_loop(self, silent=True) -> Optional[ForwardLoop]:
        """Get runnable forward loop on the model using the provided configs."""
        # check config
        data_loader = self.config["data_loader"]
        forward_loop = self.config["forward_loop"]
        assert None in [data_loader, forward_loop], "Only provide `data_loader` or `forward_loop`!"

        # check trivial case
        if not (data_loader or forward_loop):
            return None

        def forward_loop_with_silence_check(m: nn.Module) -> None:
            with no_stdout() if silent else nullcontext():
                if data_loader is not None:
                    run_forward_loop(
                        m,
                        data_loader=data_loader,
                        max_iters=self.config["max_iter_data_loader"],
                        collect_func=self.config["collect_func"],
                    )
                elif forward_loop is not None:
                    forward_loop(m)

        return forward_loop_with_silence_check

    @final
    def state_dict(self) -> SearchStateDict:
        """The state dictionary that can be stored/loaded."""
        return {key: getattr(self, key) for key in self.default_state_dict}

    def load_search_checkpoint(self) -> bool:
        """Load function for search checkpoint returning indicator whether checkpoint was loaded."""
        # check if checkpoint exists
        checkpoint: Optional[str] = self.config["checkpoint"]
        if checkpoint is None or not os.path.exists(checkpoint):
            return False

        # iterate through state dict and load keys
        print(f"Loading searcher state from {checkpoint}...")
        state_dict = torch.load(checkpoint)
        assert state_dict.keys() == self.state_dict().keys(), "Keys in checkpoint don't match!"
        for key, state in state_dict.items():
            setattr(self, key, state)
        return True

    def save_search_checkpoint(self) -> None:
        """Save function for search checkpoint."""
        # check if save requirements are satisfied
        checkpoint: Optional[str] = self.config["checkpoint"]
        if checkpoint is None or not dist.is_master():
            return

        # save state dict
        save_dirname, _ = os.path.split(checkpoint)
        if save_dirname:
            os.makedirs(save_dirname, exist_ok=True)
        torch.save(self.state_dict(), checkpoint)
