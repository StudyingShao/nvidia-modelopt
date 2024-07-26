# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Command-line entrypoint for ONNX PTQ."""

import argparse

import numpy as np

from .quantize import quantize

__all__ = ["main"]


def parse_args():
    argparser = argparse.ArgumentParser("python -m modelopt.onnx.quantization")
    group = argparser.add_mutually_exclusive_group(required=False)
    argparser.add_argument(
        "--onnx_path", required=True, type=str, help="Input onnx model without Q/DQ nodes."
    )
    argparser.add_argument(
        "--output_path",
        type=str,
        help=(
            "Output filename to save the converted ONNX model. If None, save it in the same dir as"
            " the original ONNX model with an appropriate suffix."
        ),
    )
    argparser.add_argument(
        "--calibration_method",
        type=str,
        help=(
            "Calibration method choices for fp8: {distribution (default)},"
            " int8: {entropy (default), minmax}, int4: {awq_clip (default), rtn, rtn_dq}."
        ),
    )
    group.add_argument(
        "--calibration_data_path",
        type=str,
        help="Calibration data in npz/npy format. If None, use random data for calibration.",
    )
    group.add_argument(
        "--calibration_cache_path",
        type=str,
        help="Pre-calculated activation tensor scaling factors aka calibration cache path.",
    )
    argparser.add_argument(
        "--op_types_to_quantize",
        type=str,
        default=[],
        nargs="+",
        help="A space-separated list of node types to quantize.",
    )
    argparser.add_argument(
        "--op_types_to_exclude",
        type=str,
        default=[],
        nargs="+",
        help="A space-separated list of node types to exclude from quantization.",
    )
    argparser.add_argument(
        "--nodes_to_quantize",
        type=str,
        default=[],
        nargs="+",
        help="A space-separated list of node names to quantize.",
    )
    argparser.add_argument(
        "--nodes_to_exclude",
        type=str,
        default=[],
        nargs="+",
        help="A space-separated list of node names to exclude from quantization.",
    )
    argparser.add_argument(
        "--keep_intermediate_files",
        action="store_true",
        help=(
            "If True, keep the files generated during the ONNX models' conversion/calibration."
            "Otherwise, only the converted ONNX file is kept for the user."
        ),
    )
    argparser.add_argument(
        "--use_external_data_format",
        action="store_true",
        help="If True, <MODEL_NAME>.onnx_data will be used to load and/or write weights and constants.",
    )
    argparser.add_argument(
        "--verbose",
        action="store_true",
        help="If verbose, print all the debug info.",
    )
    argparser.add_argument(
        "--quantize_mode",
        type=str,
        default="int8",
        help=("Quantization Mode. One of ['fp8', 'int8', 'int4']"),
    )
    return argparser.parse_args()


def main():
    """Command-line entrypoint for ONNX PTQ."""
    args = parse_args()
    calibration_data = None
    if args.calibration_data_path:
        calibration_data = np.load(args.calibration_data_path, allow_pickle=True)

    quantize(
        args.onnx_path,
        calibration_data=calibration_data,
        calibration_method=args.calibration_method,
        calibration_cache_path=args.calibration_cache_path,
        op_types_to_quantize=args.op_types_to_quantize,
        op_types_to_exclude=args.op_types_to_exclude,
        nodes_to_quantize=args.nodes_to_quantize,
        nodes_to_exclude=args.nodes_to_exclude,
        use_external_data_format=args.use_external_data_format,
        keep_intermediate_files=args.keep_intermediate_files,
        output_path=args.output_path,
        verbose=args.verbose,
        quantize_mode=args.quantize_mode,
    )


if __name__ == "__main__":
    main()
