#!/usr/bin/env python3
import argparse
import os
import sys

import tensorrt as trt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a TensorRT engine from an ONNX model using the TensorRT 10/11 API."
    )
    parser.add_argument("onnx_path", help="Path to the ONNX model.")
    parser.add_argument(
        "--engine",
        default=None,
        help="Output engine path. Defaults to ONNX path with .engine suffix.",
    )
    parser.add_argument(
        "--workspace-gb",
        type=float,
        default=4.0,
        help="TensorRT workspace memory limit in GiB.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable FP16 if the platform supports fast FP16.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    onnx_path = os.path.abspath(args.onnx_path)
    engine_path = args.engine
    if engine_path is None:
        engine_path = os.path.splitext(onnx_path)[0] + ".engine"
    engine_path = os.path.abspath(engine_path)

    if not os.path.exists(onnx_path):
        print(f"ONNX model not found: {onnx_path}", file=sys.stderr)
        return 1

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as model_file:
        if not parser.parse(model_file.read()):
            print("Failed to parse ONNX model:", file=sys.stderr)
            for i in range(parser.num_errors):
                print(parser.get_error(i), file=sys.stderr)
            return 1

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(args.workspace_gb * (1 << 30)),
    )

    if args.fp16:
        has_fast_fp16 = getattr(builder, "platform_has_fast_fp16", True)
        if not hasattr(trt.BuilderFlag, "FP16"):
            print("FP16 requested, but this TensorRT Python API does not expose BuilderFlag.FP16.")
        elif has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("FP16 requested, but platform does not report fast FP16 support.")

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("TensorRT engine build failed.", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, "wb") as engine_file:
        engine_file.write(serialized_engine)

    print(engine_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
