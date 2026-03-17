# main.py
import argparse
import sys
from src.config import load_config
from src.runner import Runner


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="ONNX Inference Pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--input", required=True, help="Path to image, video, or directory")
    return parser.parse_known_args()


def main():
    args, extra_args = parse_args()
    config = load_config(args.config, cli_args=extra_args)
    runner = Runner(config)
    runner.run(args.input)


if __name__ == "__main__":
    main()
