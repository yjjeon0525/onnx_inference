# main.py
import argparse
import sys
from src.config import load_config
from src.runner import Runner


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="ONNX Inference Pipeline")

    # Mode 1: Inference
    parser.add_argument("--config", help="Path to YAML config file")
    parser.add_argument("--input", help="Path to image, video, or directory")

    # Mode 2: Static model comparison
    parser.add_argument(
        "--compare-models", nargs=2, metavar=("MODEL_A", "MODEL_B"),
        help="Compare two ONNX model files (structure + weights)",
    )
    parser.add_argument(
        "--output-json", metavar="PATH",
        help="Save comparison report as JSON (used with --compare-models)",
    )

    return parser.parse_known_args()


def run_compare(args: argparse.Namespace):
    from src.model_analyzer import OnnxModelAnalyzer

    path_a, path_b = args.compare_models
    analyzer = OnnxModelAnalyzer(path_a, path_b)
    report = analyzer.analyze()
    analyzer.print_report(report)

    if args.output_json:
        analyzer.save_report(report, args.output_json)


def run_inference(args: argparse.Namespace, extra_args: list[str]):
    if not args.config or not args.input:
        print("Error: --config and --input are required for inference mode.")
        sys.exit(1)
    config = load_config(args.config, cli_args=extra_args)
    runner = Runner(config)
    runner.run(args.input)


def main():
    args, extra_args = parse_args()

    if args.compare_models:
        run_compare(args)
    else:
        run_inference(args, extra_args)


if __name__ == "__main__":
    main()
