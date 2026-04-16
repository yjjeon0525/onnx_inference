# src/config.py
from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class CropConfig:
    enabled: bool = False
    region: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])


@dataclass
class ResizeConfig:
    method: str = "INTER_LINEAR"
    input_size: Optional[list[int]] = None


@dataclass
class PreprocessConfig:
    crop: CropConfig = field(default_factory=CropConfig)
    resize: ResizeConfig = field(default_factory=ResizeConfig)


@dataclass
class PostprocessConfig:
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    class_thresholds: dict[int, float] = field(default_factory=dict)


@dataclass
class ModelConfig:
    name: str = ""
    path: str = ""
    type: str = "yolov8"
    nms_applied: bool = False
    conf_threshold: Optional[float] = None
    iou_threshold: Optional[float] = None
    class_thresholds: Optional[dict[int, float]] = None
    strides: list[int] = field(default_factory=lambda: [8, 16, 32])
    reg_max: int = 16


@dataclass
class OutputConfig:
    save_video: bool = True
    save_json: bool = False
    display: bool = True
    output_dir: str = "output/"


@dataclass
class ComparisonConfig:
    mode: str = "overlay"
    metrics: list[str] = field(
        default_factory=lambda: [
            "cosine_similarity", "precision", "recall", "inference_time"
        ]
    )


@dataclass
class VideoConfig:
    codec: str = "mp4v"
    preserve_fps: bool = True


@dataclass
class Config:
    models: list[ModelConfig] = field(default_factory=list)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    comparison: ComparisonConfig = field(default_factory=ComparisonConfig)
    class_names: Optional[list[str]] = None
    video: VideoConfig = field(default_factory=VideoConfig)


def _parse_value(value: str):
    """Parse a CLI string value into the appropriate Python type."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "null" or value.lower() == "none":
        return None
    if "," in value:
        return [_parse_value(v.strip()) for v in value.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def apply_cli_overrides(data: dict, cli_args: list[str]) -> dict:
    """Apply --dot.notation overrides to a nested dict.
    Note: Cannot set dict values (e.g. class_thresholds) via CLI. Use YAML for those."""
    i = 0
    while i < len(cli_args):
        arg = cli_args[i]
        if arg.startswith("--") and "." in arg:
            key_path = arg[2:].split(".")
            value = cli_args[i + 1]
            parsed = _parse_value(value)
            d = data
            for k in key_path[:-1]:
                d = d.setdefault(k, {})
            d[key_path[-1]] = parsed
            i += 2
        else:
            i += 1
    return data


def _build_model_config(raw: dict) -> ModelConfig:
    return ModelConfig(
        name=raw.get("name", ""),
        path=raw.get("path", ""),
        type=raw.get("type", "yolov8"),
        nms_applied=raw.get("nms_applied", False),
        conf_threshold=raw.get("conf_threshold", None),
        iou_threshold=raw.get("iou_threshold", None),
        class_thresholds=raw.get("class_thresholds", None),
        strides=raw.get("strides", [8, 16, 32]),
        reg_max=raw.get("reg_max", 16),
    )


def load_config(yaml_path: str, cli_args: list[str] | None = None) -> Config:
    """Load YAML config and merge CLI overrides."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}

    if cli_args:
        data = apply_cli_overrides(data, cli_args)

    models = [_build_model_config(m) for m in data.get("models", [])]

    pre_raw = data.get("preprocess", {})
    crop_raw = pre_raw.get("crop", {})
    resize_raw = pre_raw.get("resize", {})
    preprocess = PreprocessConfig(
        crop=CropConfig(
            enabled=crop_raw.get("enabled", False),
            region=crop_raw.get("region", [0.0, 0.0, 1.0, 1.0]),
        ),
        resize=ResizeConfig(
            method=resize_raw.get("method", "INTER_LINEAR"),
            input_size=resize_raw.get("input_size", None),
        ),
    )

    post_raw = data.get("postprocess", {})
    postprocess = PostprocessConfig(
        conf_threshold=post_raw.get("conf_threshold", 0.25),
        iou_threshold=post_raw.get("iou_threshold", 0.45),
        class_thresholds=post_raw.get("class_thresholds", {}),
    )

    out_raw = data.get("output", {})
    output = OutputConfig(
        save_video=out_raw.get("save_video", True),
        save_json=out_raw.get("save_json", False),
        display=out_raw.get("display", True),
        output_dir=out_raw.get("output_dir", "output/"),
    )

    comp_raw = data.get("comparison", {})
    comparison = ComparisonConfig(
        mode=comp_raw.get("mode", "overlay"),
        metrics=comp_raw.get(
            "metrics",
            ["cosine_similarity", "precision", "recall", "inference_time"],
        ),
    )

    vid_raw = data.get("video", {})
    video = VideoConfig(
        codec=vid_raw.get("codec", "mp4v"),
        preserve_fps=vid_raw.get("preserve_fps", True),
    )

    return Config(
        models=models,
        preprocess=preprocess,
        postprocess=postprocess,
        output=output,
        comparison=comparison,
        class_names=data.get("class_names", None),
        video=video,
    )
