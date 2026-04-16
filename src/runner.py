# src/runner.py
import os
import json
import cv2
import numpy as np
from src.config import Config
from src.preprocessor import Preprocessor
from src.inferencer import create_inferencer
from src.inferencer.base import BaseInferencer
from src.postprocessor import Postprocessor
from src.visualizer import Visualizer
from src.comparator import Comparator


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
FHD_WIDTH, FHD_HEIGHT = 1920, 1080


class Runner:
    def __init__(self, config: Config):
        self.config = config
        self.preprocessor = Preprocessor(config.preprocess)

        self.inferencers: dict[str, BaseInferencer] = {}
        self.postprocessors: dict[str, Postprocessor] = {}

        for mc in config.models:
            if not os.path.isfile(mc.path):
                raise FileNotFoundError(f"Model file not found: {mc.path}")
            inferencer = create_inferencer(mc.path, mc)
            self.inferencers[mc.name] = inferencer

            from src.config import PostprocessConfig
            pp_config = PostprocessConfig(
                conf_threshold=mc.conf_threshold if mc.conf_threshold is not None else config.postprocess.conf_threshold,
                iou_threshold=mc.iou_threshold if mc.iou_threshold is not None else config.postprocess.iou_threshold,
                class_thresholds={**(config.postprocess.class_thresholds or {}), **(mc.class_thresholds or {})},
            )
            self.postprocessors[mc.name] = Postprocessor(pp_config)

        self.visualizer = Visualizer(
            class_names=config.class_names,
            initial_mode=config.comparison.mode,
        )

        self.comparator = Comparator(config.comparison) if len(config.models) > 1 else None
        self.multi_model = len(config.models) > 1

    def _get_model_config(self, name: str):
        for mc in self.config.models:
            if mc.name == name:
                return mc
        return None

    def _process_frame(self, image: np.ndarray) -> tuple[dict, dict, np.ndarray]:
        all_detections: dict[str, np.ndarray] = {}
        all_raw: dict[str, list[np.ndarray]] = {}
        all_times: dict[str, float] = {}

        for name, inferencer in self.inferencers.items():
            input_size = inferencer.get_input_size()
            tensor, metadata = self.preprocessor.process(image, input_size)
            raw_output, elapsed = inferencer.infer(tensor)
            detections = inferencer.postprocess_raw(raw_output)

            mc = self._get_model_config(name)
            nms_applied = mc.nms_applied if mc else False
            detections = self.postprocessors[name].process(detections, metadata, nms_applied)

            all_detections[name] = detections
            all_raw[name] = raw_output
            all_times[name] = elapsed

        if self.comparator:
            self.comparator.add_frame_result(all_raw, all_detections, all_times)

        if self.multi_model:
            vis_image = self.visualizer.render_comparison(image, all_detections)
        else:
            name = list(all_detections.keys())[0]
            vis_image = self.visualizer.draw_detections(image, all_detections[name])

        return all_detections, all_times, vis_image

    def _save_json(self, data: dict, input_path: str):
        os.makedirs(self.config.output.output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(input_path))[0]
        path = os.path.join(self.config.output.output_dir, f"{base}_results.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"JSON saved: {path}")

    def _detections_to_dict(self, dets: np.ndarray, model_name: str, time_ms: float) -> dict:
        det_list = []
        for d in dets:
            entry = {
                "bbox": [float(d[0]), float(d[1]), float(d[2]), float(d[3])],
                "confidence": float(d[4]),
                "class_id": int(d[5]),
            }
            if self.config.class_names and int(d[5]) < len(self.config.class_names):
                entry["class_name"] = self.config.class_names[int(d[5])]
            det_list.append(entry)
        return {"detections": det_list, "inference_time_ms": time_ms}

    @staticmethod
    def _resize_for_display(image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if w <= FHD_WIDTH and h <= FHD_HEIGHT:
            return image
        scale = min(FHD_WIDTH / w, FHD_HEIGHT / h)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def run_image(self, image_path: str):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        all_dets, all_times, vis_image = self._process_frame(image)

        if self.config.output.display:
            cv2.imshow("ONNX Inference", self._resize_for_display(vis_image))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if self.config.output.save_video:
            os.makedirs(self.config.output.output_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(image_path))[0]
            out_path = os.path.join(self.config.output.output_dir, f"{base}_result.png")
            cv2.imwrite(out_path, vis_image)
            print(f"Image saved: {out_path}")

        if self.config.output.save_json:
            json_data = {"input": image_path, "frames": []}
            frame_models = {}
            for name, dets in all_dets.items():
                frame_models[name] = self._detections_to_dict(dets, name, all_times[name])
            json_data["frames"].append({"frame_id": 0, "models": frame_models})
            if self.comparator:
                json_data["comparison"] = self.comparator.summarize()
            self._save_json(json_data, image_path)

    def run_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) if self.config.video.preserve_fps else 30.0

        writer = None
        if self.config.output.save_video:
            os.makedirs(self.config.output.output_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(video_path))[0]
            out_path = os.path.join(self.config.output.output_dir, f"{base}_result.mp4")
            fourcc = cv2.VideoWriter_fourcc(*self.config.video.codec)

        json_frames = [] if self.config.output.save_json else None
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_dets, all_times, vis_image = self._process_frame(frame)

            if self.config.output.save_video and writer is None:
                vh, vw = vis_image.shape[:2]
                writer = cv2.VideoWriter(out_path, fourcc, fps, (vw, vh))
            if writer is not None:
                writer.write(vis_image)

            if self.config.output.display:
                cv2.imshow("ONNX Inference (t=toggle, q=quit)", self._resize_for_display(vis_image))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("t"):
                    if writer is None:
                        self.visualizer.toggle_mode()
                    else:
                        print("Toggle disabled while recording video")

            if json_frames is not None:
                frame_models = {}
                for name, dets in all_dets.items():
                    frame_models[name] = self._detections_to_dict(dets, name, all_times[name])
                json_frames.append({"frame_id": frame_id, "models": frame_models})
            frame_id += 1

        cap.release()
        if writer is not None:
            writer.release()
            print(f"Video saved: {out_path}")
        if self.config.output.display:
            cv2.destroyAllWindows()
        if self.config.output.save_json:
            json_data = {"input": video_path, "frames": json_frames}
            if self.comparator:
                json_data["comparison"] = self.comparator.summarize()
            self._save_json(json_data, video_path)
        if self.comparator:
            summary = self.comparator.summarize()
            print("\n=== Comparison Summary ===")
            for key, val in summary.items():
                print(f"  {key}: {val}")

    def run(self, input_path: str):
        if os.path.isdir(input_path):
            files = sorted(os.listdir(input_path))
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in IMAGE_EXTS:
                    print(f"Processing image: {f}")
                    self.run_image(os.path.join(input_path, f))
                elif ext in VIDEO_EXTS:
                    print(f"Processing video: {f}")
                    self.run_video(os.path.join(input_path, f))
        else:
            ext = os.path.splitext(input_path)[1].lower()
            if ext in IMAGE_EXTS:
                self.run_image(input_path)
            elif ext in VIDEO_EXTS:
                self.run_video(input_path)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
