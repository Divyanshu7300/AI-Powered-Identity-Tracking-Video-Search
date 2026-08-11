import os
from pathlib import Path
from typing import Optional

import torch
from ultralytics import YOLO


cache_dir_env = os.getenv("MOT_REID_CACHE_DIR")
if cache_dir_env and cache_dir_env != "data/cache":
    cache_root = Path(cache_dir_env).expanduser().resolve()
else:
    cache_root = (Path.home() / ".cache" / "mot_reid").resolve()

try:
    cache_root.mkdir(parents=True, exist_ok=True)
    matplotlib_dir = cache_root / "matplotlib"
    xdg_dir = cache_root / "xdg"
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    xdg_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_dir))
except Exception:
    pass



class YOLODetector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.50,
        nms_iou_threshold: float = 0.45,
        imgsz: int = 640,
        min_box_area_ratio: float = 0.0008,
        max_box_area_ratio: float = 0.85,
        device: Optional[str] = None,
    ) -> None:

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = YOLO(model_path)

        self.model.fuse()

        self.model.to(self.device)

        self.conf_threshold = conf_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.imgsz = imgsz

        self.min_box_area_ratio = min_box_area_ratio
        self.max_box_area_ratio = max_box_area_ratio

        self.class_names = self.model.names

        self.use_fp16 = self.device == "cuda"

    def _is_valid_person_box(
        self,
        bbox,
        frame_width,
        frame_height,
    ) -> bool:

        x1, y1, x2, y2 = bbox

        width = max(0, x2 - x1)
        height = max(0, y2 - y1)

        # tiny garbage detections
        if width < 8 or height < 16:
            return False

        frame_area = max(
            1,
            frame_width * frame_height,
        )

        area_ratio = (
            width * height
        ) / frame_area

        # too small or absurdly huge
        if (
            area_ratio < self.min_box_area_ratio
            or area_ratio > self.max_box_area_ratio
        ):
            return False

        # human-like aspect ratio
        aspect_ratio = width / max(height, 1)

        return 0.18 <= aspect_ratio <= 1.35

    def detect_person(self, image):
        res = self.detect_person_batch([image])
        return res[0] if res else []

    def detect_person_batch(self, images):
        results = self.detect_objects_and_persons_batch(images)
        return [res["persons"] for res in results]

    def detect_objects_and_persons_batch(self, images):
        if not images:
            return []

        # Detect person (class 0) plus common personal objects
        # 1: bicycle, 24: backpack, 25: umbrella, 26: handbag, 27: tie, 28: suitcase, 39: bottle, 67: cell phone
        relevant_classes = [0, 1, 24, 25, 26, 27, 28, 39, 67]

        batch_results = self.model(
            images,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            iou=self.nms_iou_threshold,
            classes=relevant_classes,
            agnostic_nms=False,
            verbose=False,
            half=self.use_fp16,
            device=self.device,
        )

        all_results = []
        for image, results in zip(images, batch_results):
            if image is None:
                all_results.append({"persons": [], "objects": []})
                continue
            height, width = image.shape[:2]
            persons = []
            objects = []
            boxes = results.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = self.class_names.get(class_id, f"class_{class_id}")
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1 = max(0, min(width - 1, x1))
                    y1 = max(0, min(height - 1, y1))
                    x2 = max(0, min(width - 1, x2))
                    y2 = max(0, min(height - 1, y2))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    bbox = (x1, y1, x2, y2)
                    confidence = float(box.conf[0])

                    if class_id == 0:  # person
                        if self._is_valid_person_box(bbox, width, height):
                            persons.append(
                                {
                                    "bbox": bbox,
                                    "confidence": confidence,
                                    "class_id": 0,
                                }
                            )
                    else:
                        objects.append(
                            {
                                "bbox": bbox,
                                "confidence": confidence,
                                "class_id": class_id,
                                "class_name": class_name,
                            }
                        )
            all_results.append({"persons": persons, "objects": objects})
        return all_results


