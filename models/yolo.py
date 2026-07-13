import os
from pathlib import Path
from typing import Optional

import cv2
import torch
from ultralytics import YOLO


cache_root = Path(
    os.getenv(
        "MOT_REID_CACHE_DIR",
        "/tmp/mot-reid-cache",
    )
)

cache_root.mkdir(
    parents=True,
    exist_ok=True,
)

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(cache_root / "matplotlib"),
)

os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(cache_root / "xdg"),
)


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

        if image is None:
            raise ValueError("Image is None")

        height, width = image.shape[:2]

        results = self.model(
            image,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            iou=self.nms_iou_threshold,
            classes=[0],  # person only
            agnostic_nms=False,
            verbose=False,
            half=self.use_fp16,
            device=self.device,
        )[0]
        
        persons = []
        boxes = results.boxes
        if boxes is None:
            return persons
        for box in boxes:
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0],
            )

            # clamp bbox
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width - 1, x2))
            y2 = max(0, min(height - 1, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            bbox = (x1, y1, x2, y2)

            if not self._is_valid_person_box(
                bbox,
                width,
                height,
            ):
                continue

            persons.append(
                {
                    "bbox": bbox,
                    "confidence": float(box.conf[0]),
                    "class_id": 0,
                }
            )

        return persons
