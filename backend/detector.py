"""YOLO inference wrapper used by the VisionLens API."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Sequence

from PIL import Image


@dataclass(frozen=True)
class Detection:
    """A single object detection in pixel coordinates."""

    label: str
    confidence: float
    bbox: list[float]


@dataclass(frozen=True)
class DetectionResult:
    """The model output needed by the API and browser client."""

    detections: list[Detection]
    width: int
    height: int
    inference_ms: float


class ObjectDetector:
    """Load and run an Ultralytics YOLO model.

    The model is loaded once by the FastAPI lifespan handler. A lock protects
    inference because a single PyTorch model should not be invoked concurrently
    by multiple requests without an explicit concurrency strategy.
    """

    def __init__(
        self,
        model_path: str = "yolov8s.pt",
        image_size: int = 640,
        prompts: Sequence[str] | None = None,
        augment: bool = False,
    ) -> None:
        self.model_path = model_path
        self.image_size = image_size
        self.prompts = [prompt.strip() for prompt in (prompts or []) if prompt.strip()]
        self.augment = augment
        self.model: Any | None = None
        self._inference_lock = threading.Lock()

    def load(self) -> None:
        """Load model weights into memory, raising the original load error."""

        # Importing Ultralytics lazily keeps API schema/tests usable before the
        # optional, heavyweight ML runtime has been installed or downloaded.
        from ultralytics import YOLO

        self.model = YOLO(self.model_path)
        if self.prompts:
            if not hasattr(self.model, "set_classes"):
                raise RuntimeError(
                    "YOLO_CLASSES requires an open-vocabulary YOLO-World model, "
                    "for example yolov8s-world.pt."
                )
            self.model.set_classes(self.prompts)

    def detect(self, image_bytes: bytes, conf: float, iou: float) -> DetectionResult:
        """Run detection on image bytes and return normalized API-friendly data."""

        if self.model is None:
            raise RuntimeError("The YOLO model is not loaded.")

        import numpy as np

        with Image.open(BytesIO(image_bytes)) as image:
            rgb_image = image.convert("RGB")
            width, height = rgb_image.size
            image_array = np.asarray(rgb_image)

        started = time.perf_counter()
        with self._inference_lock:
            results = self.model.predict(
                source=image_array,
                conf=conf,
                iou=iou,
                imgsz=self.image_size,
                augment=self.augment,
                max_det=300,
                verbose=False,
            )
        inference_ms = (time.perf_counter() - started) * 1000

        if not results:
            return DetectionResult([], width, height, round(inference_ms, 2))

        result = results[0]
        detections: list[Detection] = []
        boxes = result.boxes
        names: dict[int, str]
        if isinstance(result.names, dict):
            names = result.names
        else:
            names = {index: name for index, name in enumerate(result.names)}

        for index in range(len(boxes)):
            coordinates = boxes.xyxy[index].detach().cpu().tolist()
            class_id = int(boxes.cls[index].detach().cpu().item())
            confidence = float(boxes.conf[index].detach().cpu().item())
            detections.append(
                Detection(
                    label=names.get(class_id, str(class_id)),
                    confidence=round(confidence, 5),
                    bbox=[round(float(value), 2) for value in coordinates],
                )
            )

        return DetectionResult(
            detections=detections,
            width=width,
            height=height,
            inference_ms=round(inference_ms, 2),
        )

