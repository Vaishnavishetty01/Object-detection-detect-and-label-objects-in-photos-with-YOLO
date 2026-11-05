"""Endpoint tests that use a deterministic fake detector instead of downloading weights."""

from __future__ import annotations

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import main
from backend.detector import Detection, DetectionResult


class FakeDetector:
    """Small test double matching ObjectDetector's public interface."""

    def __init__(self, model_path: str, image_size: int = 640, prompts=None, augment: bool = False) -> None:
        self.model_path = model_path
        self.image_size = image_size

    def load(self) -> None:
        return None

    def detect(self, image_bytes: bytes, conf: float, iou: float) -> DetectionResult:
        with Image.open(BytesIO(image_bytes)) as image:
            width, height = image.size
        return DetectionResult(
            detections=[Detection("person", 0.91, [1.0, 2.0, width / 2, height / 2])],
            width=width,
            height=height,
            inference_ms=4.2,
        )


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch):
    """Create a TestClient without requiring a real YOLO checkpoint."""

    monkeypatch.setattr(main, "ObjectDetector", FakeDetector)
    with TestClient(main.app) as test_client:
        yield test_client


def image_bytes() -> bytes:
    image = Image.new("RGB", (64, 48), color="white")
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_detect_returns_typed_detection_payload(client: TestClient) -> None:
    response = client.post(
        "/detect",
        files={"file": ("sample.png", image_bytes(), "image/png")},
        data={"conf": "0.30", "iou": "0.50"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["image_size"] == {"width": 64, "height": 48}
    assert payload["detections"][0] == {
        "label": "person",
        "confidence": 0.91,
        "bbox": [1.0, 2.0, 32.0, 24.0],
    }
    assert payload["inference_time_ms"] == 4.2


def test_detect_rejects_unsupported_type(client: TestClient) -> None:
    response = client.post(
        "/detect",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 415
    assert "Unsupported image type" in response.json()["detail"]


def test_detect_rejects_invalid_image_bytes(client: TestClient) -> None:
    response = client.post(
        "/detect",
        files={"file": ("broken.png", b"not really png", "image/png")},
    )

    assert response.status_code == 400
    assert "valid image" in response.json()["detail"]

