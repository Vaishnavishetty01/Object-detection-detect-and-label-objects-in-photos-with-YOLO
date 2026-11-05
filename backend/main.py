"""FastAPI application for VisionLens object detection."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from .detector import DetectionResult, ObjectDetector

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MODEL_PATH = os.getenv("YOLO_MODEL", "yolov8s.pt")
IMAGE_SIZE = int(os.getenv("YOLO_IMAGE_SIZE", "640"))
PROMPTS = [item.strip() for item in os.getenv("YOLO_CLASSES", "").split(",") if item.strip()]
AUGMENT = os.getenv("YOLO_AUGMENT", "false").lower() in {"1", "true", "yes", "on"}
# Limit decompressed pixel count as an additional defense against image bombs.
Image.MAX_IMAGE_PIXELS = 25_000_000


class DetectionResponseItem(BaseModel):
    """JSON representation of a detected object."""

    label: str
    confidence: float = Field(ge=0, le=1)
    bbox: list[float] = Field(min_length=4, max_length=4)


class DetectionResponse(BaseModel):
    """Response returned by POST /detect."""

    detections: list[DetectionResponseItem]
    image_size: dict[str, int]
    inference_time_ms: float
    model: str


def _result_to_response(result: DetectionResult) -> DetectionResponse:
    return DetectionResponse(
        detections=[
            DetectionResponseItem(
                label=detection.label,
                confidence=detection.confidence,
                bbox=detection.bbox,
            )
            for detection in result.detections
        ],
        image_size={"width": result.width, "height": result.height},
        inference_time_ms=result.inference_ms,
        model=MODEL_PATH,
    )


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Create one detector for the process and release its reference at exit."""

    detector = ObjectDetector(MODEL_PATH, IMAGE_SIZE, PROMPTS, AUGMENT)
    application.state.detector = None
    application.state.detector_error = None
    try:
        detector.load()
        application.state.detector = detector
    except Exception as exc:  # Keep health/API available with a useful 503.
        application.state.detector_error = str(exc)
    yield
    application.state.detector = None


app = FastAPI(
    title="VisionLens API",
    description="YOLO-powered image object detection API.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Return process and model readiness."""

    detector = getattr(app.state, "detector", None)
    if detector is None:
        return {"status": "degraded", "model": MODEL_PATH}
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/detect", response_model=DetectionResponse)
async def detect(
    file: Annotated[UploadFile, File(description="Image to analyze")],
    conf: Annotated[float, Form(ge=0.05, le=0.99)] = 0.25,
    iou: Annotated[float, Form(ge=0.05, le=0.95)] = 0.45,
) -> DetectionResponse:
    """Detect objects in an uploaded image using the configured YOLO model."""

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported image type. Please upload a PNG, JPEG, WebP, or GIF.",
        )

    image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image is too large. The maximum size is {MAX_UPLOAD_BYTES // 1024 // 1024} MB.",
        )

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from exc

    detector: ObjectDetector | None = getattr(app.state, "detector", None)
    if detector is None:
        detail = getattr(app.state, "detector_error", None) or "The detection model is unavailable."
        raise HTTPException(status_code=503, detail=detail)

    try:
        result = await run_in_threadpool(detector.detect, image_bytes, conf, iou)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Detection failed: {exc}") from exc
    return _result_to_response(result)


if FRONTEND_DIR.exists():
    app.mount("/css", StaticFiles(directory=FRONTEND_DIR / "css"), name="css")
    app.mount("/js", StaticFiles(directory=FRONTEND_DIR / "js"), name="js")


@app.get("/", include_in_schema=False)
async def frontend() -> FileResponse:
    """Serve the single-page application in local and Docker deployments."""

    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend has not been built.")
    return FileResponse(index_path)

