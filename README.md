# VisionLens

VisionLens is a production-minded, browser-based object detection studio. It combines a FastAPI service with Ultralytics YOLO and a framework-free HTML/CSS/JavaScript client so users can upload a photo or use a webcam, inspect detections on an interactive canvas, filter and sort results, and export the output.

## Features

- Drag-and-drop or click-to-upload with image type and 10 MB size validation.
- YOLO object detection with configurable confidence and IoU thresholds.
- Near real-time webcam mode with browser camera permissions.
- Interactive canvas boxes: hover for a tooltip, click to inspect, and use arrow keys to move between boxes.
- Class filter chips and a sortable detection table.
- Object count, inference timing, class distribution chart, and selected-object inspector.
- Annotated PNG, JSON, and CSV export.
- Persisted light/dark theme, responsive layout, keyboard focus states, toast errors, skeleton loading, and reduced-motion support.
- Optional custom training/evaluation script for YOLO-format datasets.

## Project structure

```text
backend/
  detector.py          # One-time model loading and thread-safe inference
  main.py              # FastAPI app, validation, /detect, /health, static serving
  requirements.txt
frontend/
  index.html
  css/style.css
  js/app.js
training/train.py      # Fine-tune and validate custom YOLO weights
tests/test_detect.py   # API tests with a deterministic detector double
samples/
Dockerfile
README.md
```

## Run locally

Python 3.10+ is required. Python 3.11 is recommended for the supplied Docker image.

```bash
python -m venv .venv
# Windows PowerShell
.\\.venv\\Scripts\\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000). The first start may download `yolov8s.pt` from Ultralytics. The small checkpoint is the default accuracy/speed balance; the model is loaded once during application startup and reused for requests.

To use another checkpoint, set `YOLO_MODEL` before starting the server:

```powershell
$env:YOLO_MODEL = "path/to/best.pt"
$env:YOLO_IMAGE_SIZE = "640"
$env:YOLO_AUGMENT = "false"
uvicorn backend.main:app --reload
```

The API also accepts `MAX_UPLOAD_MB` (default `10`). `YOLO_IMAGE_SIZE=768` and `YOLO_AUGMENT=true` can improve difficult-image recall at the cost of speed. CORS permits the local ports `8000` and `5500`; add another trusted origin in `backend/main.py` when needed.

### Hands and open-vocabulary objects

The standard `yolov8s.pt` checkpoint is a closed-set COCO detector. It recognizes 80 classes, including `person`, but it does not have a dedicated `hand` class. For hands and other classes outside COCO, use Ultralytics YOLO-World with text prompts:

```powershell
$env:YOLO_MODEL = "yolov8s-world.pt"
$env:YOLO_CLASSES = "person,human hand,face,phone,cup,bottle,keyboard,mouse,bag"
$env:YOLO_IMAGE_SIZE = "768"
$env:YOLO_AUGMENT = "true"
uvicorn backend.main:app --reload
```

YOLO-World uses the configured prompts without retraining, so the returned labels and confidence percentages correspond to those requested categories. For production-grade hand boxes or finger landmarks, fine-tune a hand dataset using `training/train.py` or add a dedicated hand-pose model; zero-shot prompts are convenient but are not a substitute for labeled validation.

## Docker

Build and run:

```bash
docker build -t visionlens .
docker run --rm -p 8000:8000 -e YOLO_MODEL=yolov8s.pt visionlens
```

Then open [http://localhost:8000](http://localhost:8000). For a custom model, mount it and point `YOLO_MODEL` at the container path:

```bash
docker run --rm -p 8000:8000 \
  -v "$(pwd)/weights:/models:ro" \
  -e YOLO_MODEL=/models/best.pt \
  visionlens
```

## API

### `POST /detect`

Multipart form fields:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `file` | image upload | required | PNG, JPEG, WebP, or GIF, up to 10 MB |
| `conf` | float | `0.25` | Confidence threshold, `0.05`–`0.99` |
| `iou` | float | `0.45` | NMS IoU threshold, `0.05`–`0.95` |

Example:

```bash
curl -X POST http://localhost:8000/detect \
  -F "file=@samples/street.jpg" \
  -F "conf=0.25" \
  -F "iou=0.45"
```

Response shape:

```json
{
  "detections": [
    {"label": "person", "confidence": 0.93421, "bbox": [82.4, 31.0, 401.7, 612.5]}
  ],
  "image_size": {"width": 1280, "height": 720},
  "inference_time_ms": 31.42,
  "model": "yolov8s.pt"
}
```

`GET /health` reports `ok` when the model loaded and `degraded` when the process is available but inference is not ready. Interactive OpenAPI documentation is available at `/docs`.

## Testing

The tests replace the detector with a small deterministic fake, so they do not download weights or require a GPU:

```bash
pytest -q
```

## Custom training

Ultralytics expects a dataset directory with image/label pairs and a `data.yaml` similar to:

```yaml
path: /absolute/path/to/dataset
train: images/train
val: images/val
names:
  0: helmet
  1: vest
```

Each label file uses one normalized YOLO row per object: `class_id center_x center_y width height`.

Train and evaluate:

```bash
python training/train.py \
  --data datasets/safety/data.yaml \
  --model yolov8n.pt \
  --epochs 50 \
  --imgsz 640
```

The command writes training plots and validation artifacts under `runs/visionlens/`, prints mAP50 and mAP50-95, and asks Ultralytics to generate validation plots including the confusion matrix when supported by the installed version. Copy `runs/visionlens/custom-detector/weights/best.pt` to a durable weights directory and start VisionLens with `YOLO_MODEL=/path/to/best.pt`. The custom class names are read from the checkpoint, so the API and UI automatically show them.

## How YOLO works

YOLO (You Only Look Once) is a one-stage detector. The **backbone** transforms the input image into feature maps at multiple levels, learning edges, textures, and increasingly semantic patterns. The **neck** (feature pyramid/path aggregation layers) combines coarse semantic information with fine spatial detail so both small and large objects can be localized. The **head** predicts class scores, objectness, and bounding-box coordinates for candidate locations in one forward pass.

Predictions below the **confidence threshold** are discarded. Several candidates can describe the same object, so **non-maximum suppression (NMS)** keeps the highest-confidence candidate and compares overlapping boxes using **IoU (Intersection over Union)**:

```text
IoU = area of box intersection / area of box union
```

The `iou` setting controls how much overlap is tolerated before a lower-confidence box is suppressed. Lower confidence values expose more possible objects; higher values show fewer, more certain results. The `conf` and `iou` controls in VisionLens are sent to the backend for each run.

## Screenshots

Add deployment screenshots here when available:

```text
![Upload workspace](docs/screenshots/upload-workspace.png)
![Webcam mode](docs/screenshots/webcam-mode.png)
```

## Troubleshooting

- **Model download or startup failure:** confirm the machine has internet access on first run, or download a checkpoint separately and set `YOLO_MODEL` to its path. `/health` and the browser toast expose the failure.
- **Camera is unavailable:** webcam access requires HTTPS in most deployed environments, or `localhost` during development. Check browser permissions and use the Upload tab as a fallback.
- **No boxes appear:** lower the confidence slider, confirm the image contains an object class known by the selected model, and inspect `/docs` or the browser network panel for the API response.
- **Slow inference:** use `yolov8n.pt`, set `YOLO_IMAGE_SIZE=512`, resize very large images before upload, or configure an Ultralytics-supported GPU/device. For more accuracy, use `yolov8m.pt` or a custom checkpoint and optionally set `YOLO_IMAGE_SIZE=768`.
- **Confidence is not accuracy:** the displayed percentage is the model's confidence for that individual box. A true accuracy/mAP percentage requires a labeled validation dataset; use the training script's mAP50 and mAP50-95 outputs for that measurement.
- **CORS errors from another frontend port:** add that exact trusted origin to `allow_origins` in `backend/main.py` and restart the server.

## License and model terms

This application code is provided for educational and internal use. Review the current Ultralytics and pretrained-weight licensing terms before commercial deployment, and add a project license if you publish or redistribute the source.

