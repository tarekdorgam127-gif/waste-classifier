# Smart Waste Material Recognition Service

Production-oriented image classification service for recognizing waste materials from images.

The project uses **ResNet18 transfer learning** for 9-class waste classification and serves the trained model through **FastAPI** using **ONNX Runtime**.

## What the model predicts

Given an image of a waste item, the model predicts one of these 9 classes:

* Cardboard
* Food Organics
* Glass
* Metal
* Miscellaneous Trash
* Paper
* Plastic
* Textile Trash
* Vegetation

The production model achieved **87.24% test accuracy** on the RealWaste test split.

## Architecture

```text
                    ┌─────────────────────┐
                    │   RealWaste Dataset │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   PyTorch ResNet18  │
                    │   Transfer Learning │
                    └──────────┬──────────┘
                               │
                         .pth checkpoint
                               │
                               ▼
                    ┌─────────────────────┐
                    │     ONNX Export     │
                    └──────────┬──────────┘
                               │
                       .onnx + .onnx.data
                               │
                               ▼
┌──────────────┐      ┌─────────────────────┐
│    Client    │─────▶│      FastAPI        │
│ Image Upload │      │      /predict       │
└──────────────┘      └──────────┬──────────┘
                                 │
                                 ▼
                       ┌──────────────────┐
                       │  ONNX Runtime    │
                       │   CPU Inference  │
                       └────────┬─────────┘
                                │
                                ▼
                    class_name + confidence

Additional endpoints:
    /health
    /feedback

Observability:
    Structured JSON logging
    request_id
    endpoint
    latency_ms
```

## Project Structure

```text
waste-classifier/
├── configs/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── models/
│   ├── resnet18_best.pth
│   ├── resnet18.onnx
│   └── resnet18.onnx.data
├── notebooks/
├── src/
│   └── waste_classifier/
│       ├── data/
│       ├── models/
│       ├── api/
│       ├── train.py
│       ├── train_resnet.py
│       ├── evaluate.py
│       ├── evaluate_resnet.py
│       ├── export_onnx.py
│       └── test_onnx.py
├── tests/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## Quick Start

### Local setup — 3 commands

```bash
git clone <repository-url>
cd waste-classifier
pip install -e .
```

Run the API locally:

```bash
uvicorn waste_classifier.api.main:app --host 0.0.0.0 --port 8000
```

API documentation:

```text
http://localhost:8000/docs
```

## Docker

Build and run the production service:

```bash
docker compose up --build
```

The API will be available at:

```text
http://localhost:8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok"
}
```

## Prediction

Send an image to `/predict`:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/image.jpg"
```

Example response:

```json
{
  "class_name": "Plastic",
  "confidence": 0.91
}
```

## API Endpoints

### `GET /health`

Checks whether the service is running.

### `POST /predict`

Accepts an image and returns the predicted waste class and confidence.

### `POST /feedback`

Accepts the model prediction together with the actual class.

Example request:

```json
{
  "image_name": "sample.jpg",
  "predicted_class": "Plastic",
  "actual_class": "Metal"
}
```

## Model

The project compares two approaches:

### Custom CNN baseline

Test accuracy:

```text
70.55%
```

### ResNet18 Transfer Learning

Test accuracy:

```text
87.24%
```

The ResNet18 model was selected as the production candidate.

The trained PyTorch checkpoint is exported to ONNX so production inference does not require the full PyTorch training stack.

```text
PyTorch
   ↓
resnet18_best.pth
   ↓
ONNX Export
   ↓
resnet18.onnx
   ↓
ONNX Runtime
   ↓
FastAPI
```

## Testing

Run the test suite:

```bash
pytest --cov=src/waste_classifier
```

Current coverage target:

```text
≥ 80%
```

The project includes:

* Unit tests
* API tests
* FastAPI validation tests
* Dataset/split tests
* Mock-based testing

## Structured Logging

The API uses `structlog` to produce JSON logs.

Each request includes:

```json
{
  "request_id": "...",
  "endpoint": "/predict",
  "latency_ms": 12.34
}
```

This allows requests to be correlated and latency to be monitored.

## Docker Image

The production image is published on Docker Hub:

```text
tarekdorgam/waste-classifier:latest
```

Pull it with:

```bash
docker pull tarekdorgam/waste-classifier:latest
```

Run it with:

```bash
docker run -p 8000:8000 tarekdorgam/waste-classifier:latest
```

## MLOps Concepts Demonstrated

This project implements the main concepts covered in the MLOps lecture:

* Python package structure with `src/`
* `pyproject.toml`
* Type hints and modular code
* FastAPI model serving
* Pydantic schemas
* ONNX model serialization
* ONNX Runtime inference
* Multi-stage Docker builds
* Docker Compose
* Structured JSON logging
* Request correlation IDs
* Unit and API testing
* Test coverage
* Production-oriented separation between training and inference

The project represents a **Level 2 ML Pipeline maturity** approach by connecting model development with a reproducible inference and serving pipeline.
