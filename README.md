# Smart Waste Material Recognition Service

Production-oriented image classification service for recognizing waste materials from images.

The project uses **ResNet18 transfer learning** for 9-class waste classification and serves the trained model through **FastAPI** using **ONNX Runtime**.

The project also demonstrates practical MLOps concepts including **MLflow experiment tracking, Docker, Docker Hub, automated testing with GitHub Actions, structured logging, and model export to ONNX**.

---

## What the Model Predicts

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

The current ResNet18 model achieved:

**Test Accuracy: 88.78%**

on the held-out RealWaste test split.

---

## Dataset

The project uses the **RealWaste** dataset from the **UCI Machine Learning Repository**.

Dataset characteristics:

* Total images: **4,752**
* Number of classes: **9**
* Original image size: approximately **524 × 524**
* Train split: **3,326 images**
* Validation split: **713 images**
* Test split: **713 images**
* Split strategy: **stratified**

### Class Distribution

| Class               |    Images |
| ------------------- | --------: |
| Cardboard           |       461 |
| Food Organics       |       411 |
| Glass               |       420 |
| Metal               |       790 |
| Miscellaneous Trash |       495 |
| Paper               |       500 |
| Plastic             |       921 |
| Textile Trash       |       318 |
| Vegetation          |       436 |
| **Total**           | **4,752** |

The dataset is **not perfectly balanced**. Plastic and Metal have substantially more samples than Textile Trash and Food Organics.

The current training pipeline does **not use weighted loss or oversampling**. Instead, the dataset is split using stratification so that all classes are represented across train, validation, and test sets.

Dataset attribution: **RealWaste — UCI Machine Learning Repository**.

The dataset is distributed under **CC BY 4.0**.

---

## Model Development

Two approaches were evaluated.

### Custom CNN Baseline

Test accuracy:

```text
70.55%
```

### ResNet18 Transfer Learning

Test accuracy:

```text
88.78%
```

The ResNet18 model was selected as the current production candidate based on its validation and test performance.

### Model Architecture

```text
Image
  │
  ▼
Resize 224 × 224
  │
  ▼
ResNet18
Transfer Learning
  │
  ▼
9-class classifier
  │
  ▼
Class probabilities
  │
  ▼
Predicted class + confidence
```

---

## Training Configuration

The current ResNet18 training configuration is:

| Parameter         | Value                              |
| ----------------- | ---------------------------------- |
| Model             | ResNet18                           |
| Initialization    | ImageNet pretrained weights        |
| Optimizer         | Adam                               |
| Learning rate     | 0.0001                             |
| Batch size        | 32                                 |
| Epochs            | 5                                  |
| Loss              | Cross Entropy Loss                 |
| Input size        | 224 × 224                          |
| Number of classes | 9                                  |
| Device            | NVIDIA GeForce RTX 4050 Laptop GPU |
| Framework         | PyTorch                            |

### Data Augmentation

Training images use:

* Resize to 224 × 224
* Random horizontal flip
* Random rotation up to 10°
* ImageNet normalization

Validation and test images use resizing and ImageNet normalization without random augmentation.

---

## Training Results

The best validation checkpoint was selected using **validation loss**.

Example training progression from the current run:

| Epoch | Train Accuracy | Validation Accuracy | Validation Loss |
| ----: | -------------: | ------------------: | --------------: |
|     3 |         93.96% |              87.66% |          0.3382 |
|     4 |         96.60% |              89.34% |          0.3213 |
|     5 |         97.26% |              85.27% |          0.3849 |

The best checkpoint was saved from the validation-loss criterion.

Final test results:

```text
Test Loss:     0.3570
Test Accuracy: 0.8878
```

---

## Model Error Analysis

The ResNet18 model achieved good overall performance, but some classes remain difficult to distinguish.

The current classification report showed:

| Class               | Precision | Recall |     F1 |
| ------------------- | --------: | -----: | -----: |
| Cardboard           |    0.8800 | 0.9565 | 0.9167 |
| Food Organics       |    0.9815 | 0.8548 | 0.9138 |
| Glass               |    0.8710 | 0.8571 | 0.8640 |
| Metal               |    0.7730 | 0.9237 | 0.8417 |
| Miscellaneous Trash |    0.7632 | 0.7733 | 0.7682 |
| Paper               |    0.9211 | 0.9333 | 0.9272 |
| Plastic             |    0.9160 | 0.7899 | 0.8482 |
| Textile Trash       |    0.9091 | 0.8333 | 0.8696 |
| Vegetation          |    0.9545 | 0.9692 | 0.9618 |

Some of the most common errors occur between visually similar materials, particularly **Metal and Plastic**.

An error-analysis visualization is stored locally as:

```text
models/error_analysis.png
```

---

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
                         .onnx model
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

---

## Project Structure

```text
waste-classifier/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
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
├── uv.lock
└── README.md
```

Large datasets, trained checkpoints, and ONNX artifacts are intentionally excluded from Git using `.gitignore`.

---

## Local Setup

Clone the repository and install the project dependencies:

```bash
git clone <repository-url>
cd waste-classifier
pip install -e .
```

Run the API locally:

```bash
uvicorn waste_classifier.api.main:app --host 0.0.0.0 --port 8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

---

## Docker

The production API uses a lightweight Python runtime containing:

* FastAPI
* Uvicorn
* ONNX Runtime
* NumPy
* Pillow
* structlog

The production container does **not require the full PyTorch training stack**.

Build and run:

```bash
docker compose up --build
```

The API will be available on:

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

---

## Prediction API

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

The API validates the uploaded image and returns HTTP `422` for invalid or corrupted image files.

---

## API Endpoints

### `GET /health`

Checks whether the service is running.

### `POST /predict`

Accepts an image and returns:

* predicted class
* confidence score

### `POST /feedback`

Accepts model prediction together with the actual class.

Example:

```json
{
  "image_name": "sample.jpg",
  "predicted_class": "Plastic",
  "actual_class": "Metal"
}
```

The API validates the predicted and actual classes against the supported 9-class label set.

---

## ONNX Deployment

The trained PyTorch checkpoint is exported to ONNX:

```text
PyTorch
   │
   ▼
resnet18_best.pth
   │
   ▼
ONNX Export
   │
   ▼
resnet18.onnx
   │
   ▼
ONNX Runtime
   │
   ▼
FastAPI
```

The ONNX model was validated using ONNX Runtime against the PyTorch model.

The maximum observed output difference during validation was approximately:

```text
0.00000110
```

The predicted class matched between PyTorch and ONNX Runtime.

---

## MLflow Experiment Tracking

The project uses **MLflow 3.16.1** for experiment tracking.

The training pipeline logs:

* Model configuration
* Number of classes
* Batch size
* Learning rate
* Number of epochs
* Optimizer
* Device
* Image size
* Training loss
* Training accuracy
* Validation loss
* Validation accuracy
* Best validation loss
* Test loss
* Test accuracy
* Model checkpoint artifact

Current experiment:

```text
waste-classifier-resnet18
```

Current tracked run:

```text
Run ID:
7f34f6c6aee04730b16f8f9ea1e92b31
```

Final tracked test accuracy:

```text
88.78%
```

Run the MLflow UI locally:

```bash
mlflow ui --port 5000
```

Then open:

```text
http://127.0.0.1:5000
```

MLflow provides experiment history and makes future model runs easier to compare and reproduce.

---

## Testing

Run the test suite:

```bash
pytest -q
```

Current result:

```text
8 passed
```

Coverage can be generated with:

```bash
pytest --cov=src/waste_classifier
```

The project includes tests for:

* Dataset splitting
* Dataset loading
* Model forward pass
* API health endpoint
* Prediction endpoint
* Invalid image handling
* Pydantic response validation
* Class validation

---

## Structured Logging

The API uses `structlog` to produce JSON logs.

Each request includes information such as:

```json
{
  "request_id": "...",
  "endpoint": "/predict",
  "latency_ms": 12.34
}
```

The request ID is also returned through:

```text
X-Request-ID
```

This makes it possible to correlate requests and inspect inference latency.

---

## Docker Hub

The production image is published on Docker Hub as:

```text
tarekdorgam/waste-classifier:latest
```

Pull the image:

```bash
docker pull tarekdorgam/waste-classifier:latest
```

Run it:

```bash
docker run -p 8000:8000 tarekdorgam/waste-classifier:latest
```

The Docker image contains the ONNX production model and inference dependencies without the training environment.

---

## CI/CD

GitHub Actions is used for automated project validation.

The CI workflow:

1. Checks out the repository.
2. Installs Python 3.14.
3. Installs the required testing dependencies.
4. Runs the complete pytest suite.

The goal is to prevent broken code from being merged or pushed unnoticed.

The Docker image has also been built and published manually to Docker Hub.

The current GitHub workflow focuses on automated testing rather than automatically publishing a new Docker image on every commit.

---

## MLOps Concepts Demonstrated

This project demonstrates the following MLOps concepts:

* Python package structure with `src/`
* `pyproject.toml`
* Type hints and modular code
* Stratified dataset splitting
* Data augmentation
* Transfer learning
* Model evaluation
* Error analysis
* FastAPI model serving
* Pydantic validation
* ONNX model serialization
* ONNX Runtime inference
* Production-oriented separation between training and inference
* Docker
* Docker Compose
* Docker Hub
* Structured JSON logging
* Request correlation IDs
* Unit and API testing
* Test coverage
* MLflow experiment tracking
* Git version control
* GitHub Actions CI

---

## Model Versioning

The project currently uses a combination of:

* Git commits for source-code versioning
* MLflow runs for experiment tracking
* Named model artifacts such as `resnet18_best.pth`
* Docker image tags for deployment artifacts

The current production Docker image is:

```text
tarekdorgam/waste-classifier:latest
```

For a larger production system, immutable version tags such as Git commit SHA or semantic versions should replace relying only on `latest`.

---

## Current Limitations

The project is production-oriented, but it is not intended to represent a fully hardened public API.

Current limitations include:

### Authentication

The API currently has no authentication or API-key mechanism.

For public deployment, authentication and rate limiting should be added.

### Upload Limits

Invalid image files are rejected, but there is currently no explicit application-level maximum upload size such as a 5 MB or 10 MB limit.

### Monitoring

Current logging records request IDs, endpoints, and latency.

The system does not yet provide:

* P95/P99 latency dashboards
* Production drift monitoring
* Automated confidence alerts
* Feedback-based accuracy monitoring
* Prometheus/Grafana metrics

### Explainability

The API currently returns the predicted class and confidence only.

Explainability techniques such as:

* Top-K predictions
* Grad-CAM
* visual explanations

are not currently implemented.

### Model Information Endpoint

A dedicated `/model-info` endpoint is not currently implemented.

MLflow is currently used to track experiment information and model artifacts.

### Model Versioning

The current Docker Hub image uses the `latest` tag.

A future production setup should use immutable model/image versions, for example:

```text
waste-classifier:1.0.0
waste-classifier:<git-sha>
```

---

## Future Improvements

Potential next improvements include:

* Add API authentication
* Add request size limits
* Add rate limiting
* Add `/model-info`
* Add immutable Docker image versioning
* Add P95/P99 latency monitoring
* Add Prometheus/Grafana metrics
* Add confidence monitoring
* Add top-3 predictions
* Add Grad-CAM explainability
* Add class-balanced loss or sampling experiments
* Add automated Docker publishing through GitHub Actions
* Add DVC for dataset versioning
* Add automated model promotion based on evaluation metrics

---

## Summary

The project provides a complete machine-learning development and serving workflow:

```text
RealWaste Dataset
       │
       ▼
Data Splitting
       │
       ▼
Data Augmentation
       │
       ▼
ResNet18 Training
       │
       ├──────────────► MLflow Tracking
       │
       ▼
Model Evaluation
       │
       ▼
Best PyTorch Checkpoint
       │
       ▼
ONNX Export
       │
       ▼
ONNX Runtime
       │
       ▼
FastAPI
       │
       ▼
Docker
       │
       ▼
Docker Hub
       │
       ▼
GitHub + GitHub Actions CI
```

Current model performance:

```text
Custom CNN:       70.55%
ResNet18:         88.78%
```

The resulting system connects **model development, experiment tracking, evaluation, model serialization, API serving, containerization, testing, and CI** into one reproducible MLOps project.
