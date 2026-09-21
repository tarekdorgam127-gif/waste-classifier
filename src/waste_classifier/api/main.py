from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from PIL import Image, UnidentifiedImageError 
from waste_classifier.api.schemas import PredictionResponse
from waste_classifier.api.schemas import (
    FeedbackRequest,
    PredictionResponse,
)

import time
import uuid
from fastapi import FastAPI, File, HTTPException, UploadFile
from waste_classifier.api.logging import configure_logging, logger
CLASS_NAMES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper",
    "Plastic",
    "Textile Trash",
    "Vegetation",
]
app = FastAPI(
    title="Waste Classifier API",
    version="1.0.0",
)

configure_logging()

@app.middleware("http")
async def logging_middleware(request, call_next):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    response = await call_next(request)

    latency_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "request_completed",
        request_id=request_id,
        endpoint=request.url.path,
        latency_ms=round(latency_ms, 2),
    )

    response.headers["X-Request-ID"] = request_id

    return response

MODEL_PATH = Path("models/resnet18.onnx")

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"],
)

 


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=422,
            detail="Invalid image file",
        ) from exc
    image = image.resize((224, 224))
    array = np.array(image).astype(np.float32) / 255.0

    array = (
    array - np.array([0.485, 0.456, 0.406], dtype=np.float32)
    ) / np.array([0.229, 0.224, 0.225], dtype=np.float32)

    array = np.transpose(array, (2, 0, 1))
    array = np.expand_dims(array, axis=0)

    outputs = session.run(
        ["logits"],
        {"images": array},
    )[0]

    probabilities = np.exp(outputs) / np.exp(outputs).sum(
        axis=1,
        keepdims=True,
    )

    predicted_index = int(probabilities.argmax(axis=1)[0])
    confidence = float(probabilities[0, predicted_index])

    return PredictionResponse(
        class_name=CLASS_NAMES[predicted_index],
        confidence=confidence,
    )

@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    return {
        "status": "received",
        "image_name": request.image_name,
        "predicted_class": request.predicted_class,
        "actual_class": request.actual_class,
    }

 