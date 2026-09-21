FROM python:3.14-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir fastapi "uvicorn[standard]"

RUN pip install --no-cache-dir onnxruntime pillow numpy structlog python-multipart


FROM python:3.14-slim AS runtime

WORKDIR /app

COPY --from=builder /usr/local /usr/local

COPY src ./src
COPY models/resnet18.onnx* ./models/

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "waste_classifier.api.main:app", "--host", "0.0.0.0", "--port", "8000"]