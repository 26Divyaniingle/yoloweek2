
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch to avoid unnecessary CUDA downloads.
RUN pip install --no-cache-dir \
    --default-timeout=300 \
    --retries=10 \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    --default-timeout=300 \
    --retries=10 \
    -r requirements.txt

COPY app ./app

COPY day4/experiments/runs/experiment_b_finetune/weights/best.pt ./weights/best.pt

ENV YOLO_MODEL_PATH=/app/weights/best.pt

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
