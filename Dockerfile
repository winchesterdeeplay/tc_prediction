FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi==0.116.1 \
    uvicorn[standard]==0.35.0 \
    pydantic==2.11.7 \
    pydantic-settings==2.10.1 \
    pandas==2.3.1 \
    onnx==1.18.0 \
    onnxruntime==1.22.1 \
    fsspec==2025.7.0

COPY run_api.py ./
COPY api/ ./api/
COPY inference/ ./inference/
COPY core/ ./core/
COPY data_processing/ ./data_processing/
COPY weights/ ./weights/

EXPOSE 8000

CMD ["python", "run_api.py"] 