FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

# Install CPU-only torch TRƯỚC để pip không kéo CUDA version về
RUN pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Sau đó mới install sentence-transformers (sẽ dùng torch đã có)
RUN pip install --no-cache-dir sentence-transformers fastapi uvicorn

COPY main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
