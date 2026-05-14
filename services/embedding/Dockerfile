FROM python:3.10-slim

WORKDIR /app

# 1. upgrade pip (rất quan trọng)
RUN pip install --no-cache-dir --upgrade pip

# 2. install core API deps trước
RUN pip install --no-cache-dir fastapi uvicorn

# 3. install ML stack
RUN pip install --no-cache-dir sentence-transformers

# 4. install torch riêng (CPU version)
RUN pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# 5. copy code
COPY main.py .

# 6. run server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]