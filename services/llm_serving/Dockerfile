FROM python:3.10-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn httpx redis[asyncio] sse-starlette
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
