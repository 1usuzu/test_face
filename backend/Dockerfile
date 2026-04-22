FROM python:3.11-slim

WORKDIR /app

# System deps for opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

# Install backend deps first for better layer caching
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy project (backend imports ai_deepfake/, did_system/, zkp_oracle.py, etc.)
COPY . /app

EXPOSE 8000

# Run from backend directory so local module imports (e.g. zkp_oracle) resolve.
WORKDIR /app/backend
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

