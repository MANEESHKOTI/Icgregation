FROM python:3.12-slim

# Prevent Python from writing .pyc files to disk and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first for optimal Docker layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, notebooks, and dataset
COPY src/ /app/src/
COPY notebooks/ /app/notebooks/
COPY data/ /app/data/
COPY README.md /app/README.md

# Default command runs training and outputs artifacts to /app/output
CMD ["python", "src/train.py"]

