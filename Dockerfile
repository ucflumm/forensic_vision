FROM python:3.11-slim

# System deps (opencv + basic build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . /app

CMD ["python", "src/train_tamper_unet.py"]

