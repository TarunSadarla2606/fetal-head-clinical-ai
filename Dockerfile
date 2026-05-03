FROM python:3.11-slim

# System deps required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Install CPU-only PyTorch first — avoids downloading the 3 GB CUDA build
# pip will skip torch/torchvision when it processes requirements.txt below
RUN pip install --no-cache-dir \
    "torch>=2.0.0,<3" \
    "torchvision>=0.15.0" \
    --index-url https://download.pytorch.org/whl/cpu

# Install all remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full repo (demo_subjects/ and *.pth files are in the repo root)
COPY . .

# HF Spaces proxies external traffic to port 7860
EXPOSE 7860

ENV PYTHONPATH=/code

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
