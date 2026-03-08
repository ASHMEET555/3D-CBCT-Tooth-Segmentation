FROM nvcr.io/nvidia/pytorch:23.12-py3

LABEL maintainer="ashmeet@example.com"
LABEL description="3D CBCT Tooth Segmentation Pipeline"
LABEL version="1.0.0"

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install project as package
RUN pip install -e .

# Set environment variables
ENV PYTHONPATH=/workspace
ENV nnUNet_raw=/workspace/data/raw
ENV nnUNet_preprocessed=/workspace/data/processed
ENV nnUNet_results=/workspace/weights

# Create necessary directories
RUN mkdir -p /workspace/data/raw \
             /workspace/data/processed \
             /workspace/data/splits \
             /workspace/weights \
             /workspace/results

# Expose port for any web services
EXPOSE 8080

# Default command: show help
CMD ["python", "src/inference/predict.py", "--help"]
