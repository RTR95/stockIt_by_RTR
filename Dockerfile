# Stocron by RTR - Docker Image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Install system dependencies
# git is needed for some pip installs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
# We install torch first to get the nvidia libs (on x86_64)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir streamlit

# Install llama-cpp-python with GPU support if x86_64
# We use the pre-built wheel for CUDA 12.1 on Linux x86_64
# On ARM64 (Mac), we install the standard wheel (CPU/Metal)
# Note: Metal support in Docker is limited, so this runs on CPU for Mac
RUN if [ "$(uname -m)" = "x86_64" ]; then \
        echo "Detected x86_64 (Linux/Windows). Installing llama-cpp-python with CUDA support..."; \
        pip install --no-cache-dir llama-cpp-python \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121; \
    else \
        echo "Detected ARM64 (Mac/Other). Installing standard llama-cpp-python (CPU)..."; \
        pip install --no-cache-dir llama-cpp-python; \
    fi

# Set LD_LIBRARY_PATH to include torch's nvidia libs (for x86_64)
# This allows llama-cpp-python to find cuBLAS etc. without system CUDA
# These paths exist only if torch installed the nvidia wheels (which it does on x86_64 Linux)
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib

# Optional: do not fail build if this install fails
RUN pip install --no-cache-dir jugaad-data --no-deps || true

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/cache/prices data/cache/financials data/cache/metadata data/db models

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the application
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
