FROM python:3.11-slim-bookworm AS base

# Use args
ARG MINIMUM_BUILD
ARG USE_CUDA
ARG USE_CUDA_VER

## Basis ##
ENV ENV=prod \
    PORT=8080 \
    # pass build args to the build
    MINIMUM_BUILD=${MINIMUM_BUILD} \
    USE_CUDA_DOCKER=${USE_CUDA} \
    USE_CUDA_DOCKER_VER=${USE_CUDA_VER}

# Install GCC and build tools. 
# These are kept in the final image to enable installing packages on the fly.
RUN apt-get update && \
    apt-get install -y gcc build-essential curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY ./requirements.txt .
COPY ./requirements-minimum.txt .
RUN pip3 install uv
RUN if [ "$MINIMUM_BUILD" != "true" ]; then \
        if [ "$USE_CUDA_DOCKER" = "true" ]; then \
            pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$USE_CUDA_DOCKER_VER --no-cache-dir; \
        else \
            pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --no-cache-dir; \    
        fi \
    fi
# Remove the explicit transformers installation
RUN if [ "$MINIMUM_BUILD" = "true" ]; then \
        uv pip install --system -r requirements-minimum.txt --no-cache-dir; \
    else \
        uv pip install --system -r requirements.txt --no-cache-dir; \
    fi

# Copy the application code
COPY . .

# Expose the port
ENV HOST="0.0.0.0"
ENV PORT="8080"

# Install the missing OpenCV dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENTRYPOINT [ "bash", "start.sh" ]
