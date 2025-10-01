# Multi-stage build for Playlister
# Stage 1: Build Essentia from source
FROM ubuntu:22.04 AS builder

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies (minimize layer size)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libyaml-dev \
    libfftw3-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    libsamplerate0-dev \
    libtag1-dev \
    libchromaprint-dev \
    libeigen3-dev \
    python3-dev \
    python3-numpy \
    python3-yaml \
    git \
    ca-certificates \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create python symlink for waf
RUN ln -s /usr/bin/python3 /usr/bin/python

# Build Essentia (clean up build artifacts to reduce builder size)
WORKDIR /tmp
RUN git clone --depth 1 https://github.com/MTG/essentia.git && \
    cd essentia && \
    ./waf configure --mode=release --build-static --with-python && \
    ./waf && \
    ./waf install && \
    ldconfig && \
    cd /tmp && \
    rm -rf essentia

# Stage 2: Runtime image
FROM ubuntu:22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies only (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-numpy \
    python3-yaml \
    libfftw3-3 \
    libavcodec58 \
    libavformat58 \
    libavutil56 \
    libswresample3 \
    libsamplerate0 \
    libtag1v5 \
    libchromaprint1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy only essential Essentia Python bindings (skip example binaries and headers)
COPY --from=builder /usr/local/lib/python3/dist-packages/essentia /usr/local/lib/python3.10/dist-packages/essentia

# Update library cache
RUN ldconfig

# Create non-root user for running the application
RUN useradd -m -u 1000 -s /bin/bash playlister

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install only production dependencies (exclude dev/test packages)
RUN grep -vE '^(pytest|black|flake8|mypy|#)' requirements.txt > requirements-prod.txt && \
    pip3 install --no-cache-dir -r requirements-prod.txt && \
    rm requirements-prod.txt requirements.txt

# Copy application code
COPY src/ ./src/

# Create necessary directories and fix permissions
RUN mkdir -p /data /playlists /config && \
    chown -R playlister:playlister /app /data /playlists /config && \
    chmod -R 755 /app/src

# Switch to non-root user
USER playlister

# Set Python path
ENV PYTHONPATH=/app

# Default command (can be overridden)
ENTRYPOINT ["python3", "-m", "src.cli.main"]
CMD ["--help"]