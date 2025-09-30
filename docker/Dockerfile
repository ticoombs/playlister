# Multi-stage build for Playlister
# Stage 1: Build Essentia from source
FROM ubuntu:22.04 AS builder

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libyaml-dev \
    libfftw3-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libavresample-dev \
    libsamplerate0-dev \
    libtag1-dev \
    libchromaprint-dev \
    python3-dev \
    python3-numpy \
    python3-yaml \
    git \
    cmake \
    pkg-config \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Build Essentia
WORKDIR /tmp
RUN git clone https://github.com/MTG/essentia.git && \
    cd essentia && \
    ./waf configure --mode=release --build-static --with-python --with-examples --with-gaia && \
    ./waf && \
    ./waf install && \
    ldconfig

# Stage 2: Runtime image
FROM ubuntu:22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-numpy \
    python3-yaml \
    libfftw3-3 \
    libavcodec58 \
    libavformat58 \
    libavutil56 \
    libavresample4 \
    libsamplerate0 \
    libtag1v5 \
    libchromaprint1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Essentia binaries and libraries from builder
COPY --from=builder /usr/local/lib/python3.*/dist-packages/essentia /usr/local/lib/python3.10/dist-packages/essentia
COPY --from=builder /usr/local/lib/libessentia.* /usr/local/lib/
COPY --from=builder /usr/local/include/essentia /usr/local/include/essentia
COPY --from=builder /usr/local/bin/essentia_* /usr/local/bin/

# Update library cache
RUN ldconfig

# Create non-root user for running the application
RUN useradd -m -u 1000 -s /bin/bash playlister

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Create necessary directories
RUN mkdir -p /data /playlists /config && \
    chown -R playlister:playlister /app /data /playlists /config

# Switch to non-root user
USER playlister

# Set Python path
ENV PYTHONPATH=/app

# Default command (can be overridden)
ENTRYPOINT ["python3", "-m", "src.cli.main"]
CMD ["--help"]