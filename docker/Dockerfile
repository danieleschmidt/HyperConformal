# Multi-stage Dockerfile for HyperConformal production deployment

# Stage 1: Build environment
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    cmake \
    make \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir --user -e .

# Build C++ libraries
RUN cd cpp && \
    mkdir -p build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

# Stage 2: Production environment
FROM python:3.11-slim as production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r hyperconformal && \
    useradd -r -g hyperconformal -d /app -s /bin/bash -c "HyperConformal User" hyperconformal

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/hyperconformal/.local

# Copy built C++ libraries
COPY --from=builder /app/cpp/build/lib* /usr/local/lib/
COPY --from=builder /app/cpp/build/*.so /usr/local/lib/

# Copy source code
COPY --chown=hyperconformal:hyperconformal . .

# Update library path
RUN echo "/usr/local/lib" >> /etc/ld.so.conf && ldconfig

# Set environment variables
ENV PYTHONPATH=/app
ENV PATH=/home/hyperconformal/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# Health check
COPY docker/healthcheck.py /app/healthcheck.py
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python /app/healthcheck.py

# Switch to non-root user
USER hyperconformal

# Expose ports
EXPOSE 8080 8081

# Default command
CMD ["python", "-m", "hyperconformal.server"]

# Stage 3: Development environment (optional)
FROM production as development

# Switch back to root for development tools
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y \
    vim \
    curl \
    wget \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    jupyter \
    matplotlib \
    seaborn

# Switch back to non-root user
USER hyperconformal

# Development command
CMD ["python", "-m", "hyperconformal.dev_server"]