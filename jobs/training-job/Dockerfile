FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies including C compiler for Triton and CUDA support
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for C compiler
ENV CC=gcc
ENV CXX=g++

# Install UV for faster package installation
RUN pip install --no-cache-dir uv

# Copy requirements and install Python dependencies using UV
COPY pyproject.toml uv.lock ./

# Set environment for venv
ENV VIRTUAL_ENV=/opt/venv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

# Create virtual environment and install dependencies
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv sync --frozen --no-cache && \
    uv pip list

# Copy application code
COPY . .

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV=/opt/venv

# Run the application
CMD ["python", "main.py"]
