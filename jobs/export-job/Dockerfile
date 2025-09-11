FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies including C compiler for Triton and CUDA support
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    wget \
    ca-certificates \
    git \ 
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for C compiler
ENV CC=gcc
ENV CXX=g++

# Install UV for faster package installation
RUN pip install --no-cache-dir uv

# Copy requirements and install Python dependencies using UV
COPY pyproject.toml uv.lock ./

RUN uv venv && uv sync --frozen --no-cache && uv pip list

# For llama.cpp
RUN git clone https://github.com/ggml-org/llama.cpp

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"

# Run the application
CMD ["uv", "run", "main.py"]
