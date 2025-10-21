FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies and UV
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# Set environment for venv
ENV VIRTUAL_ENV=/opt/venv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

# Copy requirements
COPY pyproject.toml uv.lock ./

# Install Python dependencies using UV
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv sync --frozen --no-cache && \
    uv pip list

# Copy application code
COPY . .

# Set environment variables
ENV PORT=8080
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV=/opt/venv

# Expose port
EXPOSE 8080

# Run the application
CMD ["python", "app.py"]