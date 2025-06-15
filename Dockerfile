# ProjectChimera Trading Bot - Multi-Stage Docker Build
# Stage 1: Builder
FROM python:3.9-slim as builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_HOME="/opt/poetry"
ENV POETRY_VERSION=2.1.3
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/opt/poetry/.cache \
    POETRY_HOME=/opt/poetry

# Set work directory
WORKDIR /app

# Copy Poetry files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --only=main --no-dev && \
    poetry run pip install --upgrade pip

# Stage 2: Runtime
FROM python:3.9-slim as runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r chimera && useradd -r -g chimera chimera

# Set work directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY project_chimera/ ./project_chimera/
COPY README.md ./

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/config && \
    chown -R chimera:chimera /app

# Set Python path
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import project_chimera; print('✅ Health check passed')" || exit 1

# Switch to non-root user
USER chimera

# Expose port for monitoring/API
EXPOSE 8000

# Default command
CMD ["python", "-m", "project_chimera.systems.master_trading_system"]

# Development stage
FROM runtime as development

USER root

# Install development dependencies
COPY --from=builder /app/.venv /app/.venv
RUN /app/.venv/bin/pip install pytest pytest-cov black ruff mypy

# Install development tools
RUN apt-get update && apt-get install -y \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

USER chimera

# Development command
CMD ["python", "-m", "project_chimera.systems.master_trading_system", "--debug"]