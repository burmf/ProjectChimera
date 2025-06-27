# ProjectChimera Trading Bot - Optimized Multi-Stage Docker Build
# Stage 1: Builder
FROM python:3.11-slim as builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set work directory
WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml ./
COPY src/ ./src/
COPY README.md LICENSE ./

# Install dependencies using pip directly (faster than Poetry for production)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e .

# Stage 2: Runtime
FROM python:3.11-slim as runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r chimera && useradd -r -g chimera -m chimera

# Set work directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source code
COPY src/ ./src/
COPY README.md LICENSE ./

# Create necessary directories and set permissions
RUN mkdir -p /app/data /app/logs /app/config && \
    chown -R chimera:chimera /app

# Set environment variables
ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from src.project_chimera.orchestrator import TradingOrchestrator; print('✅ Health check passed')" || exit 1

# Switch to non-root user
USER chimera

# Expose ports for monitoring and health checks
EXPOSE 8000 8001

# Default command
CMD ["python", "-m", "src.project_chimera.orchestrator"]

# Stage 3: Development
FROM builder as development

# Install development dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev,test]"

# Install development tools
RUN apt-get update && apt-get install -y \
    git \
    vim \
    nano \
    htop \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy all source code for development
COPY . .

# Create non-root user for development
RUN groupadd -r chimera && useradd -r -g chimera -m chimera && \
    chown -R chimera:chimera /app

USER chimera

# Set environment variables for development
ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ENV=development

# Expose ports for debugging and development servers
EXPOSE 8000 8001 8501 8888

# Development command with auto-reload
CMD ["python", "-m", "src.project_chimera.orchestrator", "--verbose"]

# Stage 4: Testing
FROM development as testing

# Run tests during build (optional)
USER root
RUN pip install --no-cache-dir pytest pytest-asyncio pytest-httpx pytest-cov

USER chimera

# Copy test configuration
COPY tests/ ./tests/
COPY pyproject.toml ./

# Set test environment
ENV ENV=test

# Default test command
CMD ["pytest", "tests/", "-v", "--cov=src/project_chimera", "--cov-report=html", "--cov-report=term-missing"]