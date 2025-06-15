# ProjectChimera Makefile
# Development and deployment automation

.PHONY: help install test lint format clean docker-build docker-run docs

# Default target
help:
	@echo "ProjectChimera Development Commands"
	@echo "=================================="
	@echo ""
	@echo "Development:"
	@echo "  install     Install dependencies with Poetry"
	@echo "  test        Run test suite with coverage"
	@echo "  lint        Run linting and type checking"
	@echo "  format      Format code with Black and Ruff"
	@echo "  clean       Clean build artifacts and cache"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build    Build Docker images"
	@echo "  docker-run      Run with docker-compose"
	@echo "  docker-dev      Run development environment"
	@echo "  docker-test     Run tests in Docker"
	@echo ""
	@echo "Security:"
	@echo "  security        Run security checks"
	@echo "  safety          Check for known vulnerabilities"
	@echo ""
	@echo "Deployment:"
	@echo "  deploy-prod     Deploy to production"
	@echo "  deploy-staging  Deploy to staging"

# Development
install:
	@echo "📦 Installing dependencies..."
	poetry install --with dev
	poetry run pre-commit install

test:
	@echo "🧪 Running tests..."
	poetry run pytest tests/ -v --cov=project_chimera --cov-report=html --cov-report=term-missing

test-fast:
	@echo "⚡ Running fast tests..."
	poetry run pytest tests/ -x --tb=short

lint:
	@echo "🔍 Running linting..."
	poetry run ruff check project_chimera/ tests/
	poetry run black --check project_chimera/ tests/
	poetry run mypy project_chimera/

format:
	@echo "✨ Formatting code..."
	poetry run ruff check --fix project_chimera/ tests/
	poetry run black project_chimera/ tests/
	poetry run isort project_chimera/ tests/

clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete

# Security
security:
	@echo "🔒 Running security checks..."
	poetry run bandit -r project_chimera/ -f json -o bandit-report.json
	poetry run safety check --json --output safety-report.json

safety:
	@echo "🛡️ Checking for vulnerabilities..."
	poetry run safety check

# Docker
docker-build:
	@echo "🐳 Building Docker images..."
	docker build -t projectchimera/trading-bot:latest .
	docker build -t projectchimera/trading-bot:dev --target development .

docker-run:
	@echo "🚀 Starting production environment..."
	docker-compose up -d

docker-dev:
	@echo "🔧 Starting development environment..."
	docker-compose --profile development up -d

docker-monitoring:
	@echo "📊 Starting with monitoring..."
	docker-compose --profile monitoring up -d

docker-test:
	@echo "🧪 Running tests in Docker..."
	docker run --rm projectchimera/trading-bot:dev poetry run pytest

docker-stop:
	@echo "🛑 Stopping all services..."
	docker-compose down

docker-logs:
	@echo "📋 Showing logs..."
	docker-compose logs -f

docker-clean:
	@echo "🧹 Cleaning Docker resources..."
	docker-compose down -v
	docker system prune -f
	docker volume prune -f

# Build and package
build:
	@echo "📦 Building package..."
	poetry build

publish:
	@echo "🚀 Publishing to PyPI..."
	poetry publish

# Development server
dev:
	@echo "🔧 Starting development server..."
	poetry run python -m project_chimera.systems.master_trading_system --debug

# Database
db-init:
	@echo "🗄️ Initializing database..."
	poetry run python -m project_chimera.database.init

db-migrate:
	@echo "🔄 Running database migrations..."
	poetry run alembic upgrade head

db-reset:
	@echo "⚠️ Resetting database..."
	docker-compose exec postgres psql -U chimera -d chimera -c "DROP SCHEMA IF EXISTS trading CASCADE; CREATE SCHEMA trading;"

# Monitoring
logs:
	@echo "📋 Showing application logs..."
	tail -f logs/chimera.log

monitor:
	@echo "📊 Opening monitoring dashboard..."
	@echo "Grafana: http://localhost:3000"
	@echo "Prometheus: http://localhost:9090"

# Deployment
deploy-staging:
	@echo "🚀 Deploying to staging..."
	@echo "TODO: Implement staging deployment"

deploy-prod:
	@echo "🚀 Deploying to production..."
	@echo "⚠️ This will deploy to production. Are you sure? (ctrl+c to cancel)"
	@read -p "Press enter to continue..."
	@echo "TODO: Implement production deployment"

# Git hooks
pre-commit:
	@echo "✅ Running pre-commit hooks..."
	poetry run pre-commit run --all-files

# Documentation
docs:
	@echo "📚 Building documentation..."
	@echo "TODO: Add documentation build"

docs-serve:
	@echo "📖 Serving documentation..."
	@echo "TODO: Add documentation server"

# Environment setup
init: install
	@echo "🎉 Project initialization complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Copy .env.example to .env and configure your API keys"
	@echo "2. Run 'make test' to verify everything works"
	@echo "3. Run 'make dev' to start the development server"
	@echo "4. Run 'make docker-dev' to start with Docker"

# CI/CD helpers
ci-test:
	poetry run pytest tests/ --cov=project_chimera --cov-report=xml

ci-lint:
	poetry run ruff check project_chimera/ tests/
	poetry run black --check project_chimera/ tests/

ci-security:
	poetry run bandit -r project_chimera/
	poetry run safety check