.PHONY: help build test deploy clean

# Variables
IMAGE_NAME = fraud-detection/anomaly-detector
IMAGE_TAG = latest
NAMESPACE = fraud-detection

help:
	@echo "Available commands:"
	@echo "  build         - Build Docker image"
	@echo "  test          - Run tests"
	@echo "  deploy-local  - Deploy locally with docker-compose"
	@echo "  deploy-k8s    - Deploy to Kubernetes"
	@echo "  clean         - Clean up resources"
	@echo "  generate-load - Generate test load"
	@echo "  monitor       - Monitor system"

build:
	@echo "Building Docker image..."
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .
	@echo "✓ Image built successfully"

test:
	@echo "Running tests..."
	python -m pytest tests/ -v --cov=.
	@echo "✓ Tests completed"

deploy-local:
	@echo "Deploying locally with docker-compose..."
	docker-compose up -d
	@echo "✓ Local deployment started"
	@echo "API available at: http://localhost:8000"
	@echo "Grafana available at: http://localhost:3000 (admin/admin123)"

deploy-k8s:
	@echo "Deploying to Kubernetes..."
	./scripts/deploy.sh deploy
	@echo "✓ Kubernetes deployment completed"

clean:
	@echo "Cleaning up resources..."
	docker-compose down -v
	./scripts/deploy.sh cleanup
	docker rmi $(IMAGE_NAME):$(IMAGE_TAG) || true
	@echo "✓ Cleanup completed"

generate-load:
	@echo "Generating test load..."
	python scripts/generate_load.py --url http://localhost:8000 --rps 20 --duration 120

monitor:
	@echo "Starting system monitoring..."
	python scripts/monitor_system.py --duration 30

evaluate-models:
	@echo "Evaluating model performance..."
	python scripts/model_evaluation.py

# Development helpers
dev-setup:
	@echo "Setting up development environment..."
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	@if command -v pre-commit >/dev/null 2>&1 && [ -d .git ]; then \
		pre-commit install; \
	else \
		echo "⚠ Skipping pre-commit install (no Git repo detected)"; \
	fi
	@echo "✓ Development environment ready"

format:
	@echo "Formatting code..."
	black .
	isort .
	@echo "✓ Code formatted"

lint:
	@echo "Linting code..."
	flake8 .
	mypy anomaly_detector.py --ignore-missing-imports
	@echo "✓ Linting completed"