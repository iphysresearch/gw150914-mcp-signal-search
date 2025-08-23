# GW150914 MCP Signal Search - Makefile
# Using UV for fast Python package management

.PHONY: help install-uv install install-dev install-jupyter setup clean clean-data backup-data data-info test lint format check run-server run-client demo docker-build docker-run

# Default target
help: ## Show this help message
	@echo "GW150914 MCP Signal Search - Available Commands:"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# UV Installation
install-uv: ## Install UV package manager
	@echo "📦 Installing UV package manager..."
	@if command -v uv >/dev/null 2>&1; then \
		echo "✅ UV is already installed: $$(uv --version)"; \
	else \
		echo "⬇️  Downloading and installing UV..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo "✅ UV installation complete! Please restart your shell or run: source ~/.bashrc"; \
	fi

check-uv: ## Check if UV is installed
	@if command -v uv >/dev/null 2>&1; then \
		echo "✅ UV is installed: $$(uv --version)"; \
	else \
		echo "❌ UV is not installed. Run 'make install-uv' to install it."; \
		echo "   Or install manually: curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		exit 1; \
	fi

# Installation and Setup
install: check-uv ## Install the project with production dependencies using uv
	@echo "🚀 Installing GW150914 MCP Signal Search..."
	@if [ ! -f "uv.lock" ]; then \
		echo "🔒 Creating lock file..."; \
		uv lock; \
	fi
	uv sync --frozen
	@echo "✅ Installation complete!"

install-dev: check-uv ## Install the project with development dependencies using uv
	@echo "🛠️  Installing with development dependencies..."
	@if [ ! -f "uv.lock" ]; then \
		echo "🔒 Creating lock file..."; \
		uv lock; \
	fi
	uv sync --frozen --extra dev
	@echo "✅ Development installation complete!"

install-jupyter: ## Install with Jupyter notebook support
	@echo "📓 Installing with Jupyter support..."
	uv sync --frozen --extra jupyter
	@echo "✅ Jupyter installation complete!"

install-all: ## Install with all optional dependencies
	@echo "🎯 Installing with all dependencies..."
	uv sync --frozen --all-extras
	@echo "✅ Full installation complete!"

# Pip fallback installation (if UV is not available)
install-pip: ## Install with pip (fallback option)
	@echo "🐍 Installing with pip (fallback)..."
	pip install -e ".[dev]"
	@echo "✅ Pip installation complete!"

install-pip-prod: ## Install production dependencies with pip
	@echo "🐍 Installing production dependencies with pip..."
	pip install -e .
	@echo "✅ Pip production installation complete!"

setup: install-dev ## Complete development setup
	@echo "⚙️  Setting up development environment..."
	@if [ ! -f .env ]; then \
		echo "📝 Creating .env template..."; \
		echo "# OpenAI Configuration" > .env; \
		echo "OPENAI_API_KEY=your_openai_api_key_here" >> .env; \
		echo "OPENAI_BASE_URL=https://api.openai.com/v1" >> .env; \
		echo "" >> .env; \
		echo "# MCP Server Configuration" >> .env; \
		echo "MCP_SERVER_LOG_LEVEL=INFO" >> .env; \
		echo "" >> .env; \
		echo "⚠️  Please edit .env file with your actual API keys!"; \
	fi
	@mkdir -p /tmp/gw-mcp
	@echo "✅ Development setup complete!"

# Package Management
update: ## Update all dependencies to latest versions
	@echo "🔄 Updating dependencies..."
	uv sync --upgrade
	@echo "✅ Dependencies updated!"

lock: ## Generate/update the lock file
	@echo "🔒 Generating lock file..."
	uv lock
	@echo "✅ Lock file updated!"

# Code Quality
lint: ## Run linting checks
	@echo "🔍 Running linting checks..."
	uv run flake8 mcp-client/ mcp-server/ --max-line-length=88 --extend-ignore=E203,W503
	uv run mypy mcp-client/ mcp-server/ --ignore-missing-imports
	@echo "✅ Linting complete!"

format: ## Format code with black and isort
	@echo "🎨 Formatting code..."
	uv run black mcp-client/ mcp-server/
	uv run isort mcp-client/ mcp-server/
	@echo "✅ Code formatting complete!"

format-check: ## Check code formatting without making changes
	@echo "🔍 Checking code formatting..."
	uv run black --check mcp-client/ mcp-server/
	uv run isort --check-only mcp-client/ mcp-server/
	@echo "✅ Format check complete!"

check: format-check lint ## Run all code quality checks
	@echo "✅ All checks passed!"

# Testing
test: ## Run tests
	@echo "🧪 Running tests..."
	uv run pytest tests/ -v
	@echo "✅ Tests complete!"

test-cov: ## Run tests with coverage
	@echo "🧪 Running tests with coverage..."
	uv run pytest tests/ -v --cov=mcp_client --cov=mcp_server --cov-report=html --cov-report=term
	@echo "📊 Coverage report generated in htmlcov/"
	@echo "✅ Tests with coverage complete!"

# Running the Applications
run-server: ## Start the GW analysis MCP server
	@echo "🌊 Starting GW Analysis MCP Server..."
	@if [ ! -f .env ]; then \
		echo "❌ .env file not found! Run 'make setup' first."; \
		exit 1; \
	fi
	@mkdir -p data/logs
	@timestamp=$$(date +"%Y%m%d_%H%M%S"); \
	logfile="data/logs/server_$$timestamp.log"; \
	echo "📝 Saving logs to $$logfile"; \
	uv run python mcp-server/gw_analysis_server.py 2>&1 | tee "$$logfile"

run-client: ## Start the GW optimization client (requires server path as argument)
	@echo "🔬 Starting GW Optimization Client..."
	@if [ ! -f .env ]; then \
		echo "❌ .env file not found! Run 'make setup' first."; \
		exit 1; \
	fi
	@if [ -z "$(SERVER_PATH)" ]; then \
		echo "Usage: make run-client SERVER_PATH=path/to/server"; \
		echo "Example: make run-client SERVER_PATH=mcp-server/gw_analysis_server.py"; \
		exit 1; \
	fi
	@mkdir -p data/logs
	@timestamp=$$(date +"%Y%m%d_%H%M%S"); \
	logfile="data/logs/client_$$timestamp.log"; \
	echo "📝 Saving logs to $$logfile"; \
	uv run python mcp-client/gw_optimization_client.py $(SERVER_PATH) 2>&1 | tee "$$logfile"

demo: ## Run a complete demo (server + client)
	@echo "🎭 Starting demo mode..."
	@if [ ! -f .env ]; then \
		echo "❌ .env file not found! Run 'make setup' first."; \
		exit 1; \
	fi
	@mkdir -p data/logs
	@timestamp=$$(date +"%Y%m%d_%H%M%S"); \
	server_logfile="data/logs/server_$$timestamp.log"; \
	client_logfile="data/logs/client_$$timestamp.log"; \
	demo_logfile="data/logs/demo_$$timestamp.log"; \
	echo "📝 Saving server logs to $$server_logfile"; \
	echo "📝 Saving client logs to $$client_logfile"; \
	echo "📝 Saving combined demo logs to $$demo_logfile"; \
	echo "Starting server in background..." | tee "$$demo_logfile"; \
	uv run python mcp-server/gw_analysis_server.py > "$$server_logfile" 2>&1 & \
	server_pid=$$!; \
	sleep 2; \
	echo "Starting client..." | tee -a "$$demo_logfile"; \
	uv run python mcp-client/gw_optimization_client.py mcp-server/gw_analysis_server.py 2>&1 | tee "$$client_logfile" | tee -a "$$demo_logfile"; \
	kill $$server_pid 2>/dev/null || true

# Jupyter Notebook
notebook: install-jupyter ## Start Jupyter notebook server
	@echo "📓 Starting Jupyter notebook..."
	uv run jupyter notebook --notebook-dir=. --ip=0.0.0.0 --port=8888

lab: install-jupyter ## Start JupyterLab server
	@echo "🧪 Starting JupyterLab..."
	uv run jupyter lab --notebook-dir=. --ip=0.0.0.0 --port=8888

# Data Management
clean-data: ## Clean temporary data files (keeps original strain data)
	@echo "🗂️  Cleaning temporary data files..."
	rm -rf /tmp/gw-mcp/
	rm -f data/matched_filter_records.jsonl
	@if [ -d "data/logs" ]; then \
		echo "🗑️  Removing log files..."; \
		rm -rf data/logs/; \
	fi
	@echo "✅ Temporary data cleanup complete!"

backup-data: ## Create backup of analysis results and logs
	@echo "💾 Creating data backup..."
	@mkdir -p backups
	@timestamp=$$(date +"%Y%m%d_%H%M%S"); \
	backup_created=false; \
	if [ -f "data/matched_filter_records.jsonl" ]; then \
		cp "data/matched_filter_records.jsonl" "backups/matched_filter_records_$$timestamp.jsonl"; \
		echo "✅ Analysis results backed up to backups/matched_filter_records_$$timestamp.jsonl"; \
		backup_created=true; \
	fi; \
	if [ -d "data/logs" ] && [ "$$(find data/logs -name '*.log' | wc -l)" -gt 0 ]; then \
		mkdir -p "backups/logs_$$timestamp"; \
		cp data/logs/*.log "backups/logs_$$timestamp/"; \
		log_count=$$(find data/logs -name '*.log' | wc -l); \
		echo "✅ $$log_count log files backed up to backups/logs_$$timestamp/"; \
		backup_created=true; \
	fi; \
	if [ "$$backup_created" = false ]; then \
		echo "ℹ️  No data to backup"; \
	fi

data-info: ## Show information about data files
	@echo "📊 Data Directory Information:"
	@echo ""
	@if [ -f "data/H1-1126259446-1126259478.txt" ]; then \
		h1_lines=$$(wc -l < data/H1-1126259446-1126259478.txt); \
		h1_size=$$(ls -lh data/H1-1126259446-1126259478.txt | awk '{print $$5}'); \
		echo "  📡 H1 strain data: $$h1_lines samples ($$h1_size)"; \
	else \
		echo "  ❌ H1 strain data: Not found"; \
	fi
	@if [ -f "data/L1-1126259446-1126259478.txt" ]; then \
		l1_lines=$$(wc -l < data/L1-1126259446-1126259478.txt); \
		l1_size=$$(ls -lh data/L1-1126259446-1126259478.txt | awk '{print $$5}'); \
		echo "  📡 L1 strain data: $$l1_lines samples ($$l1_size)"; \
	else \
		echo "  ❌ L1 strain data: Not found"; \
	fi
	@if [ -f "data/matched_filter_records.jsonl" ]; then \
		results_lines=$$(wc -l < data/matched_filter_records.jsonl); \
		results_size=$$(ls -lh data/matched_filter_records.jsonl | awk '{print $$5}'); \
		echo "  📈 Analysis results: $$results_lines records ($$results_size)"; \
	else \
		echo "  📈 Analysis results: No results yet"; \
	fi
	@if [ -d "data/logs" ]; then \
		log_files=$$(find data/logs -name "*.log" -type f | wc -l); \
		if [ $$log_files -gt 0 ]; then \
			total_log_size=$$(find data/logs -name "*.log" -type f -exec ls -la {} \; | awk '{sum += $$5} END {printf "%.1fK", sum/1024}'); \
			echo "  📝 Log files: $$log_files files ($$total_log_size)"; \
			echo "    - Server logs: $$(find data/logs -name "server_*.log" -type f | wc -l) files"; \
			echo "    - Client logs: $$(find data/logs -name "client_*.log" -type f | wc -l) files"; \
			echo "    - Demo logs: $$(find data/logs -name "demo_*.log" -type f | wc -l) files"; \
		else \
			echo "  📝 Log files: No logs yet"; \
		fi \
	else \
		echo "  📝 Log files: No logs yet"; \
	fi
	@if [ -d "/tmp/gw-mcp" ]; then \
		temp_files=$$(find /tmp/gw-mcp -type f | wc -l); \
		echo "  🗂️  Temporary files: $$temp_files files in /tmp/gw-mcp/"; \
	else \
		echo "  🗂️  Temporary files: None"; \
	fi

# Cleaning
clean: ## Clean up temporary files and caches
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage
	@echo "✅ Cleanup complete!"

clean-all: clean ## Clean everything including uv cache
	@echo "🧹 Deep cleaning..."
	uv cache clean
	rm -rf .venv/
	@echo "✅ Deep cleanup complete!"

# Docker Support (Optional)
docker-build: ## Build Docker image
	@echo "🐳 Building Docker image..."
	docker build -t gw150914-mcp-signal-search:latest .
	@echo "✅ Docker image built!"

docker-run: ## Run application in Docker
	@echo "🐳 Running in Docker..."
	docker run -it --rm \
		-v $(PWD):/app \
		-p 8888:8888 \
		gw150914-mcp-signal-search:latest
	@echo "✅ Docker run complete!"

# Development Utilities
dev-shell: ## Enter development shell with all dependencies
	@echo "🐚 Entering development shell..."
	uv shell

info: ## Show project and environment information
	@echo "📋 Project Information:"
	@echo "  Project: GW150914 MCP Signal Search"
	@echo "  Python: $(shell uv python --version 2>/dev/null || echo 'Not installed')"
	@echo "  UV: $(shell uv --version 2>/dev/null || echo 'Not installed')"
	@echo "  Working Directory: $(PWD)"
	@echo "  Virtual Environment: $(shell uv venv --help >/dev/null 2>&1 && echo 'Available' || echo 'Not available')"
	@echo ""
	@echo "📁 Project Structure:"
	@echo "  mcp-client/gw_optimization_client.py - GW optimization client"
	@echo "  mcp-server/gw_analysis_server.py     - GW analysis MCP server"
	@echo "  pyproject.toml                       - Project configuration"
	@echo "  Makefile                             - This file"

# Pre-commit hooks
pre-commit-install: install-dev ## Install pre-commit hooks
	@echo "🪝 Installing pre-commit hooks..."
	uv run pre-commit install
	@echo "✅ Pre-commit hooks installed!"

pre-commit-run: ## Run pre-commit hooks on all files
	@echo "🪝 Running pre-commit hooks..."
	uv run pre-commit run --all-files
	@echo "✅ Pre-commit hooks complete!"

# Quick start
quickstart: setup ## Quick start - complete setup and run demo
	@echo "🚀 Quick start - Setting up everything..."
	@make demo
	@echo "✅ Quick start complete!"

# Show UV status
uv-status: ## Show UV installation status and project info
	@echo "📊 UV Status:"
	@uv --version || (echo "❌ UV not installed. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh")
	@echo ""
	@if [ -f "pyproject.toml" ]; then \
		echo "📦 Project configured with pyproject.toml"; \
		uv tree 2>/dev/null || echo "🔄 Run 'make install' to see dependency tree"; \
	else \
		echo "❌ No pyproject.toml found"; \
	fi
