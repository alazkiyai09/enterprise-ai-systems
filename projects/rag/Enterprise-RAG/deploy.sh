#!/bin/bash
# ============================================================
# Enterprise-RAG: Quick Deploy Script
# ============================================================

set -e

echo "================================================"
echo "Enterprise-RAG Docker Deployment"
echo "================================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed."
    echo ""
    echo "Please install Docker first:"
    echo "  curl -fsSL https://get.docker.com | sh"
    echo "  sudo usermod -aG docker \$USER"
    echo ""
    echo "Then log out and back in, and run this script again."
    exit 1
fi

# Check if docker compose is available (using built-in compose v2)
if ! docker compose version &> /dev/null; then
    echo "ERROR: docker compose is not available."
    echo ""
    echo "Please install Docker with compose support."
    exit 1
fi

echo "Docker and docker compose are available."
echo ""

# Check for .env file
if [ ! -f .env ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo ""
    echo "IMPORTANT: Edit .env and set:"
    echo "  - OPENAI_API_KEY=your-api-key"
    echo "  - API_KEYS=your-api-key-for-auth"
    echo "  - SECRET_KEY=\$(python -c \"import secrets; print(secrets.token_urlsafe(32))\")"
    echo ""
    read -p "Press Enter after you've configured .env..."
fi

# Create data directories
echo "Creating data directories..."
mkdir -p data/documents data/chroma logs

# Build the Docker image
echo ""
echo "Building Docker image..."
docker compose -f docker-compose.simple.yml build

# Start the services
echo ""
echo "Starting services..."
docker compose -f docker-compose.simple.yml up -d

# Wait for health check
echo ""
echo "Waiting for API to be healthy..."
sleep 10

# Check health
echo "Checking API health..."
for i in {1..30}; do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo ""
        echo "================================================"
        echo "SUCCESS! Enterprise-RAG is running!"
        echo "================================================"
        echo ""
        echo "API:      http://localhost:8000"
        echo "Docs:     http://localhost:8000/docs"
        echo "Health:   http://localhost:8000/health"
        echo ""
        echo "Test the API:"
        echo "  curl http://localhost:8000/health"
        echo ""
        echo "To stop:"
        echo "  docker compose -f docker-compose.simple.yml down"
        echo ""
        exit 0
    fi
    echo "Waiting... ($i/30)"
    sleep 5
done

echo ""
echo "WARNING: API health check failed. Check logs:"
echo "  docker compose -f docker-compose.simple.yml logs rag-api"
exit 1
