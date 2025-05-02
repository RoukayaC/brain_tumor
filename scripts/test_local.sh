#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

echo "=== Building and Testing Brain Tumor Classification API Locally ==="

# --- Build Docker Image ---
echo "Building Docker image..."
docker build -f flask/Dockerfile.prod -t brain-tumor-api-local flask/
echo "Image built successfully."

# --- Run Docker Container ---
echo "Starting Docker container on port 8080..."
docker run -d --name brain-tumor-api-test -p 8080:8080 brain-tumor-api-local
echo "Container started."

# --- Wait for container to start fully ---
echo "Waiting for API to start (10 seconds)..."
sleep 10

# --- Test the API ---
echo "Testing API health endpoint..."
curl -s http://localhost:8080/ > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ API is responding correctly!"
    echo "You can access the API at: http://localhost:8080/"
    echo "To stop the container run: docker stop brain-tumor-api-test && docker rm brain-tumor-api-test"
else
    echo "❌ API is not responding. Check container logs:"
    docker logs brain-tumor-api-test
    echo "Stopping container..."
    docker stop brain-tumor-api-test
    docker rm brain-tumor-api-test
fi
