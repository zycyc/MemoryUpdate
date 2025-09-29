#!/bin/bash

# Docker startup script with --init flag to prevent zombie processes
# The --init flag uses tini as PID 1 which properly reaps child processes

echo "Stopping existing container if running..."
docker stop verl_container 2>/dev/null || true
docker rm verl_container 2>/dev/null || true

echo "Starting new container with --init flag for proper process cleanup..."
docker run --name verl_container -d \
  --init \
  --gpus all \
  --cap-add=SYS_PTRACE \
  --ipc=host \
  --network=host \
  --privileged \
  --security-opt seccomp=unconfined \
  -v $(pwd):/workspace/memupdate \
  -v $(pwd)/verl:/workspace/verl \
  -v ~/.cache/huggingface/hub:/root/.cache/huggingface/hub \
  verlai/verl:app-verl0.5-transformers4.55.4-sglang0.4.10.post2-mcore0.13.0-te2.2 \
  sleep infinity

echo "Container started with PID 1 zombie reaping enabled!"

echo ""
echo "Installing required dependencies..."

# Install langmem for Python 3.10 (container default)
docker exec verl_container bash -c "python3 -m pip install langmem"

# Apply langmem Python 3.10 compatibility patch
# (fixes typing.NotRequired which is only available in Python 3.11+)
docker exec verl_container bash -c "
  sed -i 's/typing.NotRequired/typing_extensions.NotRequired/g' /usr/local/lib/python3.10/dist-packages/langmem/knowledge/extraction.py && 
  sed -i '/^import typing$/a import typing_extensions' /usr/local/lib/python3.10/dist-packages/langmem/knowledge/extraction.py
"

# Install memupdate package (no deps to avoid version conflicts)
docker exec verl_container bash -c "
  cd /workspace/memupdate && 
  python3 -m pip install -e . --no-deps
"

echo "Dependencies installed successfully!"
echo ""
echo "To enter the container:"
echo "  docker exec -it verl_container bash"
echo ""
echo "To run training:"
echo "  docker exec verl_container bash -c 'cd /workspace/memupdate && bash run_training_container.sh'"