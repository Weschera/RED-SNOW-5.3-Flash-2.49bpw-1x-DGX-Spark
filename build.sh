#!/usr/bin/env bash
# Build vcruz305/exllamav3 at the commit RED-SNOW's DEPLOY.md pins, into a Docker image for one DGX Spark (GB10, sm_121).
set -euo pipefail
EXL3_REF="${EXL3_REF:-7c1636fead6b6d802dc3ebf0e7ef02c4e76e9c2f}"
SRC="${SRC:-$HOME/exllamav3-redsnow}"
IMAGE="${IMAGE:-redsnow-exl3:7c1636f}"
BASE="${BASE:-nvcr.io/nvidia/pytorch:26.07-py3}"

[ -d "$SRC/.git" ] || git clone https://github.com/vcruz305/exllamav3.git "$SRC"
git -C "$SRC" fetch -q origin "$EXL3_REF" 2>/dev/null || git -C "$SRC" fetch -q origin
git -C "$SRC" checkout -q "$EXL3_REF"
echo "exllamav3 @ $(git -C "$SRC" rev-parse --short HEAD)"

cat > "$SRC/Dockerfile.redsnow" <<EOF
FROM $BASE
ENV TORCH_CUDA_ARCH_LIST=12.1 MAX_JOBS=16
COPY . /opt/exllamav3
WORKDIR /opt/exllamav3
RUN pip install -q -r requirements.txt && pip install -q --no-build-isolation -e . \\
 && pip install -q fastapi uvicorn huggingface_hub jsonschema "transformers>=5.0" jinja2
RUN python -c "import exllamav3, exllamav3_ext, transformers; print('ext ok, transformers', transformers.__version__)"
EOF
docker build -f "$SRC/Dockerfile.redsnow" -t "$IMAGE" "$SRC"
echo "built $IMAGE"
