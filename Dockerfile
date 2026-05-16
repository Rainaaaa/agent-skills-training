# syntax=docker/dockerfile:1.7
#
# agent-skills-training — GPU image for all three stages
#   (pretraining + HCL + downstream {SFT, inference, baseline eval}).
#
# Base: official PyTorch image with CUDA 12.1 + cuDNN 8 + Python.
# Comes with torch already installed, so `requirements.txt` only needs
# the HF stack.
#
# Build:
#     docker build -t agent-skills-training .
#
# Smoke test:
#     docker run --rm --gpus all \
#         -v $(pwd)/inputs:/app/inputs:ro \
#         -v $(pwd)/outputs:/app/outputs \
#         agent-skills-training \
#         stages/pretraining/train.py --help
#
# Real run (mount dataset + model registry):
#     docker run --rm --gpus all \
#         -v /path/to/datasets:/data \
#         -v $(pwd)/model_path.json:/app/model_path.json:ro \
#         -e AGENTSKILLS_FULL_CPT_TRAIN=/data/full_cpt/train.parquet \
#         -e AGENTSKILLS_FULL_CPT_VAL=/data/full_cpt/val.parquet \
#         -e AGENTSKILLS_FULL_CPT_TEST=/data/full_cpt/test.parquet \
#         -e AGENTSKILLS_BACKBONE=Foundation-Sec-8B-Reasoning \
#         agent-skills-training \
#         stages/pretraining/train.py \
#             --config stages/pretraining/configs/example.yaml

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Sane HF defaults so cache + downloads land inside the mounted volume.
    HF_HOME=/app/outputs/hf_cache \
    TRANSFORMERS_CACHE=/app/outputs/hf_cache/transformers \
    HF_DATASETS_CACHE=/app/outputs/hf_cache/datasets \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# System deps for git-lfs + curl (needed by some HF downloaders).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Source — every stage shares one image.
COPY common  /app/common
COPY stages  /app/stages
COPY scripts /app/scripts
COPY model_path.json /app/model_path.json
COPY README.md       /app/README.md

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["--help"]
