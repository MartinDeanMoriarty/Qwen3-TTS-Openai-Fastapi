# Qwen3-TTS OpenAI-Compatible API Server
#
# Targets:
#   production  GPU image (PyTorch cu128 wheels; they bundle the CUDA runtime,
#               the host driver comes in through the NVIDIA container toolkit)
#   cpu-base    CPU-only image
#
# Package versions are pinned by docker/constraints.txt, the environment the
# benchmarks in docs/PERFORMANCE.md were measured with.

ARG TORCH_VERSION=2.11.0

# =============================================================================
# GPU image
# =============================================================================
# Ubuntu 24.04 ships Python 3.12. The former Ubuntu 22.04 base had
# Python 3.11.0rc1, a release candidate on which TorchDynamo segfaulted while
# compiling the model.
FROM ubuntu:24.04 AS production
ARG TORCH_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    PATH="/opt/venv/bin:$PATH"

# build-essential and python3-dev stay: Triton compiles its launcher at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev build-essential \
        curl ffmpeg libsndfile1 sox libsox-dev \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir torch==${TORCH_VERSION} torchaudio==${TORCH_VERSION} \
        --index-url https://download.pytorch.org/whl/cu128

COPY docker/requirements.txt docker/constraints.txt /tmp/deps/
RUN pip install --no-cache-dir -r /tmp/deps/requirements.txt -c /tmp/deps/constraints.txt

WORKDIR /app
COPY . .
# Dependencies are already in place; --no-deps keeps the gradio demo stack out
RUN pip install --no-cache-dir --no-deps -e .

# Ubuntu 24.04 comes with a uid-1000 "ubuntu" user; replace it so the mounted
# Hugging Face cache (owned by uid 1000 on the host) stays writable
RUN userdel -r ubuntu 2>/dev/null; \
    useradd --uid 1000 --create-home --shell /bin/bash appuser \
    && mkdir -p /tmp/numba_cache /home/appuser/.cache/compile \
    && chown -R appuser:appuser /app /tmp/numba_cache /home/appuser/.cache
USER appuser

# torch.compile and Triton caches; mount a volume on .cache/compile so later
# starts reuse the compiled kernels (~30 s instead of ~2 min).
# Expandable segments cut the reserved VRAM from ~4.7 to ~3.4 GB (less
# fragmentation), at unchanged speed.
ENV HOST=0.0.0.0 \
    PORT=8880 \
    WORKERS=1 \
    PYTHONPATH=/app \
    TTS_BACKEND=fast \
    TORCHINDUCTOR_CACHE_DIR=/home/appuser/.cache/compile/inductor \
    TRITON_CACHE_DIR=/home/appuser/.cache/compile/triton \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXPOSE 8880

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["python", "-m", "api.main"]

# =============================================================================
# CPU-only image
# =============================================================================
FROM python:3.12-slim AS cpu-base
ARG TORCH_VERSION

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl ffmpeg libsndfile1 sox libsox-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==${TORCH_VERSION} torchaudio==${TORCH_VERSION} \
        --index-url https://download.pytorch.org/whl/cpu

COPY docker/requirements.txt docker/constraints.txt /tmp/deps/
RUN pip install --no-cache-dir -r /tmp/deps/requirements.txt -c /tmp/deps/constraints.txt

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir --no-deps -e .

RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /tmp/numba_cache \
    && chown -R appuser:appuser /app /tmp/numba_cache
USER appuser

ENV HOST=0.0.0.0 \
    PORT=8880 \
    WORKERS=1 \
    PYTHONPATH=/app \
    TTS_BACKEND=official

EXPOSE 8880

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["python", "-m", "api.main"]
