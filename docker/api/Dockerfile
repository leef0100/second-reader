FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_SYSTEM_PYTHON=1 UV_COMPILE_BYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md alembic.ini ./
COPY alembic ./alembic
COPY src ./src
COPY configs ./configs
# CPU-only torch keeps the image small; CUDA deployments override the index.
RUN uv pip install --index-strategy unsafe-best-match \
      --extra-index-url https://download.pytorch.org/whl/cpu . \
    && useradd -r -u 10001 app && mkdir -p /data/blobs && chown -R app /data
USER app
LABEL org.opencontainers.image.description="Second Reader — For research use only. Not for diagnostic use."
