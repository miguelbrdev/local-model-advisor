# Local Model Advisor API image.
#
# The project is *not* pip-installed: its modules resolve paths relative to
# the source tree (data/catalog_bootstrap.json, runtime/), so the sources are
# copied to /app and run from there.  Only the runtime dependencies declared
# in pyproject.toml are installed, read from that single source of truth.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml ./
RUN python -c "import tomllib; d = tomllib.load(open('pyproject.toml', 'rb')); \
print('\n'.join(d['project']['dependencies']))" > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY data ./data

# SQLite conversations + catalogue snapshots live here; mounted as a volume.
RUN mkdir -p /app/runtime
VOLUME ["/app/runtime"]

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
