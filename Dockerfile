FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY pyproject.toml ./
COPY support_rag ./support_rag
RUN --mount=type=cache,target=/root/.cache/pip pip install . \
    && useradd --create-home --shell /usr/sbin/nologin rag \
    && install -d -o rag -g rag /home/rag/.cache

USER rag
EXPOSE 8000
CMD ["uvicorn", "support_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
