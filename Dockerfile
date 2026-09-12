# Two stages: node builds the UI, python serves it.
# The result is one process -- FastAPI mounts frontend/dist itself, so there is
# no node in the running container and nothing to keep in sync at runtime.

FROM node:20-slim AS ui
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY backend/ ./backend/
COPY data/ ./data/

# Built assets land where main.py expects them: ../frontend/dist
COPY --from=ui /app/frontend/dist ./frontend/dist

# Hugging Face runs the container as uid 1000, not root. The observer writes a
# record per run into runs/, so that directory has to exist and be owned by the
# user that will actually be writing to it.
RUN useradd -m -u 1000 user && mkdir -p /app/runs && chown -R user:user /app
USER user

EXPOSE 7860
# Shell form on purpose: hosts inject the port differently (Render sets PORT,
# HF expects 7860), and ${PORT:-7860} lets one image satisfy both.
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860}
