"""Gauge on Modal: the same FastAPI server (API plus built website) behind one HTTPS address.

Build the website first, then deploy:

    cd frontend && npm ci && npm run build && cd ..
    .venv-modal/bin/modal deploy modal_app.py

Needs a Modal secret named gauge-meta holding MODEL_API_KEY (created by you, never committed).

One container on purpose: jobs run as background tasks inside the server and share one SQLite file, so the server must
stay up (min_containers=1) and must be the only writer (max_containers=1). The database, uploaded videos and the SAM
cache live on a Modal Volume mounted at /data, so they survive restarts.
"""
import modal

app = modal.App("gauge")
data = modal.Volume.from_name("gauge-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("ffmpeg")
    .pip_install_from_requirements("backend/requirements.txt")
    .env({"PYTHONUNBUFFERED": "1", "PYTHONPATH": "/app/backend", "GAUGE_DATA_DIR": "/data", "GAUGE_DAILY_SAM_FRAMES": "30000"})
    .add_local_dir("backend/app", "/app/backend/app", ignore=["**/__pycache__", "**/*.pyc"])
    .add_local_dir("frontend/dist", "/app/frontend/dist")
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("gauge-meta")],
    volumes={"/data": data},
    cpu=2.0,
    memory=4096,
    min_containers=1,
    max_containers=1,
    timeout=3600,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web():
    from app.main import app as gauge_app

    return gauge_app
