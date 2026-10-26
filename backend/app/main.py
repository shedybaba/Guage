"""The Gauge API. Upload a video, ask questions in plain English, watch progress, get measured answers.

  POST /api/videos                       upload a video -> {video_id, fps, frame_count, width, height, duration}
  POST /api/videos/{id}/questions        {question, known_values?} -> {job_id}
  POST /api/jobs/{id}/answers            answer what a paused job needs ("needs_input") -> {job_id} of a new job
  GET  /api/jobs/{id}                    status, progress, plan, result, what is needed
  GET  /api/jobs/{id}/events             live progress (server-sent events), replayable
  GET  /api/jobs/{id}/tracks             the final tracks (boxes, paths) for the overlay player
  GET  /api/jobs/{id}/data.csv           the tracked data
  GET  /api/videos/{id}/video            the video, for playback
  GET  /api/videos/{id}/segmentation     SAM's own output for a phrase (masks included) when it is stored
  GET  /api/videos/{id}/data.csv         the tracked data of the latest answered question
  POST /api/transcribe                   a spoken question (audio, up to 30 s) -> {text} for the user to review
  GET  /api/health

Interactive documentation is served at /docs.
"""
import asyncio
import json
import logging
import os
import re
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, sam, transcribe, video
from .jobs import ALLOWED_KNOWN, JobRunner, Limits, SubmitError
from .store import TERMINAL, Store, new_id
from .tools.registry import KNOWN_VALUE_KEYS
from .units import QUANTITIES, UnitError, convert, plausible

log = logging.getLogger("gauge.api")
ID = re.compile(r"^[0-9a-f]{32}$")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".3gp"}
NAME_TO_KEY = {v: k for k, v in KNOWN_VALUE_KEYS.items()}


class QuestionIn(BaseModel):
    question: str = Field(min_length=1, max_length=1000, description="The question, in plain words.")
    known_values: dict[str, float | str] | None = Field(
        default=None, description="Values you already know: string_length_m, mass_kg, spring_mass_kg, reference_length_m, "
                                  "ramp_angle_deg, time_s, scale_m_per_px, scale_uncertainty_fraction, reference_object.")


class Answer(BaseModel):
    value: float | None = None
    unit: str | None = None
    text: str | None = None


class AnswersIn(BaseModel):
    answers: dict[str, Answer] = Field(description="name -> value and unit (or text for reference_object).")


def _check_id(value, what="id"):
    if not ID.fullmatch(value or ""):
        raise HTTPException(404, f"No such {what}.")


def limits_from_env():
    def num(name, default):
        try:
            return type(default)(os.getenv(name, default))
        except ValueError:
            return default
    return Limits(max_active=num("GAUGE_MAX_ACTIVE_JOBS", 2), max_waiting=num("GAUGE_MAX_WAITING_JOBS", 12),
                  job_timeout_s=num("GAUGE_JOB_TIMEOUT_S", 900.0), daily_sam_frames=num("GAUGE_DAILY_SAM_FRAMES", 100_000),
                  jobs_per_hour=num("GAUGE_JOBS_PER_HOUR", 120))


class SamTrackerFactory:
    """Builds the SAM tracker for a video. The upload to Meta happens only when a live SAM request is needed."""

    def __init__(self, store, data_dir):
        self.store, self.data_dir, self._locks = store, Path(data_dir), {}

    async def __call__(self, vrow, on_progress):
        from .meta_client import async_client, sync_client
        vid = vrow["id"]

        async def file_id(stale=None):
            """The uploaded video's id, uploading it first if needed. `stale` is an id Meta no longer has."""
            async with self._locks.setdefault(vid, asyncio.Lock()):
                current = self.store.get_video(vid)
                if current["file_id"] and current["file_id"] != stale:  # (another question may have re-uploaded already)
                    return current["file_id"]
                new = await asyncio.to_thread(lambda: video.upload(sync_client(), current["prepared"]))
                self.store.set_file_id(vid, new)
                return new

        info = vrow["info"]
        return sam.SamTracker(async_client(), file_id, (info["width"], info["height"]), key=vrow["key"],
                              on_progress=on_progress)


class NoKeyClient:
    """Stands in for the model client when the key is missing; any use raises, and the job fails with a plain message."""

    def __getattr__(self, name):
        raise RuntimeError("MODEL_API_KEY is missing.")


class ImmutableAssets(StaticFiles):
    """Built assets have a hash in their name, so a browser may keep them forever."""

    def file_response(self, *args, **kwargs):
        r = super().file_response(*args, **kwargs)
        r.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return r


def create_app(store=None, runner=None, data_dir=None, frontend_dir=None, transcriber=None):
    """Build the app. Tests pass a store and a runner with fake model and tracker; production passes nothing.

    If the frontend has been built (frontend/dist), the same server serves it, so there is one address and no CORS.
    """
    data_dir = Path(data_dir or config.DATA_DIR)
    frontend = Path(frontend_dir or os.getenv("GAUGE_FRONTEND_DIST") or config.ROOT / "frontend" / "dist")

    @asynccontextmanager
    async def lifespan(app):
        app.state.data_dir = data_dir
        app.state.store = store or Store(data_dir / "gauge.sqlite3")
        stale = app.state.store.mark_stale_failed()
        if stale:
            log.warning("marked %d unfinished job(s) as failed after a restart", stale)
        app.state.transcriber = transcriber or transcribe.MetaTranscriber()
        app.state.voice_times = []
        if runner is not None:
            app.state.runner = runner
        else:
            from .llm import SparkLLM
            from .meta_client import async_client
            try:
                model_client = async_client()
            except RuntimeError:  # no MODEL_API_KEY: stay up so /api/health can say so, and fail questions plainly
                log.error("MODEL_API_KEY is not set: the site will run, but every question will fail until it is.")
                model_client = NoKeyClient()
            app.state.runner = JobRunner(app.state.store, data_dir, SparkLLM(model_client),
                                         SamTrackerFactory(app.state.store, data_dir), limits=limits_from_env())
        yield
        await app.state.runner.shutdown()
        if store is None:
            app.state.store.close()

    app = FastAPI(title="Gauge", version="0.1.0", lifespan=lifespan,
                  description="Upload a video of an experiment or a machine, ask a question in plain English, and get "
                              "real measurements with visual proof. Numbers come from code, never from a language model.")
    app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["GET", "POST"],
                       allow_headers=["*"], expose_headers=["Content-Range", "Accept-Ranges"])

    def st(request):
        return request.app.state.store

    @app.get("/api/health")
    async def health(request: Request):
        s = st(request)
        return {"status": "ok", "version": app.version, "sam_model": config.SAM_MODEL, "language_model": config.SPARK_MODEL,
                "api_key_configured": bool(os.getenv("MODEL_API_KEY")), "active_jobs": s.count_active()}

    # ------------------------------------------------------------------ voice
    @app.post("/api/transcribe")
    async def transcribe_question(request: Request, file: UploadFile):
        """Turn a recorded question into text. The text is only shown to the user; nothing is asked or measured."""
        now = time.time()
        recent = request.app.state.voice_times
        recent[:] = [t for t in recent if now - t < 3600]
        if len(recent) >= config.VOICE_PER_HOUR:
            raise HTTPException(429, "Too many voice questions in the last hour. Please type your question instead.")
        data = await file.read(config.MAX_AUDIO_MB * 1024 * 1024 + 1)
        if not data:
            raise HTTPException(400, "The recording is empty.")
        if len(data) > config.MAX_AUDIO_MB * 1024 * 1024:
            raise HTTPException(413, "That recording is too large. Questions can be up to 30 seconds long.")
        folder = request.app.state.data_dir / "tmp" / new_id()
        folder.mkdir(parents=True, exist_ok=True)
        try:
            src = folder / "recording.bin"
            src.write_bytes(data)
            wav = folder / "recording.wav"
            try:
                seconds = await asyncio.to_thread(transcribe.to_wav, src, wav)
                if seconds > transcribe.MAX_SECONDS + 2:
                    raise transcribe.TranscribeError("That recording is longer than 30 seconds. Please keep the question short.", 413)
                if seconds < 0.3:
                    raise transcribe.TranscribeError("That recording was too short to hear anything. Please try again.", 400)
                recent.append(now)
                text = await request.app.state.transcriber.transcribe(wav)
            except transcribe.TranscribeError as e:
                raise HTTPException(e.status, str(e)) from None
        finally:
            import shutil
            shutil.rmtree(folder, ignore_errors=True)  # recordings are never kept
        if not text:
            raise HTTPException(422, "No speech was heard. Please try again, closer to the microphone, or type your question.")
        return {"text": text, "seconds": round(seconds, 1)}

    # ------------------------------------------------------------------ videos
    @app.post("/api/videos", status_code=201)
    async def upload_video(request: Request, file: UploadFile):
        s, vid = st(request), new_id()
        folder = request.app.state.data_dir / "videos" / vid
        folder.mkdir(parents=True, exist_ok=True)
        ext = Path(file.filename or "").suffix.lower()
        dest = folder / f"original{ext if ext in VIDEO_EXTENSIONS else '.bin'}"
        limit, size = config.MAX_UPLOAD_MB * 1024 * 1024, 0

        def discard():
            import shutil
            shutil.rmtree(folder, ignore_errors=True)

        with dest.open("wb") as out:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > limit:
                    out.close()
                    discard()
                    raise HTTPException(413, f"That file is larger than the {config.MAX_UPLOAD_MB} MB limit.")
                out.write(chunk)
        if size == 0:
            discard()
            raise HTTPException(400, "The file is empty.")
        try:
            info = await asyncio.to_thread(video.probe, dest)
        except Exception:  # noqa: BLE001 - ffprobe could not read it
            discard()
            raise HTTPException(400, "This does not look like a video that Gauge can read. Use an MP4 or MOV file.")
        if info.frame_count < config.MIN_FRAMES or info.duration < 1.0:
            discard()
            raise HTTPException(400, "The video is too short to measure anything (it needs at least about a second).")
        if info.frame_count > config.MAX_FRAMES:
            discard()
            raise HTTPException(400, f"The video has {info.frame_count} frames; the limit is {config.MAX_FRAMES} (about 10 minutes).")
        prepared = folder / "prepared.mp4"
        try:
            await asyncio.to_thread(video.prepare, dest, prepared)
        except (subprocess.CalledProcessError, OSError):
            discard()
            raise HTTPException(400, "Gauge could not convert this video. Try exporting it as an MP4 (H.264).")
        key = await asyncio.to_thread(sam.video_key, dest)
        (folder / "timestamps.json").write_text(json.dumps(info.timestamps))
        meta = {"fps": info.fps, "frame_count": info.frame_count, "width": info.width, "height": info.height,
                "duration": info.duration, "rotation": info.rotation, "codec": info.codec}
        s.add_video(vid, file.filename or "video", dest, prepared, key, meta)
        return {"video_id": vid, "fps": info.fps, "frame_count": info.frame_count, "width": info.width,
                "height": info.height, "duration": info.duration}

    def get_video(request, video_id):
        _check_id(video_id, "video")
        v = st(request).get_video(video_id)
        if v is None:
            raise HTTPException(404, "No such video.")
        return v

    @app.get("/api/videos/{video_id}")
    async def video_info(request: Request, video_id: str):
        v = get_video(request, video_id)
        jobs = st(request).jobs_for_video(video_id)
        return {"video_id": video_id, **v["info"], "filename": v["filename"],
                "questions": [{"job_id": j["id"], "question": j["question"], "status": j["status"]} for j in jobs]}

    @app.get("/api/videos/{video_id}/video")
    async def video_file(request: Request, video_id: str):
        v = get_video(request, video_id)
        return FileResponse(v["prepared"], media_type="video/mp4")

    @app.get("/api/videos/{video_id}/segmentation")
    async def segmentation(request: Request, video_id: str, phrase: str, format: str = "json"):
        """SAM's output for a phrase from the cache. format=raw returns SAM's own text (with masks) for the overlay player."""
        v = get_video(request, video_id)
        cached = sam.load_cached(v["key"], phrase)
        if cached is None:
            raise HTTPException(404, "That phrase has not been tracked in this video yet. Ask a question about it first.")
        if format == "raw":
            raw = sam.load_raw(v["key"], phrase)
            if raw is None:
                raise HTTPException(404, "SAM's raw output was not stored for this phrase.")
            return PlainTextResponse(raw)
        return {"phrase": phrase, "frames_billed": cached.frames_billed,
                "points": [{"frame": p.frame, "object_id": p.object_id, "box": [p.x1, p.y1, p.x2, p.y2],
                            "centre": [p.cx, p.cy], "confidence": p.confidence} for p in cached.points]}

    @app.get("/api/videos/{video_id}/data.csv")
    async def video_csv(request: Request, video_id: str, job_id: str | None = None):
        get_video(request, video_id)
        jobs = [j for j in reversed(st(request).jobs_for_video(video_id)) if job_id is None or j["id"] == job_id]
        for j in jobs:
            f = request.app.state.data_dir / "jobs" / j["id"] / "data.csv"
            if f.exists():
                return FileResponse(f, media_type="text/csv", filename=f"gauge_{video_id[:8]}.csv")
        raise HTTPException(404, "No tracked data yet: ask a question about this video first.")

    # ------------------------------------------------------------------ questions and jobs
    async def submit(request, video_id, question, known, parent=None):
        try:
            return await request.app.state.runner.submit(video_id, question, known, parent)
        except SubmitError as e:
            raise HTTPException(e.status, e.message)

    @app.post("/api/videos/{video_id}/questions", status_code=202)
    async def ask(request: Request, video_id: str, body: QuestionIn):
        get_video(request, video_id)
        bad = sorted(set(body.known_values or {}) - ALLOWED_KNOWN)
        if bad:
            raise HTTPException(422, f"Unknown value(s): {', '.join(bad)}. Allowed: {', '.join(sorted(ALLOWED_KNOWN))}.")
        return {"job_id": await submit(request, video_id, body.question, body.known_values)}

    @app.post("/api/jobs/{job_id}/answers", status_code=202)
    async def answer(request: Request, job_id: str, body: AnswersIn):
        """Answer what a paused job needs; starts a new job for the same question with the answers included."""
        job = get_job(request, job_id)
        if job["status"] != "needs_input":
            raise HTTPException(409, "This question is not waiting for an answer.")
        known = dict(job["known_values"] or {})
        for name, a in body.answers.items():
            if name == "reference_object":
                if not (a.text or "").strip():
                    raise HTTPException(422, "reference_object needs a name in `text`.")
                known["reference_object"] = a.text.strip()
            elif name in QUANTITIES:
                if a.value is None:
                    raise HTTPException(422, f"{name} needs a `value` and a `unit`.")
                try:
                    si, _ = convert(name, a.value, a.unit or "")
                except UnitError as e:
                    raise HTTPException(422, str(e))
                if not plausible(name, si):
                    raise HTTPException(422, f"{a.value:g} {a.unit or ''} does not look right for the {name.replace('_', ' ')}.")
                known[NAME_TO_KEY[name]] = si
            else:
                raise HTTPException(422, f"Cannot answer '{name}'.")
        return {"job_id": await submit(request, job["video_id"], job["question"], known, job_id)}

    def get_job(request, job_id):
        _check_id(job_id, "question")
        j = st(request).get_job(job_id)
        if j is None:
            raise HTTPException(404, "No such question.")
        return j

    def public_job(j):
        plan = j["plan"] or {}
        return {"job_id": j["id"], "video_id": j["video_id"], "parent_id": j["parent_id"], "question": j["question"],
                "status": j["status"],
                "progress": j["progress"], "message": j["message"], "created": j["created"], "updated": j["updated"],
                "plan": {k: plan.get(k) for k in ("tool", "outputs", "interpretation", "targets", "notes", "values", "line")}
                if plan else None,
                "result": j["result"], "needs": j["needs"], "error": j["error"]}

    @app.get("/api/jobs/{job_id}")
    async def job_status(request: Request, job_id: str):
        return public_job(get_job(request, job_id))

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(request: Request, job_id: str, after: int = 0):
        """Server-sent events: every step so far, then live. Reconnect with Last-Event-ID (or ?after=) to continue."""
        get_job(request, job_id)
        s = st(request)
        try:
            last = int(request.headers.get("last-event-id", after))
        except ValueError:
            last = after

        async def stream():
            nonlocal last
            while True:
                job = s.get_job(job_id)
                finished = job["status"] in TERMINAL  # read BEFORE the events, so the closing events are never missed
                for e in s.events_after(job_id, last):
                    last = e["id"]
                    yield f"id: {e['id']}\nevent: {e['kind']}\ndata: {json.dumps(e)}\n\n"
                if finished:
                    yield f"event: end\ndata: {json.dumps({'status': job['status']})}\n\n"
                    return
                if await request.is_disconnected():
                    return
                yield ": keep-alive\n\n"
                await asyncio.sleep(0.3)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/jobs/{job_id}/tracks")
    async def job_tracks(request: Request, job_id: str):
        get_job(request, job_id)
        f = request.app.state.data_dir / "jobs" / job_id / "tracks.json"
        if not f.exists():
            raise HTTPException(404, "This question has no tracks (it was not answered, or no object was found).")
        return FileResponse(f, media_type="application/json")

    @app.get("/api/jobs/{job_id}/data.csv")
    async def job_csv(request: Request, job_id: str):
        j = get_job(request, job_id)
        f = request.app.state.data_dir / "jobs" / job_id / "data.csv"
        if not f.exists():
            raise HTTPException(404, "This question has no tracked data.")
        return FileResponse(f, media_type="text/csv", filename=f"gauge_{j['video_id'][:8]}_{job_id[:8]}.csv")

    if (frontend / "index.html").is_file():
        if (frontend / "assets").is_dir():
            app.mount("/assets", ImmutableAssets(directory=frontend / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        async def single_page_app(path: str):
            """Any address that is not the API opens the app (so /analyze/<id> works on reload and when shared)."""
            if path.startswith("api/") or path in ("docs", "redoc", "openapi.json"):
                raise HTTPException(404, "Not found.")
            candidate = (frontend / path).resolve()
            if path and candidate.is_file() and frontend.resolve() in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(frontend / "index.html", headers={"Cache-Control": "no-cache"})

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception):
        log.exception("unhandled error on %s", request.url.path)
        return JSONResponse({"detail": "Something went wrong on the server. The problem was recorded."}, status_code=500)

    return app


app = create_app()
