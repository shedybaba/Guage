"""The job runner: one question about one video, from plan to explained answer, with live progress.

A job moves through   queued -> planning -> tracking -> measuring -> explaining -> done
and may stop early as   needs_input  (something is missing, or a number the user said could not be checked)
                        failed       (a system problem: the message is for the user, the details go to the server log).
"Done" also covers honest non-answers: an unsupported question, or a measurement that could not be made.

Background work is plain asyncio tasks writing to the SQLite store (no Celery, no Redis). Every step is an EVENT in
the store, so any number of browsers can watch, and reconnecting loses nothing.
"""
import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

from . import runner, video
from .explain import build_facts, explain
from .llm import LLMError, Usage
from .planner import Plan, PlannerError, make_plan
from .quality import track_objects_with_checks
from .store import jsonable
from .tools.registry import KNOWN_VALUE_KEYS, used_values

log = logging.getLogger("gauge.jobs")
SAM_USD_PER_FRAME = 0.20 / 1000
NUMERIC_KNOWN = set(KNOWN_VALUE_KEYS) | {"scale_m_per_px", "scale_uncertainty_fraction"}
ALLOWED_KNOWN = NUMERIC_KNOWN | {"reference_object"}


class SubmitError(Exception):
    """A question was refused before it started (a limit was reached). `status` is the HTTP status to use."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status, self.message = status, message


@dataclass
class Limits:
    max_active: int = 2  # jobs working at the same time
    max_waiting: int = 12  # jobs working or queued: more than this are refused
    job_timeout_s: float = 900.0
    daily_sam_frames: int = 100_000  # about 20 US dollars of SAM
    jobs_per_hour: int = 120
    question_chars: int = 1000


def load_video_info(data_dir, video_row):
    """Rebuild the ffprobe facts of a stored video, including every frame's timestamp."""
    info = dict(video_row["info"])
    ts = json.loads((Path(data_dir) / "videos" / video_row["id"] / "timestamps.json").read_text())
    return video.VideoInfo(timestamps=ts, **info)


def clean_known_values(raw):
    """Only the values the API understands, as numbers (and one object name); anything else is dropped."""
    out = {}
    for k, v in (raw or {}).items():
        if k in NUMERIC_KNOWN and isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf"):
            out[k] = float(v)
        elif k == "reference_object" and isinstance(v, str) and v.strip():
            out[k] = v.strip()[:60]
    return out


class JobRunner:
    """Runs jobs in the background. llm: SparkLLM. tracker_factory(video_row, on_progress) -> SAM tracker (async)."""

    def __init__(self, store, data_dir, llm, tracker_factory, keyframes_fn=None, limits=None):
        self.store, self.data_dir, self.llm = store, Path(data_dir), llm
        self.tracker_factory = tracker_factory
        self.keyframes_fn = keyframes_fn or self._default_keyframes
        self.limits = limits or Limits()
        self._sem = asyncio.Semaphore(self.limits.max_active)
        self._tasks = {}

    # ---- submitting
    async def submit(self, video_id, question, known_values=None, parent_id=None):
        question = (question or "").strip()
        if not question:
            raise SubmitError(422, "The question is empty.")
        if len(question) > self.limits.question_chars:
            raise SubmitError(422, f"The question is too long (the limit is {self.limits.question_chars} characters).")
        if self.store.get_video(video_id) is None:
            raise SubmitError(404, "That video was not found.")
        if self.store.count_active() >= self.limits.max_waiting:
            raise SubmitError(429, "Gauge is busy with other questions right now. Please try again in a minute.")
        hour_jobs, _ = self.store.usage_since(time.time() - 3600)
        if hour_jobs >= self.limits.jobs_per_hour:
            raise SubmitError(429, "Too many questions have been asked in the last hour. Please try again later.")
        _, frames = self.store.usage_since(time.time() - 86400)
        if frames >= self.limits.daily_sam_frames:
            raise SubmitError(503, "Gauge has reached its daily usage limit. Please try again tomorrow.")
        job_id = self.store.create_job(video_id, question, clean_known_values(known_values), parent_id)
        self._tasks[job_id] = asyncio.create_task(self._guard(job_id))
        return job_id

    async def wait(self, job_id):
        task = self._tasks.get(job_id)
        if task:
            await task

    async def shutdown(self):
        for t in list(self._tasks.values()):
            t.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)

    # ---- plumbing
    def _emit(self, job_id, kind, message="", data=None, status=None, progress=None):
        fields = {}
        if status:
            fields["status"] = status
        if progress is not None:
            fields["progress"] = progress
        if message and kind in ("status", "plan", "progress", "needs", "result", "error"):
            fields["message"] = message
        if fields:
            self.store.update_job(job_id, **fields)
        return self.store.add_event(job_id, kind, message, data)

    def _finish(self, job_id, status, message, kind, result=None, needs=None, error=None, usage=None):
        """Write the final state, then the closing event (events last, so a watcher sees the state when the event arrives)."""
        fields = dict(status=status, message=message, result=result, needs=needs, error=error, usage=usage)
        if status == "done":
            fields["progress"] = 1.0  # a paused or failed job keeps the progress it had reached
        self.store.update_job(job_id, **fields)
        self.store.add_event(job_id, kind, message, result if result is not None else needs)

    async def _guard(self, job_id):
        try:
            async with self._sem:
                await asyncio.wait_for(self._pipeline(job_id), self.limits.job_timeout_s)
        except asyncio.TimeoutError:
            self._fail(job_id, "This took too long and was stopped. Try a shorter clip.", "timeout")
        except asyncio.CancelledError:
            self._fail(job_id, "This question was cancelled because Gauge is shutting down.", "cancelled")
            raise
        except Exception:  # noqa: BLE001 - the user gets a plain message, the details go to the server log
            log.exception("job %s failed", job_id)
            self._fail(job_id, "Something went wrong while measuring. The problem was recorded; please try again.", "internal")
        finally:
            self._tasks.pop(job_id, None)

    def _fail(self, job_id, message, code):
        job = self.store.get_job(job_id)
        if job and job["status"] in ("done", "failed", "needs_input"):
            return
        self.store.update_job(job_id, status="failed", message=message, error=code)
        self.store.add_event(job_id, "error", message, {"code": code})

    def _default_keyframes(self, video_row):
        cache = self.data_dir / "videos" / video_row["id"] / "keyframes.json"
        if cache.exists():
            return json.loads(cache.read_text())
        frames = video.keyframes(video_row["prepared"], 3)
        cache.write_text(json.dumps(frames))
        return frames

    # ---- the pipeline
    async def _pipeline(self, job_id):
        job = self.store.get_job(job_id)
        vrow = self.store.get_video(job["video_id"])
        info = load_video_info(self.data_dir, vrow)
        size = (info.width, info.height)
        known = job["known_values"] or {}
        usage = Usage()

        # 1. understand the question
        self._emit(job_id, "status", "Reading your question and looking at the video", status="planning", progress=0.03)
        prev = self.store.latest_plan(job["video_id"])
        try:
            frames = await asyncio.to_thread(self.keyframes_fn, vrow)
            plan_known = {k: v for k, v in known.items() if k in KNOWN_VALUE_KEYS or k == "reference_object"}
            plan = await make_plan(self.llm, job["question"],
                                   video={"width": info.width, "height": info.height, "fps": info.fps, "duration": info.duration},
                                   keyframes=frames, known_values=plan_known,
                                   previous_plan=Plan.from_dict(prev) if prev else None)
        except PlannerError:
            self._fail(job_id, "Gauge could not read the question just now (the language model did not answer). "
                               "Please try again.", "planner")
            return
        usage.add(plan.usage)
        self.store.update_job(job_id, plan=plan.to_dict())
        self._emit(job_id, "plan", plan.interpretation, {"plan": plan.to_dict()}, progress=0.08)

        if plan.unsupported:
            result = {"kind": "unsupported", "answerable": False, "question": job["question"],
                      "reason": plan.unsupported["reason"], "can_measure": plan.unsupported["can_measure"],
                      "usage": self._usage(0, usage)}
            self._finish(job_id, "done", plan.unsupported["reason"], "result", result, usage=result["usage"])
            return

        needs = self._needs(plan, known)
        if needs["missing"] or needs["confirm"]:
            msg = needs["missing"][0]["question"] if needs["missing"] else needs["confirm"][0]["question"]
            self._finish(job_id, "needs_input", msg, "needs", needs=needs, usage=self._usage(0, usage))
            return

        # 2. track
        total = info.frame_count
        last = {"t": 0.0}

        def on_progress(what, n):
            now = time.monotonic()
            if now - last["t"] < 0.4 and n < total:
                return
            last["t"] = now
            frac = 0.10 + 0.55 * min(1.0, n / max(total, 1))
            self._emit(job_id, "progress", f"Tracking {what}: frame {min(n, total)} of {total}", {"done": n, "total": total},
                       progress=max(frac, self.store.get_job(job_id)["progress"]))

        def on_log(entry):
            self.store.add_event(job_id, "log", entry.message, {"step": entry.step, "ok": entry.ok})

        self._emit(job_id, "status", "Finding and following the object in every frame", status="tracking", progress=0.10)
        tracker = await self.tracker_factory(vrow, on_progress)
        checked = await track_objects_with_checks(tracker, plan.main_phrases, total, size, check_camera=plan.check_camera,
                                                  on_log=on_log)
        if not checked.tracks:
            advice = checked.advice
            result = {"kind": "no_result", "answerable": False, "question": job["question"],
                      "interpretation": plan.interpretation,
                      "reason": "Gauge could not find the object in the video, under any of the names it tried.",
                      "advice": advice, "checks": self._checks(checked.log), "usage": self._usage(checked.frames_billed, usage)}
            self._finish(job_id, "done", result["reason"], "result", result, usage=result["usage"])
            return
        self._emit(job_id, "progress", f"Found {checked.count} object(s)", progress=0.68)

        # 3. scale, then the measurement
        self._emit(job_id, "status", "Measuring", status="measuring", progress=0.70)
        scale, extra_billed = None, 0
        try:
            scale = await runner.measure_scale(tracker, plan, info, known, on_log=on_log)
            extra_billed = scale.frames_billed if scale else 0
            tool_result = await asyncio.to_thread(runner.run_tool, plan, checked, info, scale)
        except runner.MeasureError as e:
            tool_result = {"value": None, "warnings": [str(e)]}
        billed = checked.frames_billed + extra_billed
        if tool_result.get("value") is None:
            reason = tool_result["warnings"][-1] if tool_result.get("warnings") else "No measurement could be made."
            result = {"kind": "no_result", "answerable": False, "question": job["question"],
                      "interpretation": plan.interpretation, "reason": reason,
                      "advice": checked.advice, "warnings": tool_result.get("warnings", []),
                      "checks": self._checks(checked.log), "usage": self._usage(billed, usage)}
            self._save_tracks(job_id, checked, info)
            self._finish(job_id, "done", reason, "result", result, usage=result["usage"])
            return

        # 4. explain
        self._emit(job_id, "status", "Writing the explanation", status="explaining", progress=0.85)
        checks = [e.message for e in checked.log] + (scale.notes if scale else [])
        facts = build_facts(plan.tool, tool_result, plan, checks=checks, question=job["question"])
        explanation = await explain(self.llm, facts)
        usage.add(explanation.usage)
        self._save_tracks(job_id, checked, info)
        result = {
            "kind": "measurement", "answerable": True, "tool": plan.tool, "question": job["question"],
            "interpretation": plan.interpretation, "headline": runner.fact_dict(facts.headline),
            "numbers": [runner.fact_dict(n) for n in facts.numbers], "inputs": facts.inputs, "statements": facts.statements,
            "explanation": explanation.to_dict(), "warnings": facts.warnings, "notes": plan.notes,
            "checks": self._checks(checked.log) + [{"step": "scale", "message": n, "ok": None} for n in (scale.notes if scale else [])],
            "chart": tool_result.get("series_for_chart", {}),
            "tracks": {"url": f"/api/jobs/{job_id}/tracks", "objects": sorted(str(k) for k in checked.tracks),
                       "phrase": checked.phrase, "video_url": f"/api/videos/{job['video_id']}/video"},
            "downloads": {"csv": f"/api/jobs/{job_id}/data.csv"},
            "usage": self._usage(billed, usage),
        }
        self._finish(job_id, "done", f"{facts.headline.label}: {facts.headline.display}", "result", result, usage=result["usage"])

    # ---- helpers
    @staticmethod
    def _checks(log_entries):
        return [{"step": e.step, "message": e.message, "ok": e.ok} for e in log_entries]

    @staticmethod
    def _usage(sam_frames, llm_usage):
        return {"sam_frames_billed": int(sam_frames or 0), "sam_cost_usd": round((sam_frames or 0) * SAM_USD_PER_FRAME, 5),
                "llm_calls": llm_usage.calls, "llm_tokens": llm_usage.input_tokens + llm_usage.output_tokens,
                "llm_cost_usd": round(llm_usage.cost_usd, 6) if llm_usage.cost_usd is not None else None}

    def _needs(self, plan, known):
        """What must be answered before measuring: missing inputs, and numbers the user said that could not be checked."""
        missing = [vars(m) for m in plan.missing]
        wanted = used_values(plan.tool, plan.outputs, plan.compare_with_ramp, plan.quantity)
        confirm = []
        for name, v in plan.values.items():
            if name in wanted and v.needs_confirmation and v.source != "given":
                confirm.append({"name": name, "said": v.said, "value": v.value, "unit": v.unit, "reason": v.note,
                                "question": f"Please confirm: is the {name.replace('_', ' ')} {v.said}"
                                            + (f" ({v.value:g} {v.unit})" if v.unit and v.said.replace(' ', '') != f"{v.value:g}{v.unit}" else "")
                                            + "?" + (f" {v.note}" if v.note else "")})
        # a scale needs an object to measure it from
        if (plan.tool in runner.SCALE_NEEDED and "reference_length" in plan.values and not plan.reference_phrases
                and not known.get("scale_m_per_px") and not any(m["name"] == "reference_object" for m in missing)):
            missing.append({"name": "reference_object", "dimension": None,
                            "question": "Which object in the picture has that known length? Name it (for example 'metre stick')."})
        return {"missing": missing, "confirm": confirm, "notes": plan.notes, "interpretation": plan.interpretation}

    def _save_tracks(self, job_id, checked, info):
        folder = self.data_dir / "jobs" / job_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "tracks.json").write_text(json.dumps(runner.tracks_payload(checked.tracks, info)))
        (folder / "data.csv").write_text(runner.tracks_csv(checked.tracks, info))
