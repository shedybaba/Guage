"""Tests for the job runner: the whole question-to-answer pipeline with fake model and fake SAM.

Everything inside Gauge runs for real (planner checks, quality loop, tools, scale, number check, store, events).
"""
import asyncio
import json

import pytest

from app.jobs import JobRunner, Limits, SubmitError, clean_known_values
from app.llm import LLMError
from app.store import Store, new_id
from tests.fakes import FPS, Harness, RoutedLLM

N = 360


@pytest.fixture
def env(tmp_path):
    store = Store(tmp_path / "t.sqlite3")
    vid = new_id()
    folder = tmp_path / "videos" / vid
    folder.mkdir(parents=True)
    (folder / "timestamps.json").write_text(json.dumps([i / FPS for i in range(N)]))
    store.add_video(vid, "clip.mp4", folder / "original.mp4", folder / "prepared.mp4", "key123",
                    {"fps": FPS, "frame_count": N, "width": 320, "height": 240, "duration": N / FPS, "rotation": 0, "codec": "h264"})
    yield store, tmp_path, vid
    store.close()


def run(harness, vid, question, known=None, parent=None):
    async def go():
        job_id = await harness.runner.submit(vid, question, known, parent)
        await harness.runner.wait(job_id)
        return job_id
    return asyncio.run(go())


def job(store, jid):
    return store.get_job(jid)


# ------------------------------------------------------------ the happy path

def test_a_period_question_goes_all_the_way_to_an_answer(env):
    store, tmp, vid = env
    h = Harness(store, tmp)
    j = job(store, run(h, vid, "What is the swing period?"))
    assert j["status"] == "done" and j["progress"] == 1.0
    r = j["result"]
    assert r["kind"] == "measurement" and r["answerable"] and r["tool"] == "pendulum_period"
    assert r["headline"]["units"] == "s" and abs(r["headline"]["value"] - 1.2) < 0.01
    assert "±" in r["headline"]["display"] and r["explanation"]["source"] == "model"
    assert r["explanation"]["summary"] == f"The result is {r['headline']['display']}."
    assert r["usage"]["llm_calls"] == 2 and r["usage"]["llm_cost_usd"] == pytest.approx(0.0006)
    assert any(c["step"] == "track" for c in r["checks"])
    assert r["tracks"]["objects"] == ["0"] and r["downloads"]["csv"].endswith("/data.csv")


def test_progress_is_reported_in_order_and_the_log_is_streamed(env):
    store, tmp, vid = env
    jid = run(Harness(store, tmp), vid, "What is the swing period?")
    events = store.events_after(jid)
    kinds = [e["kind"] for e in events]
    assert kinds[0] == "status" and kinds[-1] == "result"
    statuses = [e["message"] for e in events if e["kind"] == "status"]
    assert statuses == ["Reading your question and looking at the video", "Finding and following the object in every frame",
                        "Measuring", "Writing the explanation"]
    assert any(e["kind"] == "plan" for e in events) and any(e["kind"] == "log" for e in events)
    assert [e["id"] for e in events] == sorted(e["id"] for e in events)
    assert any(e["kind"] == "progress" and "frame" in e["message"] for e in events)  # SAM progress came through


def test_the_final_state_is_written_before_the_closing_event(env):
    store, tmp, vid = env
    jid = run(Harness(store, tmp), vid, "What is the swing period?")
    assert store.get_job(jid)["status"] == "done" and store.events_after(jid)[-1]["kind"] == "result"


def test_tracks_and_csv_are_saved(env):
    store, tmp, vid = env
    jid = run(Harness(store, tmp), vid, "What is the swing period?")
    tracks = json.loads((tmp / "jobs" / jid / "tracks.json").read_text())
    assert tracks["objects"]["0"]["frames"][:2] == [0, 1] and len(tracks["objects"]["0"]["box"]) == N
    lines = (tmp / "jobs" / jid / "data.csv").read_text().strip().splitlines()
    assert lines[0].startswith("frame,time_s,object_id") and len(lines) == N + 1


def test_a_question_with_a_stated_number_gives_g(env):
    store, tmp, vid = env
    j = job(store, run(Harness(store, tmp), vid, "What value of g does that give? The string is half a metre."))
    assert j["status"] == "done"
    g = next(n for n in j["result"]["numbers"] if n["id"] == "g")
    assert g["units"] == "m/s^2" and abs(g["value"] - 4 * 3.14159265 ** 2 * 0.5 / 1.2 ** 2) < 0.3
    assert any(i["display"] == "0.5 metre" for i in j["result"]["inputs"])


# ------------------------------------------------------------ pausing for input

def test_a_missing_input_pauses_the_job_before_any_tracking(env):
    store, tmp, vid = env
    h = Harness(store, tmp)
    j = job(store, run(h, vid, "What value of g does that give?"))
    assert j["status"] == "needs_input" and h.tracker_calls == 0  # nothing was billed
    assert [m["name"] for m in j["needs"]["missing"]] == ["string_length"]
    assert "string" in j["message"].lower()
    assert store.events_after(j["id"])[-1]["kind"] == "needs"


def test_answering_the_missing_input_finishes_the_question(env):
    store, tmp, vid = env
    h = Harness(store, tmp)
    first = run(h, vid, "What value of g does that give?")
    second = run(h, vid, "What value of g does that give?", {"string_length_m": 0.5}, parent=first)
    j = job(store, second)
    assert j["status"] == "done" and j["parent_id"] == first and h.tracker_calls == 1
    assert any(n["id"] == "g" for n in j["result"]["numbers"])


def test_a_number_that_could_not_be_checked_is_confirmed_before_it_is_used(env):
    store, tmp, vid = env
    h = Harness(store, tmp)
    j = job(store, run(h, vid, "g? the string is about a forearm"))
    assert j["status"] == "needs_input" and h.tracker_calls == 0
    c = j["needs"]["confirm"][0]
    assert c["name"] == "string_length" and c["value"] == pytest.approx(0.3) and "confirm" in c["question"].lower()
    done = job(store, run(h, vid, "g? the string is about a forearm", {"string_length_m": 0.3}, parent=j["id"]))
    assert done["status"] == "done"  # the user's confirmation (given as a value) is trusted


def test_a_value_the_measurement_does_not_use_never_blocks(env):
    store, tmp, vid = env
    routes = dict(RoutedLLM().routes)
    routes["swing period"] = {**routes["swing period"], "values": [
        {"name": "mass", "value": 5, "unit": "kg", "quote": "no such quote"}]}  # dropped, and irrelevant anyway
    assert job(store, run(Harness(store, tmp, RoutedLLM(routes)), vid, "What is the swing period?"))["status"] == "done"


# ------------------------------------------------------------ honest non-answers

def test_an_unsupported_question_is_answered_honestly_without_tracking(env):
    store, tmp, vid = env
    h = Harness(store, tmp)
    j = job(store, run(h, vid, "How high can she jump?"))
    assert j["status"] == "done" and j["result"]["kind"] == "unsupported" and not j["result"]["answerable"]
    assert j["result"]["reason"] == "Gauge does not measure sports." and len(j["result"]["can_measure"]) >= 5
    assert h.tracker_calls == 0 and j["usage"]["sam_frames_billed"] == 0


def test_an_object_that_cannot_be_found_gives_advice_not_an_error(env):
    store, tmp, vid = env
    j = job(store, run(Harness(store, tmp), vid, "find the ghost"))
    assert j["status"] == "done" and j["result"]["kind"] == "no_result"
    assert "could not find" in j["result"]["reason"] and j["result"]["advice"]
    assert any(c["step"] == "track" and c["ok"] is False for c in j["result"]["checks"])


def test_a_measurement_that_cannot_be_made_says_why(env):
    store, tmp, vid = env
    j = job(store, run(Harness(store, tmp), vid, "still object"))
    r = j["result"]
    assert j["status"] == "done" and r["kind"] == "no_result" and "No clear oscillation" in r["reason"]
    assert (tmp / "jobs" / j["id"] / "tracks.json").exists()  # the tracks are still available


# ------------------------------------------------------------ scale from a reference object

def test_free_fall_measures_its_own_scale_from_the_reference_object(env):
    store, tmp, vid = env
    h = Harness(store, tmp)
    j = job(store, run(h, vid, "What is g from this drop with a stick? The stick is 1 m."))
    assert j["status"] == "done", j["message"]
    r = j["result"]
    g = r["headline"]["value"]
    assert abs(g - 9.81) < 0.5 and r["headline"]["units"] == "m/s^2"  # the scale came from 250 px = 1 m
    assert any("250 pixels" in c["message"] for c in r["checks"])
    assert r["headline"]["uncertainty"] > 0.01 * g  # the scale's own uncertainty is in the answer
    assert h.trackers[0].phrases == ["ball", "metre stick"]


def test_a_known_length_without_a_named_object_asks_which_object(env):
    store, tmp, vid = env
    h = Harness(store, tmp)
    j = job(store, run(h, vid, "drop without a name: what is g? The stick is 1 m."))
    assert j["status"] == "needs_input" and h.tracker_calls == 0
    assert [m["name"] for m in j["needs"]["missing"]] == ["reference_object"]
    done = job(store, run(h, vid, "drop without a name: what is g? The stick is 1 m.",
                          {"reference_object": "metre stick"}, parent=j["id"]))
    assert done["status"] == "done" and abs(done["result"]["headline"]["value"] - 9.81) < 0.5


def test_a_scale_given_directly_skips_tracking_a_reference(env):
    store, tmp, vid = env
    h = Harness(store, tmp)
    j = job(store, run(h, vid, "drop without a name: what is g? The stick is 1 m.",
                       {"scale_m_per_px": 0.004, "scale_uncertainty_fraction": 0.01}))
    assert j["status"] == "done" and h.trackers[0].phrases == ["ball"]
    assert abs(j["result"]["headline"]["value"] - 9.81) < 0.5


# ------------------------------------------------------------ several objects and follow-ups

def test_counting_works_through_the_same_pipeline(env):
    store, tmp, vid = env
    j = job(store, run(Harness(store, tmp), vid, "count them"))
    assert j["status"] == "done" and j["result"]["headline"]["value"] == 3 and j["result"]["tracks"]["objects"] == ["0", "1", "2"]


def test_a_follow_up_keeps_the_numbers_stated_earlier(env):
    store, tmp, vid = env
    h = Harness(store, tmp)
    run(h, vid, "What value of g does that give? The string is half a metre.")
    follow = job(store, run(h, vid, "What value of g does that give?"))  # no length this time
    assert follow["status"] == "done"  # the 0.5 m from the earlier question still applies
    assert follow["plan"]["values"]["string_length"]["source"] == "earlier"


# ------------------------------------------------------------ failures are handled, not leaked

def test_a_model_outage_is_a_plain_failure(env):
    store, tmp, vid = env
    j = job(store, run(Harness(store, tmp, RoutedLLM(fail_plan=LLMError("503 upstream secret-detail"))), vid, "swing period"))
    assert j["status"] == "failed" and j["error"] == "planner"
    assert "secret-detail" not in (j["message"] or "") and "try again" in j["message"]


def test_an_unexpected_error_does_not_leak_its_details(env):
    store, tmp, vid = env
    j = job(store, run(Harness(store, tmp, tracker_kwargs={"boom": RuntimeError("password=hunter2 in /srv/secret")}), vid, "swing period"))
    assert j["status"] == "failed" and j["error"] == "internal"
    text = json.dumps([j["message"]] + [e["message"] for e in store.events_after(j["id"])])
    assert "hunter2" not in text and "/srv/secret" not in text


def test_a_job_that_takes_too_long_is_stopped(env):
    store, tmp, vid = env
    h = Harness(store, tmp, limits=Limits(job_timeout_s=0.3), tracker_kwargs={"delay": 5.0})
    j = job(store, run(h, vid, "swing period"))
    assert j["status"] == "failed" and j["error"] == "timeout" and "too long" in j["message"]


def test_a_bad_explanation_falls_back_to_code_written_text_and_the_job_still_succeeds(env):
    store, tmp, vid = env
    j = job(store, run(Harness(store, tmp, RoutedLLM(fail_explain=True)), vid, "swing period"))
    assert j["status"] == "done" and j["result"]["explanation"]["source"] == "fallback"
    assert "123.456" not in json.dumps(j["result"]["explanation"]["summary"])


def test_only_two_jobs_work_at_once(env):
    store, tmp, vid = env
    h = Harness(store, tmp, limits=Limits(max_active=1), tracker_kwargs={"delay": 0.3})

    async def go():
        a = await h.runner.submit(vid, "swing period")
        b = await h.runner.submit(vid, "swing period")
        await asyncio.sleep(0.15)
        mid = (store.get_job(a)["status"], store.get_job(b)["status"])
        await h.runner.wait(a)
        await h.runner.wait(b)
        return mid, store.get_job(a), store.get_job(b)

    mid, a, b = asyncio.run(go())
    assert mid[1] == "queued" and mid[0] != "queued"  # the second waits its turn
    assert a["status"] == b["status"] == "done"


# ------------------------------------------------------------ limits and validation

def submit_error(h, vid, question="swing period"):
    async def go():
        with pytest.raises(SubmitError) as e:
            await h.runner.submit(vid, question)
        return e.value
    return asyncio.run(go())


def test_empty_long_and_unknown_video_are_refused(env):
    store, tmp, vid = env
    h = Harness(store, tmp)
    assert submit_error(h, vid, "   ").status == 422
    assert submit_error(h, vid, "x" * 1001).status == 422
    assert submit_error(h, new_id()).status == 404


def test_a_busy_gauge_refuses_new_questions(env):
    store, tmp, vid = env
    for _ in range(3):
        store.update_job(store.create_job(vid, "q"), status="tracking")
    assert submit_error(Harness(store, tmp, limits=Limits(max_waiting=3)), vid).status == 429


def test_the_hourly_limit_is_enforced(env):
    store, tmp, vid = env
    for _ in range(3):
        store.create_job(vid, "q")
    e = submit_error(Harness(store, tmp, limits=Limits(jobs_per_hour=3)), vid)
    assert e.status == 429 and "hour" in e.message


def test_the_daily_sam_budget_is_enforced(env):
    store, tmp, vid = env
    store.update_job(store.create_job(vid, "q"), usage={"sam_frames_billed": 5000})
    e = submit_error(Harness(store, tmp, limits=Limits(daily_sam_frames=5000)), vid)
    assert e.status == 503 and "daily" in e.message


def test_known_values_are_cleaned():
    assert clean_known_values({"string_length_m": 0.5, "evil": 1, "mass_kg": "heavy", "scale_m_per_px": float("nan"),
                               "reference_object": " metre stick ", "time_s": True}) == \
        {"string_length_m": 0.5, "reference_object": "metre stick"}
    assert clean_known_values(None) == {}
