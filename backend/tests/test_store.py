"""Tests for the SQLite store: videos, jobs, events, restart recovery and the usage cap's input."""
import json
import time

import numpy as np
import pytest

from app.store import Store, jsonable, new_id


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.sqlite3")
    yield s
    s.close()


def add_video(store, vid=None):
    vid = vid or new_id()
    store.add_video(vid, "clip.mp4", "/d/original.mp4", "/d/prepared.mp4", "abc123", {"fps": 30.0, "frame_count": 362})
    return vid


def test_a_video_round_trips_with_its_info(store):
    vid = add_video(store)
    v = store.get_video(vid)
    assert v["info"] == {"fps": 30.0, "frame_count": 362} and v["key"] == "abc123" and v["file_id"] is None
    store.set_file_id(vid, "file-1")
    assert store.get_video(vid)["file_id"] == "file-1"
    assert store.get_video("nope") is None


def test_a_job_starts_queued_and_updates(store):
    vid = add_video(store)
    jid = store.create_job(vid, "What is the period?", {"string_length_m": 0.5})
    j = store.get_job(jid)
    assert j["status"] == "queued" and j["progress"] == 0 and j["known_values"] == {"string_length_m": 0.5}
    store.update_job(jid, status="tracking", progress=0.4, message="Tracking", plan={"tool": "pendulum_period"})
    j = store.get_job(jid)
    assert (j["status"], j["progress"], j["plan"]) == ("tracking", 0.4, {"tool": "pendulum_period"})
    assert j["updated"] >= j["created"]


def test_only_known_fields_can_be_updated(store):
    jid = store.create_job(add_video(store), "q")
    with pytest.raises(ValueError):
        store.update_job(jid, question="changed")


def test_numpy_and_nan_never_break_a_write(store):
    jid = store.create_job(add_video(store), "q")
    result = {"value": np.float64(1.5), "arr": np.array([1.0, np.nan, 3.0]), "n": np.int64(7), "bad": float("inf"),
              "tup": (1, 2), "nested": {"x": np.float32(2.5)}}
    store.update_job(jid, result=result)
    got = store.get_job(jid)["result"]
    assert got == {"value": 1.5, "arr": [1.0, None, 3.0], "n": 7, "bad": None, "tup": [1, 2], "nested": {"x": 2.5}}
    json.dumps(got)


def test_jsonable_handles_objects_with_attributes():
    class Thing:
        def __init__(self):
            self.a, self.b = 1, np.float64(2.0)
    assert jsonable({"t": Thing()}) == {"t": {"a": 1, "b": 2.0}}


def test_events_are_ordered_and_can_be_replayed_from_any_point(store):
    jid = store.create_job(add_video(store), "q")
    ids = [store.add_event(jid, "status", f"step {i}", {"i": i}) for i in range(5)]
    assert ids == sorted(ids)
    assert [e["message"] for e in store.events_after(jid)] == [f"step {i}" for i in range(5)]
    assert [e["data"]["i"] for e in store.events_after(jid, ids[2])] == [3, 4]
    assert store.events_after(jid, ids[-1]) == []


def test_events_of_different_jobs_do_not_mix(store):
    v = add_video(store)
    a, b = store.create_job(v, "a"), store.create_job(v, "b")
    store.add_event(a, "status", "for a")
    store.add_event(b, "status", "for b")
    assert [e["message"] for e in store.events_after(a)] == ["for a"]


def test_jobs_of_a_video_come_back_in_order_and_the_latest_plan_is_found(store):
    v = add_video(store)
    j1, j2, j3 = (store.create_job(v, f"q{i}") for i in range(3))
    store.update_job(j1, plan={"tool": "pendulum_period", "tag": "first"})
    store.update_job(j2, plan={"tool": None, "unsupported": {"reason": "x"}})  # no tool: not useful as context
    assert [j["question"] for j in store.jobs_for_video(v)] == ["q0", "q1", "q2"]
    assert store.latest_plan(v)["tag"] == "first"
    store.update_job(j3, plan={"tool": "count_objects", "tag": "third"})
    assert store.latest_plan(v)["tag"] == "third"
    assert store.latest_plan(add_video(store)) is None


def test_active_jobs_are_counted(store):
    v = add_video(store)
    a, b, c = (store.create_job(v, "q") for _ in range(3))
    store.update_job(b, status="tracking")
    store.update_job(c, status="done")
    assert store.count_active() == 2


def test_a_restart_fails_jobs_that_were_in_progress_and_says_why(store):
    v = add_video(store)
    running, waiting, finished, paused = (store.create_job(v, "q") for _ in range(4))
    store.update_job(running, status="tracking")
    store.update_job(finished, status="done", result={"ok": 1})
    store.update_job(paused, status="needs_input", needs={"missing": []})
    assert store.mark_stale_failed() == 2
    assert store.get_job(running)["status"] == store.get_job(waiting)["status"] == "failed"
    assert "restarted" in store.get_job(running)["message"]
    assert store.get_job(finished)["status"] == "done" and store.get_job(paused)["status"] == "needs_input"
    assert store.events_after(running)[-1]["kind"] == "error"
    assert store.mark_stale_failed() == 0


def test_usage_since_adds_up_billed_frames(store):
    v = add_video(store)
    old, new1, new2 = (store.create_job(v, "q") for _ in range(3))
    store.update_job(old, usage={"sam_frames_billed": 5000})
    store.update_job(new1, usage={"sam_frames_billed": 300})
    store.update_job(new2, usage={"sam_frames_billed": None})
    jobs, frames = store.usage_since(0)
    assert (jobs, frames) == (3, 5300)
    time.sleep(0.01)
    cut = time.time()
    store.create_job(v, "later")
    assert store.usage_since(cut) == (1, 0)


def test_the_store_survives_being_reopened(tmp_path):
    path = tmp_path / "persist.sqlite3"
    s = Store(path)
    vid = add_video(s)
    jid = s.create_job(vid, "q")
    s.add_event(jid, "status", "hello")
    s.close()
    again = Store(path)
    assert again.get_video(vid) is not None and again.get_job(jid)["question"] == "q"
    assert again.events_after(jid)[0]["message"] == "hello"
    again.close()
