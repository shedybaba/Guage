"""Tests for the HTTP API through the real FastAPI app, with fake model and fake SAM behind the job runner."""
import gzip
import time

import pytest
from fastapi.testclient import TestClient

from app import config, sam
from app.main import create_app
from app.store import Store, new_id
from tests.fakes import Harness, make_clip


@pytest.fixture(scope="module")
def clip_bytes(tmp_path_factory):
    path = make_clip(tmp_path_factory.mktemp("clip") / "c.mp4")
    return path.read_bytes()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(sam, "DATA_DIR", tmp_path)  # the SAM cache lives in the test folder
    store = Store(tmp_path / "t.sqlite3")
    harness = Harness(store, tmp_path)
    app = create_app(store=store, runner=harness.runner, data_dir=tmp_path, frontend_dir=tmp_path / "no-frontend")
    with TestClient(app, raise_server_exceptions=False) as c:
        c.harness, c.store, c.tmp = harness, store, tmp_path
        yield c
    store.close()


def upload(client, data, name="clip.mp4"):
    return client.post("/api/videos", files={"file": (name, data, "video/mp4")})


@pytest.fixture
def video_id(client, clip_bytes):
    r = upload(client, clip_bytes)
    assert r.status_code == 201, r.text
    return r.json()["video_id"]


def wait(client, job_id, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "failed", "needs_input"):
            return j
        time.sleep(0.05)
    raise AssertionError("the job never finished")


def ask(client, vid, question, known=None):
    body = {"question": question, **({"known_values": known} if known else {})}
    r = client.post(f"/api/videos/{vid}/questions", json=body)
    assert r.status_code == 202, r.text
    return r.json()["job_id"]


# ------------------------------------------------------------ health and uploads

def test_health_says_what_is_running_and_never_shows_a_key(client):
    r = client.get("/api/health").json()
    assert r["status"] == "ok" and r["sam_model"] == "sam-3.1" and "muse-spark" in r["language_model"]
    assert isinstance(r["api_key_configured"], bool)
    assert "key" not in " ".join(str(v) for k, v in r.items() if k != "api_key_configured").lower()


def test_uploading_a_video_returns_what_gauge_learned_about_it(client, clip_bytes):
    r = upload(client, clip_bytes)
    d = r.json()
    assert r.status_code == 201
    assert set(d) == {"video_id", "fps", "frame_count", "width", "height", "duration"}
    assert d["frame_count"] == 360 and d["width"] == 320 and d["height"] == 240 and abs(d["fps"] - 30) < 0.1
    assert (client.tmp / "videos" / d["video_id"] / "prepared.mp4").exists()


def test_a_file_that_is_not_a_video_is_refused_and_leaves_nothing_behind(client):
    r = upload(client, b"this is not a video at all" * 100, "notes.mp4")
    assert r.status_code == 400 and "does not look like a video" in r.json()["detail"]
    assert list((client.tmp / "videos").iterdir()) == []


def test_an_empty_file_is_refused(client):
    assert upload(client, b"").status_code == 400


def test_a_file_over_the_size_limit_is_refused_while_uploading(client, clip_bytes, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 0.01)  # 10 KB
    r = upload(client, clip_bytes)
    assert r.status_code == 413 and "limit" in r.json()["detail"]
    assert list((client.tmp / "videos").iterdir()) == []


def test_a_too_long_video_is_refused(client, clip_bytes, monkeypatch):
    monkeypatch.setattr(config, "MAX_FRAMES", 100)
    assert upload(client, clip_bytes).status_code == 400


def test_a_hostile_file_name_cannot_escape_the_data_folder(client, clip_bytes):
    r = upload(client, clip_bytes, "../../../../tmp/evil.mp4")
    assert r.status_code == 201
    vid = r.json()["video_id"]
    assert (client.tmp / "videos" / vid / "original.mp4").exists()
    assert not (client.tmp.parent / "evil.mp4").exists()


def test_video_info_lists_the_questions_asked(client, video_id):
    jid = ask(client, video_id, "What is the swing period?")
    wait(client, jid)
    info = client.get(f"/api/videos/{video_id}").json()
    assert info["frame_count"] == 360 and info["questions"] == [
        {"job_id": jid, "question": "What is the swing period?", "status": "done"}]


def test_the_video_can_be_played_with_range_requests(client, video_id):
    full = client.get(f"/api/videos/{video_id}/video")
    assert full.status_code == 200 and full.headers["content-type"] == "video/mp4" and len(full.content) > 1000
    part = client.get(f"/api/videos/{video_id}/video", headers={"Range": "bytes=0-99"})
    assert part.status_code == 206 and len(part.content) == 100 and part.headers["content-range"].startswith("bytes 0-99/")


# ------------------------------------------------------------ asking and answers

def test_asking_a_question_gives_a_job_that_finishes_with_an_answer(client, video_id):
    jid = ask(client, video_id, "What is the swing period?")
    j = wait(client, jid)
    assert j["status"] == "done" and j["progress"] == 1.0 and j["video_id"] == video_id
    assert j["result"]["headline"]["units"] == "s" and j["plan"]["tool"] == "pendulum_period"
    assert j["plan"]["interpretation"] and j["needs"] is None and j["error"] is None


def test_a_paused_question_can_be_answered_with_units(client, video_id):
    jid = ask(client, video_id, "What value of g does that give?")
    j = wait(client, jid)
    assert j["status"] == "needs_input" and j["needs"]["missing"][0]["name"] == "string_length"
    r = client.post(f"/api/jobs/{jid}/answers", json={"answers": {"string_length": {"value": 50, "unit": "cm"}}})
    assert r.status_code == 202
    j2 = wait(client, r.json()["job_id"])
    assert j2["status"] == "done" and any(n["id"] == "g" for n in j2["result"]["numbers"])
    assert j2["result"]["inputs"][0]["display"] == "0.5 m"  # 50 cm was converted to metres by code
    assert j2["parent_id"] == jid and client.get(f"/api/jobs/{jid}").json()["parent_id"] is None  # the answer links back


def test_answers_are_validated(client, video_id):
    jid = ask(client, video_id, "What value of g does that give?")
    wait(client, jid)
    url = f"/api/jobs/{jid}/answers"
    assert client.post(url, json={"answers": {"string_length": {"value": 5, "unit": "kg"}}}).status_code == 422  # wrong unit
    assert client.post(url, json={"answers": {"string_length": {"value": 5000, "unit": "m"}}}).status_code == 422  # implausible
    assert client.post(url, json={"answers": {"string_length": {"unit": "m"}}}).status_code == 422  # no value
    assert client.post(url, json={"answers": {"banana": {"value": 1, "unit": "m"}}}).status_code == 422
    assert client.post(url, json={"answers": {"reference_object": {}}}).status_code == 422


def test_only_a_paused_question_can_be_answered(client, video_id):
    jid = ask(client, video_id, "What is the swing period?")
    wait(client, jid)
    assert client.post(f"/api/jobs/{jid}/answers", json={"answers": {}}).status_code == 409


def test_the_question_is_validated(client, video_id):
    url = f"/api/videos/{video_id}/questions"
    assert client.post(url, json={"question": ""}).status_code == 422
    assert client.post(url, json={"question": "x" * 1001}).status_code == 422
    assert client.post(url, json={"question": "swing period", "known_values": {"evil": 1}}).status_code == 422
    assert client.post(url, json={}).status_code == 422
    assert client.post(f"/api/videos/{new_id()}/questions", json={"question": "swing period"}).status_code == 404


def test_daily_budget_returns_a_clear_message(client, video_id):
    client.harness.runner.limits.daily_sam_frames = 100
    client.store.update_job(client.store.create_job(video_id, "old"), usage={"sam_frames_billed": 500})
    r = client.post(f"/api/videos/{video_id}/questions", json={"question": "swing period"})
    assert r.status_code == 503 and "daily usage limit" in r.json()["detail"]


# ------------------------------------------------------------ live events

def read_events(client, jid, **kw):
    with client.stream("GET", f"/api/jobs/{jid}/events", **kw) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        text = "".join(r.iter_text())
    events = []
    for block in text.split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line and not line.startswith(":"))
        if "event" in fields:
            events.append(fields)
    return events


def test_the_event_stream_replays_the_whole_story_and_ends(client, video_id):
    jid = ask(client, video_id, "What is the swing period?")
    wait(client, jid)
    ev = read_events(client, jid)
    kinds = [e["event"] for e in ev]
    assert kinds[0] == "status" and "plan" in kinds and "log" in kinds and kinds[-2:] == ["result", "end"]
    ids = [int(e["id"]) for e in ev if "id" in e]
    assert ids == sorted(ids)


def test_a_reconnecting_browser_continues_where_it_left_off(client, video_id):
    jid = ask(client, video_id, "What is the swing period?")
    wait(client, jid)
    ev = read_events(client, jid)
    middle = int([e for e in ev if "id" in e][2]["id"])
    rest = read_events(client, jid, headers={"Last-Event-ID": str(middle)})
    assert all(int(e["id"]) > middle for e in rest if "id" in e) and rest[-1]["event"] == "end"
    again = read_events(client, jid, params={"after": middle})
    assert [e.get("id") for e in again] == [e.get("id") for e in rest]


def test_the_stream_of_a_paused_job_also_ends(client, video_id):
    jid = ask(client, video_id, "What value of g does that give?")
    wait(client, jid)
    assert read_events(client, jid)[-1]["event"] == "end"


def test_a_live_job_streams_as_it_goes(client, video_id):
    client.harness.runner.tracker_kwargs = None
    client.harness.trackers.clear()
    jid = ask(client, video_id, "What is the swing period?")
    ev = read_events(client, jid)  # opened while the job runs
    assert ev[-1]["event"] == "end" and any(e["event"] == "result" for e in ev)


# ------------------------------------------------------------ results and downloads

def test_tracks_for_the_overlay_player(client, video_id):
    jid = ask(client, video_id, "What is the swing period?")
    wait(client, jid)
    d = client.get(f"/api/jobs/{jid}/tracks").json()
    assert d["width"] == 320 and d["frame_count"] == 360 and set(d["objects"]) == {"0"}
    o = d["objects"]["0"]
    assert len(o["frames"]) == len(o["box"]) == len(o["centre"]) == len(o["t"]) == 360 and o["t"][1] > 0


def test_the_csv_download(client, video_id):
    jid = ask(client, video_id, "What is the swing period?")
    wait(client, jid)
    for url in (f"/api/jobs/{jid}/data.csv", f"/api/videos/{video_id}/data.csv", f"/api/videos/{video_id}/data.csv?job_id={jid}"):
        r = client.get(url)
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
        assert r.text.splitlines()[0].startswith("frame,time_s,object_id") and len(r.text.splitlines()) == 361


def test_no_data_yet_is_a_clear_404(client, video_id):
    assert client.get(f"/api/videos/{video_id}/data.csv").status_code == 404
    jid = ask(client, video_id, "What value of g does that give?")
    wait(client, jid)
    assert client.get(f"/api/jobs/{jid}/tracks").status_code == 404
    assert client.get(f"/api/jobs/{jid}/data.csv").status_code == 404


def test_sam_segmentation_is_served_from_the_cache(client, video_id):
    key = client.store.get_video(video_id)["key"]
    assert client.get(f"/api/videos/{video_id}/segmentation", params={"phrase": "glass marble"}).status_code == 404
    path = sam._cache_file(key, "glass marble")
    path.parent.mkdir(parents=True, exist_ok=True)
    import json
    path.write_text(json.dumps({"phrase": "glass marble", "frames_billed": 360, "diagnostics": [], "points": [
        {"frame": 0, "object_id": "0", "x1": 1, "y1": 2, "x2": 3, "y2": 4, "confidence": 0.9, "cx": 2, "cy": 3, "area": 4,
         "long_side": 2}]}))
    d = client.get(f"/api/videos/{video_id}/segmentation", params={"phrase": "glass marble"}).json()
    assert d["points"][0]["box"] == [1, 2, 3, 4] and d["frames_billed"] == 360
    assert client.get(f"/api/videos/{video_id}/segmentation", params={"phrase": "glass marble", "format": "raw"}).status_code == 404
    with gzip.open(path.with_suffix(".raw.gz"), "wt") as f:
        f.write("<0f>0<|box;x1=1;y1=2;x2=3;y2=4;w=320;h=240|>")
    raw = client.get(f"/api/videos/{video_id}/segmentation", params={"phrase": "glass marble", "format": "raw"})
    assert raw.status_code == 200 and raw.text.startswith("<0f>0")


# ------------------------------------------------------------ safety

def test_ids_that_are_not_ids_are_404_not_errors(client):
    for url in ("/api/jobs/../../etc/passwd", "/api/jobs/nope", "/api/videos/%2e%2e", f"/api/jobs/{new_id()}",
                f"/api/videos/{new_id()}", f"/api/jobs/{'g' * 32}/events"):
        assert client.get(url).status_code == 404, url


def test_cors_allows_the_frontend_and_nobody_else(client):
    ok = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert ok.headers["access-control-allow-origin"] == "http://localhost:5173"
    bad = client.get("/api/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in bad.headers


def test_a_server_error_does_not_leak_details(client, video_id, monkeypatch):
    def boom(job_id):
        raise RuntimeError("database password is hunter2")
    monkeypatch.setattr(client.store, "get_job", boom)
    r = client.get(f"/api/jobs/{new_id()}")
    assert r.status_code == 500 and "hunter2" not in r.text and "recorded" in r.json()["detail"]


def test_the_api_documents_itself(client):
    spec = client.get("/openapi.json").json()
    assert {"/api/videos", "/api/videos/{video_id}/questions", "/api/jobs/{job_id}", "/api/jobs/{job_id}/events"} <= set(spec["paths"])
    assert client.get("/docs").status_code == 200


def test_a_restart_fails_jobs_that_were_in_progress(tmp_path):
    store = Store(tmp_path / "r.sqlite3")
    vid = new_id()
    store.add_video(vid, "c.mp4", "o", "p", "k", {"fps": 30.0, "frame_count": 100, "width": 10, "height": 10, "duration": 3.0})
    stuck = store.create_job(vid, "q")
    store.update_job(stuck, status="tracking")
    app = create_app(store=store, runner=Harness(store, tmp_path).runner, data_dir=tmp_path, frontend_dir=tmp_path / "no-frontend")
    with TestClient(app) as c:
        j = c.get(f"/api/jobs/{stuck}").json()
    assert j["status"] == "failed" and "restarted" in j["message"]
    store.close()


# ------------------------------------------------------------ the frontend, served from the same address

@pytest.fixture
def site(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>Gauge</title><div id=root></div>")
    (dist / "assets" / "app-abc123.js").write_text("console.log('hi')")
    (dist / "favicon.svg").write_text("<svg/>")
    (tmp_path / "secret.txt").write_text("do not serve me")
    store = Store(tmp_path / "s.sqlite3")
    app = create_app(store=store, runner=Harness(store, tmp_path).runner, data_dir=tmp_path, frontend_dir=dist)
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    store.close()


def test_the_app_opens_at_the_root_and_at_any_deep_link(site):
    for path in ("/", "/analyze", f"/analyze/{new_id()}", "/api-docs-page", "/anything/at/all"):
        r = site.get(path)
        assert r.status_code == 200 and "<div id=root>" in r.text, path
        assert r.headers["cache-control"] == "no-cache"


def test_built_assets_are_cached_for_ever_and_other_files_are_served(site):
    r = site.get("/assets/app-abc123.js")
    assert r.status_code == 200 and "immutable" in r.headers["cache-control"] and "max-age=31536000" in r.headers["cache-control"]
    assert site.get("/favicon.svg").text == "<svg/>"


def test_every_page_of_the_app_is_reachable_on_the_production_server(site):
    # the pages the frontend routes to: a path the API reserves would show a JSON error instead of the page
    for path in ("/", "/analyze", f"/analyze/{new_id()}", "/developers"):
        r = site.get(path)
        assert r.status_code == 200 and "<div id=root>" in r.text and "detail" not in r.text, path


def test_the_api_and_its_docs_are_never_swallowed_by_the_app(site):
    assert site.get("/api/health").json()["status"] == "ok"
    assert site.get("/api/nothing-here").status_code == 404
    assert "<div id=root>" not in site.get("/api/nothing-here").text
    assert site.get("/docs").status_code == 200 and "swagger" in site.get("/docs").text.lower()
    assert site.get("/openapi.json").status_code == 200


def test_the_app_cannot_be_used_to_read_files_outside_its_folder(site):
    for path in ("/../secret.txt", "/%2e%2e/secret.txt", "/..%2fsecret.txt", "/assets/../../secret.txt"):
        r = site.get(path)
        assert "do not serve me" not in r.text, path


def test_without_a_built_frontend_the_api_still_runs(client):
    assert client.get("/api/health").status_code == 200
    assert client.get("/").status_code == 404


def test_the_server_starts_without_a_model_key_and_fails_questions_plainly(tmp_path, monkeypatch, clip_bytes):
    """A missing MODEL_API_KEY on a new deployment must not crash the server: health says so, questions fail with a clear message."""
    monkeypatch.setenv("MODEL_API_KEY", "")
    monkeypatch.setattr(sam, "DATA_DIR", tmp_path)
    app = create_app(store=Store(tmp_path / "t.sqlite3"), data_dir=tmp_path, frontend_dir=tmp_path / "no-frontend")
    with TestClient(app, raise_server_exceptions=False) as c:
        assert c.get("/api/health").json()["api_key_configured"] is False
        vid = upload(c, clip_bytes).json()["video_id"]
        job = c.post(f"/api/videos/{vid}/questions", json={"question": "What is the swing period?"}).json()["job_id"]
        j = wait(c, job)
        assert j["status"] == "failed"
        assert "MODEL_API_KEY" not in str(j)  # the user is not shown configuration details


# ------------------------------------------------------------ rotation and growth through the whole chain

def test_rotation_speed_is_measured_through_the_api(client, video_id):
    j = wait(client, ask(client, video_id, "How many rpm does the fan turn at?"))
    assert j["status"] == "done" and j["result"]["kind"] == "measurement" and j["result"]["tool"] == "rotation_rate"
    head = j["result"]["headline"]
    assert head["units"] == "rpm" and head["value"] == pytest.approx(90.0, rel=0.01)
    assert {"t", "turns", "fitted"} <= set(j["result"]["chart"])
    assert any(n["id"] == "frequency" for n in j["result"]["numbers"])


def test_total_growth_needs_no_time_and_is_measured(client, video_id):
    j = wait(client, ask(client, video_id, "How much did the plant grow?"))
    assert j["status"] == "done" and j["result"]["tool"] == "growth_over_time"
    head = j["result"]["headline"]
    assert head["units"] == "%" and head["value"] == pytest.approx(133.0, abs=6.0)
    assert any("No real time" in w for w in j["result"]["warnings"])  # the rate cannot be given, and it says so


def test_a_growth_rate_asks_how_long_the_time_lapse_covers_then_answers(client, video_id):
    first = wait(client, ask(client, video_id, "How fast does the plant grow?"))
    assert first["status"] == "needs_input"
    miss = first["needs"]["missing"]
    assert [m["name"] for m in miss] == ["real_duration"] and miss[0]["dimension"] == "duration"
    assert client.harness.tracker_calls == 0  # nothing was tracked or billed while it waited
    r = client.post(f"/api/jobs/{first['job_id']}/answers", json={"answers": {"real_duration": {"value": 3, "unit": "days"}}})
    assert r.status_code == 202, r.text
    j = wait(client, r.json()["job_id"])
    assert j["status"] == "done" and j["result"]["headline"]["label"] == "Average growth rate"
    assert "/" in j["result"]["headline"]["units"]


def test_the_time_lapse_length_can_be_said_in_the_question(client, video_id):
    j = wait(client, ask(client, video_id, "At what rate does the plant grow over three days?"))
    assert j["status"] == "done" and j["result"]["headline"]["label"] == "Average growth rate"
    assert any(i["label"].startswith("real duration") for i in j["result"]["inputs"])


def test_answers_for_the_time_lapse_are_validated(client, video_id):
    first = wait(client, ask(client, video_id, "How fast does the plant grow?"))
    url = f"/api/jobs/{first['job_id']}/answers"
    assert client.post(url, json={"answers": {"real_duration": {"value": 3, "unit": "kg"}}}).status_code == 422  # wrong kind of unit
    assert client.post(url, json={"answers": {"real_duration": {"value": 0.1, "unit": "s"}}}).status_code == 422  # absurdly short


def test_the_models_self_check_reports_the_kind_of_failure_and_never_a_key(client, monkeypatch):
    import app.meta_client as mc

    class Boom:
        class models:
            @staticmethod
            async def list():
                import httpx
                import openai
                req = httpx.Request("GET", "https://api.meta.ai/v1/models")
                raise openai.PermissionDeniedError("secret text sk-ABC", response=httpx.Response(403, request=req), body=None)
    monkeypatch.setattr(mc, "async_client", lambda: Boom())
    r = client.get("/api/health/models")
    assert r.status_code == 200 and r.json()["error"] == "PermissionDeniedError HTTP 403" and "secret" not in r.text and "sk-" not in r.text
    ok = type("L", (), {"data": [type("M", (), {"id": "sam-3.1"})(), type("M", (), {"id": "muse-spark-1.3"})()]})()

    class Good:
        class models:
            @staticmethod
            async def list():
                return ok
    client.app.state.models_cache = {"at": 0.0, "body": None}
    monkeypatch.setattr(mc, "async_client", lambda: Good())
    body = client.get("/api/health/models").json()
    assert body["models"] == ["muse-spark-1.3", "sam-3.1"] and body["wanted_available"]["sam_model"] is True


def test_the_probe_reports_what_happened_from_this_server_without_leaking(client, monkeypatch):
    import httpx
    import openai

    import app.meta_client as mc
    req = httpx.Request("POST", "https://api.meta.ai/v1/responses")

    class Denied:
        class responses:
            @staticmethod
            async def create(**kw):
                raise openai.NotFoundError("secret text sk-ABC", response=httpx.Response(404, request=req),
                                           body={"code": "model_not_found", "message": "secret text sk-ABC"})
    monkeypatch.setattr(mc, "async_client", lambda: Denied())
    r = client.get("/api/health/probe")
    body = r.json()
    assert r.status_code == 200 and "secret" not in r.text and "sk-" not in r.text
    assert body["sam"] == {"worked": False, "error": "NotFoundError HTTP 404 code=model_not_found"}
    assert all(v["worked"] is False for v in body["muse_spark"].values())
    monkeypatch.setattr(mc, "async_client", lambda: (_ for _ in ()).throw(AssertionError("must be cached")))
    assert client.get("/api/health/probe").json() == body  # cached: a second call makes no new request


# ------------------------------------------------------------ per-visitor limits
def test_one_visitor_is_limited_but_another_is_not(client, video_id, monkeypatch):
    monkeypatch.setattr(config, "VISITOR_QUESTIONS_PER_HOUR", 2)
    mine, other = {"x-forwarded-for": "203.0.113.1"}, {"x-forwarded-for": "203.0.113.2"}
    body = {"question": "What is the swing period?"}
    url = f"/api/videos/{video_id}/questions"
    assert client.post(url, json=body, headers=mine).status_code == 202
    assert client.post(url, json=body, headers=mine).status_code == 202
    third = client.post(url, json=body, headers=mine)
    assert third.status_code == 429 and "a lot of questions" in third.json()["detail"]
    assert client.post(url, json=body, headers=other).status_code == 202


def test_uploads_are_limited_per_visitor(client, clip_bytes, monkeypatch):
    monkeypatch.setattr(config, "VISITOR_UPLOADS_PER_HOUR", 1)
    h = {"x-forwarded-for": "203.0.113.7"}
    assert client.post("/api/videos", files={"file": ("a.mp4", clip_bytes, "video/mp4")}, headers=h).status_code == 201
    r = client.post("/api/videos", files={"file": ("b.mp4", clip_bytes, "video/mp4")}, headers=h)
    assert r.status_code == 429
