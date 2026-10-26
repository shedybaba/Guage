"""When Meta no longer has an uploaded video (files expire after 30 days), Gauge uploads it again once and carries on."""
import asyncio
from types import SimpleNamespace

import httpx
import openai
import pytest

from app import main, sam, video
from app.store import Store


def not_found(code="file_not_found"):
    req = httpx.Request("POST", "https://api.meta.ai/v1/responses")
    return openai.NotFoundError("No file found.", response=httpx.Response(404, request=req), body={"code": code})


def done():
    return SimpleNamespace(records=[], diagnostics=[], raw_output=""), {"usage": {"video_frames_processed": 50}}


class Script:
    """A fake `sam._stream`: pops one outcome per call (an exception to raise, or a result) and records the file ids used."""

    def __init__(self, *outcomes):
        self.outcomes, self.used = list(outcomes), []

    async def __call__(self, client, file_id, text, metadata, progress):
        self.used.append(file_id)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def run(file_id, tmp_path, monkeypatch, script):
    monkeypatch.setattr(sam, "DATA_DIR", tmp_path)
    monkeypatch.setattr(sam, "_stream", script)
    return asyncio.run(sam._run(None, file_id, "marble", {}, "key", "marble", use_cache=False))


def provider():
    calls = []

    async def file_id(stale=None):
        calls.append(stale)
        return "new-id" if stale else "old-id"
    file_id.calls = calls
    return file_id


def test_a_missing_file_is_uploaded_again_once_and_the_request_repeated(tmp_path, monkeypatch):
    script, p = Script(not_found(), done()), provider()
    result = run(p, tmp_path, monkeypatch, script)
    assert script.used == ["old-id", "new-id"]
    assert p.calls == [None, "old-id"]  # the second call says which id was found missing
    assert result.frames_billed == 50


def test_it_does_not_loop_if_the_new_upload_is_missing_too(tmp_path, monkeypatch):
    script = Script(not_found(), not_found())
    with pytest.raises(openai.NotFoundError):
        run(provider(), tmp_path, monkeypatch, script)
    assert len(script.used) == 2


def test_other_not_found_errors_are_not_retried(tmp_path, monkeypatch):
    script, p = Script(not_found("model_not_found")), provider()
    with pytest.raises(openai.NotFoundError):
        run(p, tmp_path, monkeypatch, script)
    assert len(script.used) == 1 and p.calls == [None]


def test_a_fixed_id_cannot_be_refreshed_so_the_error_is_raised(tmp_path, monkeypatch):
    with pytest.raises(openai.NotFoundError):
        run("123", tmp_path, monkeypatch, Script(not_found()))


def test_the_factory_uploads_again_only_if_the_stored_id_is_the_missing_one(tmp_path, monkeypatch):
    store = Store(tmp_path / "t.sqlite3")
    store.add_video("v1", "c.mp4", tmp_path / "o.mp4", tmp_path / "p.mp4", "key", {"width": 320, "height": 240, "fps": 30, "frame_count": 100})
    store.set_file_id("v1", "old-id")
    uploads = []
    monkeypatch.setattr(video, "upload", lambda client, path: uploads.append(path) or f"new-id-{len(uploads)}")
    import app.meta_client as mc
    monkeypatch.setattr(mc, "sync_client", lambda: None)
    monkeypatch.setattr(mc, "async_client", lambda: None)
    tracker = asyncio.run(main.SamTrackerFactory(store, tmp_path)(store.get_video("v1"), None))
    ask = tracker.file_id
    assert asyncio.run(ask()) == "old-id" and uploads == []  # normal use: no upload
    assert asyncio.run(ask(stale="old-id")) == "new-id-1"  # Meta lost it: upload again and remember the new id
    assert asyncio.run(ask(stale="old-id")) == "new-id-1" and len(uploads) == 1  # a second question already refreshed it
    assert store.get_video("v1")["file_id"] == "new-id-1"
    store.close()
