"""Tests for spoken questions: audio conversion, the request sent to Meta, and the /api/transcribe route."""
import asyncio
import json
import subprocess

import httpx
import pytest
from fastapi.testclient import TestClient

from app import config, transcribe
from app.main import create_app
from app.store import Store
from tests.fakes import Harness


def tone(path, seconds=2.0, fmt_args=("-c:a", "libopus")):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", *fmt_args, str(path)], check=True)
    return path.read_bytes()


class FakeTranscriber:
    def __init__(self, text="what is the swing period", error=None):
        self.text, self.error, self.calls = text, error, 0

    async def transcribe(self, wav):
        self.calls += 1
        assert wav.read_bytes()[:4] == b"RIFF"
        if self.error:
            raise self.error
        return self.text


@pytest.fixture
def make_client(tmp_path):
    clients = []

    def make(transcriber):
        store = Store(tmp_path / f"t{len(clients)}.sqlite3")
        app = create_app(store=store, runner=Harness(store, tmp_path).runner, data_dir=tmp_path,
                         frontend_dir=tmp_path / "none", transcriber=transcriber)
        c = TestClient(app, raise_server_exceptions=False)
        c.__enter__()
        clients.append((c, store))
        return c

    yield make
    for c, store in clients:
        c.__exit__(None, None, None)
        store.close()


async def run(factory, handler, wav):
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        return await factory(client).transcribe(wav)


def send(client, data, name="q.webm", type_="audio/webm"):
    return client.post("/api/transcribe", files={"file": (name, data, type_)})


def test_converts_browser_audio_to_the_wav_meta_requires(tmp_path):
    src = tmp_path / "in.webm"
    tone(src)
    seconds = transcribe.to_wav(src, tmp_path / "out.wav")
    assert seconds == pytest.approx(2.0, abs=0.1)
    import wave
    with wave.open(str(tmp_path / "out.wav")) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 24000)


def test_unreadable_audio_is_a_plain_error(tmp_path):
    (tmp_path / "x.bin").write_bytes(b"not audio at all")
    with pytest.raises(transcribe.TranscribeError, match="could not read"):
        transcribe.to_wav(tmp_path / "x.bin", tmp_path / "x.wav")


def test_request_sent_to_meta_matches_the_documented_format(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    seen = {}

    def handler(request: httpx.Request):
        seen["url"], seen["auth"], seen["body"] = str(request.url), request.headers["authorization"], request.content
        return httpx.Response(200, json={"sessionId": "s", "transcript": " What is the swing period?. ", "audioDurationMs": 1000, "turns": []})

    wav = tmp_path / "a.wav"
    tone_src = tmp_path / "a.webm"
    tone(tone_src, 1.0)
    transcribe.to_wav(tone_src, wav)
    text = asyncio.run(run(transcribe.MetaTranscriber, handler, wav))
    assert text == "What is the swing period?"
    assert seen["url"] == "https://api.meta.ai/v1/asr/transcribe"
    assert seen["auth"] == "Bearer test-key"
    body = seen["body"]
    assert b'name="request"' in body and b'name="audio"' in body and b"RIFF" in body
    settings = json.loads(body.split(b'name="request"', 1)[1].split(b"\r\n\r\n", 1)[1].split(b"\r\n--", 1)[0])
    assert settings["model"] == "muse-voice-transcribe-1.0" and settings["audioEncoding"] == "WAV" and settings["mode"] == "PUSH_TO_TALK"


@pytest.mark.parametrize("status,expected", [(429, 429), (400, 502), (500, 502)])
def test_meta_errors_become_plain_messages(tmp_path, monkeypatch, status, expected):
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    src, wav = tmp_path / "a.webm", tmp_path / "a.wav"
    tone(src, 1.0)
    transcribe.to_wav(src, wav)
    with pytest.raises(transcribe.TranscribeError) as e:
        asyncio.run(run(transcribe.MetaTranscriber, lambda r: httpx.Response(status, text="secret internal detail"), wav))
    assert e.value.status == expected and "secret" not in str(e.value)


def test_route_returns_the_text(make_client, tmp_path):
    fake = FakeTranscriber()
    r = send(make_client(fake), tone(tmp_path / "q.webm", 2.0))
    assert r.status_code == 200 and r.json()["text"] == "what is the swing period" and r.json()["seconds"] == pytest.approx(2.0, abs=0.2)


def test_route_works_with_other_browser_formats(make_client, tmp_path):
    data = tone(tmp_path / "q.m4a", 1.5, ("-c:a", "aac"))
    assert send(make_client(FakeTranscriber()), data, "q.m4a", "audio/mp4").status_code == 200


def test_recordings_are_deleted_afterwards(make_client, tmp_path):
    send(make_client(FakeTranscriber()), tone(tmp_path / "q.webm", 1.0))
    assert not list((tmp_path / "tmp").glob("*/*"))


def test_empty_garbage_short_and_long_recordings_are_refused(make_client, tmp_path):
    fake = FakeTranscriber()
    c = make_client(fake)
    assert send(c, b"").status_code == 400
    assert send(c, b"garbage" * 100).status_code == 400
    assert send(c, tone(tmp_path / "s.webm", 0.1)).status_code == 400
    long = send(c, tone(tmp_path / "l.webm", 40.0))
    assert long.status_code == 413 and "30 seconds" in long.json()["detail"]
    assert fake.calls == 0  # nothing was sent (or billed) for any of them


def test_silence_gives_a_helpful_message(make_client, tmp_path):
    r = send(make_client(FakeTranscriber(text="")), tone(tmp_path / "q.webm", 1.0))
    assert r.status_code == 422 and "No speech" in r.json()["detail"]


def test_service_errors_reach_the_user_plainly(make_client, tmp_path):
    err = transcribe.TranscribeError("The speech service is busy. Please try again in a moment.", 429)
    r = send(make_client(FakeTranscriber(error=err)), tone(tmp_path / "q.webm", 1.0))
    assert r.status_code == 429 and "busy" in r.json()["detail"]


def test_oversized_upload_is_refused_before_conversion(make_client, monkeypatch):
    monkeypatch.setattr(config, "MAX_AUDIO_MB", 1)
    fake = FakeTranscriber()
    assert send(make_client(fake), b"x" * (1024 * 1024 + 10)).status_code == 413 and fake.calls == 0


def test_hourly_cap(make_client, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VOICE_PER_HOUR", 2)
    c, data = make_client(FakeTranscriber()), tone(tmp_path / "q.webm", 1.0)
    assert [send(c, data).status_code for _ in range(3)] == [200, 200, 429]
