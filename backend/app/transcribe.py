"""Spoken questions: audio from the browser -> text, with Muse Voice Transcribe.

The browser records webm/opus (or mp4/aac on Safari). Meta's endpoint takes only mono 16-bit PCM WAV at 16 or 24 kHz,
so ffmpeg converts first. The text goes back to the question box for the user to read and correct; it is never
asked automatically, because a mis-heard number would ruin a measurement.
"""
import json
import re
import subprocess
import wave
from pathlib import Path

import httpx

from .config import META_BASE_URL, meta_api_key

MODEL = "muse-voice-transcribe-1.0"
MAX_SECONDS = 30.0
SAMPLE_RATE = 24000
# Words the model should lean towards. This only biases recognition (Meta: "does not guarantee exact spelling").
KEYWORDS = ["pendulum", "period", "frequency", "spring", "oscillation", "acceleration", "momentum", "collision",
            "metre", "centimetres", "kilograms", "grams", "Gauge"]


class TranscribeError(Exception):
    """A problem the user can be told about in plain words."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def to_wav(src: Path, dest: Path) -> float:
    """Convert any audio ffmpeg can read to mono 16-bit WAV. Returns its length in seconds."""
    try:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
                        "-c:a", "pcm_s16le", "-map_metadata", "-1", str(dest)], check=True, capture_output=True, timeout=30)
        with wave.open(str(dest)) as w:
            return w.getnframes() / w.getframerate()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, wave.Error, OSError):
        raise TranscribeError("Gauge could not read that recording. Please try again, or type your question.") from None


class MetaTranscriber:
    """Sends a WAV file to Meta and returns the transcript text."""

    def __init__(self, client: httpx.AsyncClient | None = None):
        self._client = client

    async def transcribe(self, wav: Path) -> str:
        settings = {"mode": "PUSH_TO_TALK", "model": MODEL, "audioEncoding": "WAV", "keywords": KEYWORDS}
        client = self._client or httpx.AsyncClient(timeout=60)
        try:
            r = await client.post(f"{META_BASE_URL}/asr/transcribe", headers={"Authorization": f"Bearer {meta_api_key()}"},
                                  files={"request": (None, json.dumps(settings), "application/json"),
                                         "audio": (wav.name, wav.read_bytes(), "audio/wav")})
        except httpx.HTTPError:
            raise TranscribeError("The speech service could not be reached. Please type your question instead.", 502) from None
        finally:
            if self._client is None:
                await client.aclose()
        if r.status_code == 429:
            raise TranscribeError("The speech service is busy. Please try again in a moment.", 429)
        if r.status_code >= 400:
            raise TranscribeError("The speech service could not transcribe that recording. Please type your question instead.", 502)
        return re.sub(r"([.?!])\.+$", r"\1", str(r.json().get("transcript", "")).strip())  # Meta sometimes ends "metre.."
