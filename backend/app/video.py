"""Video handling: probe with ffprobe, remux for fast start, upload once to Meta.

Phones often record variable frame rate, so we read the real timestamp of
every frame instead of assuming frame / fps.
"""
import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


@dataclass
class VideoInfo:
    fps: float
    frame_count: int
    width: int  # displayed size, after applying the rotation flag
    height: int
    duration: float
    rotation: int
    codec: str
    timestamps: list  # seconds, one per frame


def _run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def probe(path) -> VideoInfo:
    """Read fps, size, rotation and per-frame timestamps with ffprobe."""
    meta = json.loads(_run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams",
                            "-show_format", "-of", "json", str(path)]))
    if not meta.get("streams"):
        raise ValueError("No video stream found in this file.")
    s = meta["streams"][0]
    rotation = 0
    for sd in s.get("side_data_list", []):
        if "rotation" in sd:
            rotation = int(sd["rotation"])
    rotation = rotation or int(s.get("tags", {}).get("rotate", 0))
    w, h = int(s["width"]), int(s["height"])
    if abs(rotation) % 180 == 90:
        w, h = h, w

    raw = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(path)])
    ts = sorted(float(v) for v in raw.replace(",", " ").split() if v != "N/A")
    if len(ts) < 2:
        raise ValueError("Could not read frame timestamps from this video.")
    duration = ts[-1] - ts[0]
    fps = (len(ts) - 1) / duration if duration > 0 else float(Fraction(s["avg_frame_rate"]))
    return VideoInfo(fps=fps, frame_count=len(ts), width=w, height=h, duration=duration,
                     rotation=rotation, codec=s.get("codec_name", ""), timestamps=ts)


def prepare(src, dst) -> Path:
    """Remux to MP4 with faststart so Meta does not have to. WebM/VP9 is converted to H.264."""
    info = probe(src)
    if info.codec in ("h264", "hevc", "mpeg4"):
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-c", "copy", "-movflags", "+faststart", str(dst)]
    else:
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-an", "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True)
    return Path(dst)


def upload(client, path) -> str:
    """Upload once to the Meta Files API and return the file_id to reuse."""
    with open(path, "rb") as f:
        return client.files.create(purpose="user_data", file=f).id
