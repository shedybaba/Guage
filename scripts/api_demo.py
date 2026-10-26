"""Drive a running Gauge API like a frontend would: upload, ask, watch the live events, answer, download.

    backend/.venv/bin/python scripts/api_demo.py samples/pendulum.mp4 "What is the swing period?"
    backend/.venv/bin/python scripts/api_demo.py samples/pendulum.mp4 "What value of g does that give?" --answer string_length=50cm

Start the server first:  cd backend && .venv/bin/python -m uvicorn app.main:app --port 8000
"""
import argparse
import json
import re
import sys
import time

import httpx

ap = argparse.ArgumentParser()
ap.add_argument("clip")
ap.add_argument("question")
ap.add_argument("--base", default="http://localhost:8000")
ap.add_argument("--video-id", help="reuse an uploaded video")
ap.add_argument("--answer", action="append", default=[], metavar="NAME=VALUEunit", help="e.g. string_length=50cm")
args = ap.parse_args()
c = httpx.Client(base_url=args.base, timeout=120)


def stream(job_id):
    t0 = time.time()
    with c.stream("GET", f"/api/jobs/{job_id}/events") as r:
        event = None
        for line in r.iter_lines():
            if line.startswith("event:"):
                event = line.split(": ", 1)[1]
            elif line.startswith("data:") and event not in ("end", None):
                e = json.loads(line[6:])
                if event in ("status", "plan", "needs", "result", "error", "progress"):
                    print(f"  [{time.time() - t0:5.1f}s] {event:8s} {e['message'][:110]}")
                elif event == "log":
                    print(f"  [{time.time() - t0:5.1f}s]   check   {e['message'][:100]}")


vid = args.video_id
if not vid:
    with open(args.clip, "rb") as f:
        r = c.post("/api/videos", files={"file": (args.clip.split("/")[-1], f, "video/mp4")})
    r.raise_for_status()
    print("uploaded:", r.json())
    vid = r.json()["video_id"]
job = c.post(f"/api/videos/{vid}/questions", json={"question": args.question}).json()["job_id"]
print(f"Q: {args.question}  (job {job[:8]})")
stream(job)
j = c.get(f"/api/jobs/{job}").json()
if j["status"] == "needs_input" and args.answer:
    answers = {}
    for a in args.answer:
        name, rest = a.split("=", 1)
        m = re.fullmatch(r"([\d.]+)\s*([A-Za-z°]*)", rest)
        answers[name] = {"value": float(m.group(1)), "unit": m.group(2)} if m else {"text": rest}
    print("answering:", answers)
    job = c.post(f"/api/jobs/{job}/answers", json={"answers": answers}).json()["job_id"]
    stream(job)
    j = c.get(f"/api/jobs/{job}").json()
print(f"\nSTATUS: {j['status']}  progress {j['progress']:.0%}")
if j["status"] == "needs_input":
    for m in j["needs"]["missing"] + j["needs"]["confirm"]:
        print("  needs:", m["question"])
r = j["result"]
if r and r["answerable"]:
    print("ANSWER:", r["headline"]["label"], "=", r["headline"]["display"])
    print("  ", r["explanation"]["summary"])
    print(f"   explanation by: {r['explanation']['source']};  usage: {r['usage']}")
    csv = c.get(r["downloads"]["csv"])
    print(f"   csv: {len(csv.text.splitlines()) - 1} rows;  tracks: {len(c.get(r['tracks']['url']).json()['objects'])} object(s)")
elif r:
    print("NO ANSWER:", r.get("reason"))
