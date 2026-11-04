"""Pick object names SAM can really see, before paying for a video pass.

SAM is picky about wording: on real clips "steel ball" found 7 balls and "steel balls" found none, "sprout" found
three sprouts and "sprouts" none, while "green bottles" found 19 and "green bottle" none. A wrong name used to cost a
whole video pass (about 17 US cents for a 30 second clip) before Gauge tried the next one. A picture request costs a
fraction of a cent, so each candidate name is tried on one sample picture first, and the names that find something
go to the video request first.
"""
import asyncio

MAX_CANDIDATES = 6
MAX_NAMES = 3  # the tracking loop tries at most this many names


def flip_number(phrase: str) -> str:
    """The same name in the other number: 'sprouts' -> 'sprout', 'bottle' -> 'bottles' (only the last word changes)."""
    words = phrase.split()
    w = words[-1]
    if w.endswith("ies") and len(w) > 4:
        alt = w[:-3] + "y"
    elif w.endswith(("ches", "shes", "xes", "sses")):
        alt = w[:-2]
    elif w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        alt = w[:-1]
    elif w.endswith(("ch", "sh", "x", "ss")):
        alt = w + "es"
    elif w.endswith("y") and len(w) > 2 and w[-2] not in "aeiou":
        alt = w[:-1] + "ies"
    else:
        alt = w + "s"
    return " ".join(words[:-1] + [alt])


def candidates(planned) -> list:
    """The planner's names in order, then each one in the other number, without repeats."""
    out = []
    for p in list(planned) + [flip_number(p) for p in planned]:
        if p and p not in out:
            out.append(p)
    return out[:MAX_CANDIDATES]


def choose(planned, counts, prefer_many=False):
    """Names for the video pass: the ones that found something first (best first), then the planner's own as fallbacks.

    `counts` maps a name to the number of objects SAM found on the sample pictures (None = the check itself failed).
    `prefer_many`: for counting, a name that finds more of the objects is the better name.
    """
    cands = candidates(planned)
    found = [p for p in cands if counts.get(p)]
    if not found:
        return list(planned)
    if prefer_many:
        found.sort(key=lambda p: -counts[p])
    rest = [p for p in planned if p not in found]
    return (found + rest)[:MAX_NAMES]


def describe(planned, counts, chosen) -> str:
    """One plain sentence for the 'how Gauge checked its work' log."""
    parts = []
    for p in candidates(planned):
        n = counts.get(p)
        if n is None:
            continue
        parts.append(f"'{p}' found {n}" if n else f"'{p}' found nothing")
    if not parts:
        return ""
    head = "Tried the object names on a sample picture before tracking: " + "; ".join(parts) + "."
    if list(chosen) != list(planned):
        head += f" Tracking '{chosen[0]}' first."
    return head


async def probe_counts(client, frames, phrases, model="sam-3.1") -> dict:
    """Objects SAM finds for each name: on the middle sample picture, and on the others only if it found nothing there."""
    if not frames:
        return {}
    order = [frames[len(frames) // 2]] + [f for i, f in enumerate(frames) if i != len(frames) // 2]

    async def one(phrase):
        try:
            for url in order:
                r = await client.responses.create(
                    model=model, metadata={"include_confidence": "true", "mask_encoding": "one_bit"},
                    input=[{"type": "message", "role": "user", "content": [
                        {"type": "input_text", "text": phrase}, {"type": "input_image", "image_url": url}]}])
                n = (getattr(r, "output_text", "") or "").count("<|box")
                if n:
                    return phrase, n
            return phrase, 0
        except Exception:  # noqa: BLE001 - a failed check must never stop the question
            return phrase, None

    return dict(await asyncio.gather(*[one(p) for p in phrases]))
