"""Choosing object names SAM can see: the other-number variants, the ordering, and the job wiring."""
import asyncio

from app import phrases
from tests.fakes import Harness
from tests.test_jobs import env, job, run  # noqa: F401


def test_flip_number_changes_only_the_last_word():
    assert phrases.flip_number("green sprouts") == "green sprout"
    assert phrases.flip_number("steel ball") == "steel balls"
    assert phrases.flip_number("green bottles") == "green bottle"
    assert phrases.flip_number("small box") == "small boxes"
    assert phrases.flip_number("boxes") == "box"
    assert phrases.flip_number("party") == "parties"
    assert phrases.flip_number("parties") == "party"
    assert phrases.flip_number("glass") == "glasses"


def test_candidates_keep_the_planners_order_and_add_variants_once():
    assert phrases.candidates(["sprouts", "seedlings"]) == ["sprouts", "seedlings", "sprout", "seedling"]
    assert phrases.candidates(["ball", "balls"]) == ["ball", "balls"]  # no repeats


def test_choose_puts_names_that_found_something_first_and_keeps_the_rest_as_fallbacks():
    planned = ["silver cart", "spring"]
    counts = {"silver cart": 0, "spring": 2, "silver carts": 0, "springs": 2}
    chosen = phrases.choose(planned, counts)
    assert chosen[0] == "spring" and "silver cart" in chosen and len(chosen) <= 3
    # nothing found anywhere: leave the planner's names alone
    assert phrases.choose(planned, {"silver cart": 0, "spring": 0}) == planned
    # a failed check (None) is not a 'found nothing'
    assert phrases.choose(planned, {"silver cart": None, "spring": None}) == planned


def test_for_counting_the_name_that_finds_more_objects_wins():
    planned = ["green bottle"]
    counts = {"green bottle": 1, "green bottles": 19}
    assert phrases.choose(planned, counts, prefer_many=True)[0] == "green bottles"
    assert phrases.choose(planned, counts)[0] == "green bottle"


def test_describe_is_plain_and_mentions_the_name_used_first():
    planned = ["sprouts"]
    counts = {"sprouts": 0, "sprout": 3}
    text = phrases.describe(planned, counts, phrases.choose(planned, counts))
    assert "'sprouts' found nothing" in text and "'sprout' found 3" in text and "Tracking 'sprout' first" in text
    assert phrases.describe(planned, {}, planned) == ""


def test_probe_counts_tries_the_middle_picture_first_and_survives_failures():
    class R:  # a SAM answer with two boxes
        output_text = "<0f>0<|box;x1=1;y1=1;x2=5;y2=5|>,1<|box;x1=9;y1=9;x2=20;y2=20|>"

    class Client:
        def __init__(self):
            self.calls = []
            self.responses = self

        async def create(self, **kw):
            text = kw["input"][0]["content"][0]["text"]
            url = kw["input"][0]["content"][1]["image_url"]
            self.calls.append((text, url))
            if text == "broken":
                raise RuntimeError("boom")
            return R() if text == "found" else type("E", (), {"output_text": ""})()

    c = Client()
    counts = asyncio.run(phrases.probe_counts(c, ["a", "b", "c"], ["found", "missing", "broken"]))
    assert counts == {"found": 2, "missing": 0, "broken": None}
    assert c.calls[0][1] == "b"  # the middle picture first
    assert [u for t, u in c.calls if t == "missing"] == ["b", "a", "c"]  # nothing there: the others are tried
    assert [u for t, u in c.calls if t == "found"] == ["b"]  # found at once: one picture is enough


def test_the_job_tracks_the_name_that_the_sample_picture_found_first(env):  # noqa: F811
    store, tmp, vid = env
    seen = {}

    async def probe(frames, names):
        seen["names"] = names
        return {n: (3 if n == "marble" else 0) for n in names}

    h = Harness(store, tmp, probe_fn=probe)
    j = job(store, run(h, vid, "What is the swing period?"))
    assert j["status"] == "done", j["message"]
    assert seen["names"][0] == "glass marble" and "glass marbles" in seen["names"]
    # nothing in the fake picture check matched the planner's name, so its own name is tracked as before
    assert h.trackers[0].phrases[0] == "glass marble"
    line = [c for c in j["result"]["checks"] if c["step"] == "names"]
    assert len(line) == 1 and "found nothing" in line[0]["message"] and "first" not in line[0]["message"]


def test_a_name_found_on_the_sample_picture_is_tracked_first_and_logged(env):  # noqa: F811
    store, tmp, vid = env

    async def probe(frames, names):
        return {n: (2 if n == "glass marbles" else 0) for n in names}

    h = Harness(store, tmp, probe_fn=probe)
    j = job(store, run(h, vid, "What is the swing period?"))
    assert h.trackers[0].phrases[0] == "glass marbles"
    line = [c for c in j["result"]["checks"] if c["step"] == "names"]
    assert len(line) == 1 and "'glass marbles' found 2" in line[0]["message"]
