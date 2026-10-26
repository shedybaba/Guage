"""Tests for the model wrapper (SparkLLM) and the keyframe helper, with a fake client: no network, no cost."""
import asyncio
import base64
import subprocess
from types import SimpleNamespace

import pytest

from app import video
from app.llm import LLMError, SparkLLM, Usage, usage_from

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["a"], "properties": {"a": {"type": "number"}}}


def response(text, status="completed", inp=100, out=50, reasoning=30):
    return SimpleNamespace(output_text=text, status=status,
                           usage=SimpleNamespace(input_tokens=inp, output_tokens=out,
                                                 output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning)))


class FakeClient:
    def __init__(self, *items):
        self.items, self.calls = list(items), []
        self.responses = SimpleNamespace(create=self.create)

    async def create(self, **kw):
        self.calls.append(kw)
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def ask(client, **kw):
    return asyncio.run(SparkLLM(client, **kw).json("be brief", [{"type": "input_text", "text": "hi"}], SCHEMA, "thing"))


def test_a_complete_answer_is_parsed_and_costed():
    r = ask(FakeClient(response('{"a": 1.5}', inp=1_000_000, out=1_000_000)), model="muse-spark-1.3-contributor")
    assert r.data == {"a": 1.5} and r.attempts == 1
    assert r.usage.cost_usd == pytest.approx(0.10 + 0.20)  # a million tokens each way at the contributor prices
    assert r.usage.input_tokens == 1_000_000 and r.usage.reasoning_tokens == 30 and r.usage.calls == 1


def test_the_standard_model_costs_more_and_an_unknown_one_has_no_price():
    assert ask(FakeClient(response('{"a": 1}', inp=1_000_000, out=0)), model="muse-spark-1.3").usage.cost_usd == pytest.approx(1.25)
    assert ask(FakeClient(response('{"a": 1}')), model="something-new").usage.cost_usd is None


def test_the_request_has_the_shape_the_live_api_accepted():
    client = FakeClient(response('{"a": 1}'))
    ask(client, model="m", effort="minimal", max_output_tokens=1234)
    kw = client.calls[0]
    assert kw["model"] == "m" and kw["instructions"] == "be brief" and kw["max_output_tokens"] == 1234
    assert kw["reasoning"] == {"effort": "minimal"}
    assert kw["input"] == [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}]
    assert kw["text"] == {"format": {"type": "json_schema", "name": "thing", "schema": SCHEMA, "strict": True}}


def test_an_answer_cut_off_by_the_token_budget_is_retried_with_double_the_budget():
    client = FakeClient(response('{"a":', status="incomplete", out=1000), response('{"a": 2}', out=900))
    r = ask(client, max_output_tokens=1000)
    assert r.data == {"a": 2} and r.attempts == 2
    assert [c["max_output_tokens"] for c in client.calls] == [1000, 2000]
    assert r.usage.calls == 2 and r.usage.output_tokens == 1900  # both attempts are paid for, and counted


def test_broken_json_is_retried():
    client = FakeClient(response('{"a": '), response('{"a": 3}'))
    assert ask(client).data == {"a": 3}


def test_giving_up_after_the_attempts_is_an_error_not_a_guess():
    client = FakeClient(*[response("not json at all") for _ in range(3)])
    with pytest.raises(LLMError, match="complete JSON"):
        ask(client)
    assert len(client.calls) == 3


def test_an_api_error_becomes_an_llm_error_without_leaking_anything_else():
    with pytest.raises(LLMError, match="request failed"):
        ask(FakeClient(RuntimeError("503 service unavailable")))


def test_a_json_list_is_not_accepted_as_the_object_we_asked_for():
    client = FakeClient(response("[1, 2]"), response("[1, 2]"), response("[1, 2]"))
    with pytest.raises(LLMError):
        ask(client)


def test_usage_adds_up():
    u = Usage(10, 5, 2, 1, 0.1).add(Usage(20, 6, 3, 1, 0.2))
    assert (u.input_tokens, u.output_tokens, u.reasoning_tokens, u.calls) == (30, 11, 5, 2) and u.cost_usd == pytest.approx(0.3)
    assert Usage(1, 1, 0, 1, None).add(Usage(1, 1, 0, 1, 0.1)).cost_usd is None  # one unknown price makes the total unknown


def test_usage_without_a_usage_block_is_unpriced_not_crashing():
    assert usage_from(SimpleNamespace(output_text="{}"), "m").cost_usd is None


def test_keyframes_are_small_decodable_jpegs_spread_through_the_clip(tmp_path):
    clip = tmp_path / "c.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=640x480:rate=30", "-t", "3",
                    "-pix_fmt", "yuv420p", str(clip)], check=True)
    frames = video.keyframes(clip, n=3, width=240)
    assert len(frames) == 3 and len(set(frames)) == 3  # three different moments
    for f in frames:
        assert f.startswith("data:image/jpeg;base64,")
        raw = base64.b64decode(f.split(",", 1)[1])
        assert raw[:2] == b"\xff\xd8" and len(raw) < 60_000  # a real JPEG, and small
