"""Calls to Muse Spark (Meta Model API, Responses API) that return JSON checked against a schema.

Facts found on the live API:
- text.format with type json_schema works, strict mode included (nullable fields as anyOf or a type list).
- Reasoning cannot be switched off and its tokens share max_output_tokens. A small budget therefore cuts the JSON
  off halfway (seen at 500 tokens), so the budget is generous (cost is tiny) and a cut-off answer is retried
  with double the budget.
Nothing here ever prints a key or sends one anywhere but Meta's endpoint (see meta_client.py).
"""
import json
import logging
from dataclasses import dataclass, field

from .config import SPARK_MODEL

# USD per million tokens (input, output), from the pricing page
log = logging.getLogger("gauge.llm")
# Tried in this order if Meta answers 'model_not_found': the cheap contributor model is not in Meta's public model list and
# has answered 404 on a deployed server while working elsewhere; the standard model is listed and costs about a cent a question.
FALLBACK_MODELS = ("muse-spark-1.3",)
PRICES = {"muse-spark-1.3-contributor": (0.10, 0.20), "muse-spark-1.3": (1.25, 4.25)}


class LLMError(RuntimeError):
    """The model could not give a usable answer."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    calls: int = 0
    cost_usd: float | None = 0.0

    def add(self, other):
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.calls += other.calls
        self.cost_usd = None if self.cost_usd is None or other.cost_usd is None else self.cost_usd + other.cost_usd
        return self


@dataclass
class JsonResult:
    data: dict
    usage: Usage
    attempts: int = 1
    raw: str = field(default="", repr=False)


def usage_from(response, model):
    u = getattr(response, "usage", None)
    if u is None:
        return Usage(calls=1, cost_usd=None)
    details = getattr(u, "output_tokens_details", None)
    inp, out = int(getattr(u, "input_tokens", 0) or 0), int(getattr(u, "output_tokens", 0) or 0)
    price = PRICES.get(model)
    return Usage(inp, out, int(getattr(details, "reasoning_tokens", 0) or 0), 1,
                 (inp * price[0] + out * price[1]) / 1e6 if price else None)


class SparkLLM:
    """One place that talks to Muse Spark. `client` is an AsyncOpenAI pointed at Meta (or a test double)."""

    def __init__(self, client, model=None, effort="minimal", max_output_tokens=4000, max_attempts=3, fallbacks=FALLBACK_MODELS):
        self.client, self.model = client, model or SPARK_MODEL
        self.fallbacks = [m for m in fallbacks if m != self.model]
        self.effort, self.max_output_tokens, self.max_attempts = effort, max_output_tokens, max_attempts

    async def json(self, instructions, parts, schema, name, effort=None):
        """Ask for JSON matching `schema`. parts: input_text / input_image content parts for the user message."""
        budget, usage, last = self.max_output_tokens, Usage(), ""
        for attempt in range(1, self.max_attempts + 1):
            try:
                r = await self.client.responses.create(
                    model=self.model, instructions=instructions,
                    input=[{"type": "message", "role": "user", "content": parts}],
                    reasoning={"effort": effort or self.effort}, max_output_tokens=budget,
                    text={"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}})
            except Exception as e:  # network / API errors: the SDK has already retried what is retryable
                if getattr(e, "code", None) == "model_not_found" and self.fallbacks:
                    log.warning("model %s was not found; switching to %s", self.model, self.fallbacks[0])
                    self.model = self.fallbacks.pop(0)  # sticky: later questions keep using the model that works
                    continue
                raise LLMError(f"The model request failed ({type(e).__name__}): {str(e)[:200]}") from e
            usage.add(usage_from(r, self.model))
            last = getattr(r, "output_text", "") or ""
            if getattr(r, "status", "completed") == "incomplete":  # ran out of tokens (reasoning shares the budget)
                budget *= 2
                continue
            try:
                data = json.loads(last)
            except json.JSONDecodeError:
                budget *= 2
                continue
            if isinstance(data, dict):
                return JsonResult(data, usage, attempt, last)
        raise LLMError("The model did not return a complete JSON answer.")
