"""OpenAI SDK clients pointed at Meta's OpenAI-compatible API.

The SDK retries 429 and 5xx errors with exponential backoff and honours
Retry-After. A 400 is never retried: it means our request is wrong.
"""
from openai import AsyncOpenAI, OpenAI

from .config import META_BASE_URL, meta_api_key

MAX_RETRIES = 5


def sync_client() -> OpenAI:
    return OpenAI(base_url=META_BASE_URL, api_key=meta_api_key(), max_retries=MAX_RETRIES)


def async_client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=META_BASE_URL, api_key=meta_api_key(), max_retries=MAX_RETRIES)
