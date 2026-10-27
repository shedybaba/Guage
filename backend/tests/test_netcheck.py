"""The network self-check must report where requests go without ever leaking a secret."""
import httpx

from app import netcheck


def test_env_shows_names_but_never_secret_values(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "sk-secret-value")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-other-secret")
    monkeypatch.setenv("HTTPS_PROXY", "http://user:hunter2@proxy.example:3128")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://elsewhere.example/v1")
    shown = str(netcheck._env())
    assert "sk-secret-value" not in shown and "sk-other-secret" not in shown and "hunter2" not in shown
    assert netcheck._env()["MODEL_API_KEY"] == "set"
    assert netcheck._env()["HTTPS_PROXY"] == "set, host proxy.example"
    assert netcheck._env()["OPENAI_BASE_URL"] == "https://elsewhere.example/v1"


def test_headers_hide_cookies_and_caller_tokens():
    h = netcheck._headers(httpx.Headers({"Set-Cookie": "a=b", "proxy-status": "e_clientaddr=xyz", "x-request-id": "r1"}))
    assert h == {"set-cookie": "(hidden)", "proxy-status": "(hidden)", "x-request-id": "r1"}
