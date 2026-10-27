"""Where do this machine's Meta requests actually go? A deployment self-check.

Prints (and `GET /api/health/network` returns) the same facts on any machine, so a laptop and a server can be compared
line by line: the host and path the SDK really requests, what DNS resolves the host to, which proxy or SDK variables are
set (names only), the TLS certificate and the address the connection reached, and what Meta answers to a models request
with and without the key (status, response headers, short error code, never any key or message text).

Run it locally with `cd backend && .venv/bin/python -m app.netcheck`.
"""
import asyncio
import hashlib
import json
import os
import platform
import socket
import ssl
from urllib.parse import urlsplit

import httpx
import openai

from . import config

# Environment variables that change where or how a request goes. Only whether they are set (and a proxy's HOST) is shown.
WATCHED_ENV = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy",
               "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_API_KEY",
               "OPENAI_ORG_ID", "OPENAI_ORGANIZATION", "OPENAI_PROJECT_ID", "MODEL_API_KEY", "GAUGE_SPARK_MODEL"]
PROXY_VARS = {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"}
SHOWN_VALUE_VARS = {"NO_PROXY", "no_proxy", "OPENAI_BASE_URL", "OPENAI_API_BASE", "GAUGE_SPARK_MODEL"}  # not secret: shown as set


def _env() -> dict:
    out = {}
    for name in WATCHED_ENV:
        value = os.environ.get(name)
        if value is None:
            continue
        if name in PROXY_VARS:  # a proxy URL can hold a password: show the host only
            out[name] = "set, host " + (urlsplit(value).hostname or "?")
        elif name in SHOWN_VALUE_VARS:
            out[name] = value
        else:
            out[name] = "set"
    return out


def _dns(host: str) -> dict:
    try:
        infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except OSError as e:
        return {"error": type(e).__name__}
    return {"addresses": sorted({i[4][0] for i in infos})}


def _tls(host: str) -> dict:
    """Connect once, read the certificate and the address actually reached. Sends no HTTP request."""
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((host, 443), timeout=10) as raw:
            peer = raw.getpeername()[0]
            with ctx.wrap_socket(raw, server_hostname=host) as tls:
                der = tls.getpeercert(binary_form=True)
                cert = tls.getpeercert()
                subject = dict(x[0] for x in cert.get("subject", ()))
                issuer = dict(x[0] for x in cert.get("issuer", ()))
                return {"connected_to": peer, "tls": tls.version(), "cert_cn": subject.get("commonName"),
                        "cert_issuer": issuer.get("organizationName") or issuer.get("commonName"),
                        "cert_not_after": cert.get("notAfter"), "cert_sha256": hashlib.sha256(der).hexdigest()[:16],
                        "cert_names": [v for k, v in cert.get("subjectAltName", ()) if k == "DNS"][:6]}
    except Exception as e:  # noqa: BLE001
        return {"error": type(e).__name__}


def _headers(h: httpx.Headers) -> dict:
    hidden = {"set-cookie", "proxy-status"}  # proxy-status carries opaque tokens that encode the caller's address
    return {k: ("(hidden)" if k.lower() in hidden else v) for k, v in h.items()}


async def _models_request(with_key: bool) -> dict:
    """GET <base>/models exactly as the SDK sends it (or with no key), recording the URL, the status and the answer's headers."""
    seen = {}

    async def on_request(request: httpx.Request):
        seen["method"], seen["url"] = request.method, str(request.url)
        seen["request_header_names"] = sorted(k.lower() for k in request.headers.keys())

    http = httpx.AsyncClient(event_hooks={"request": [on_request]}, timeout=20)
    out = {}
    try:
        if with_key:
            key = os.environ.get("MODEL_API_KEY")
            if not key:
                return {"skipped": "no MODEL_API_KEY"}
            client = openai.AsyncOpenAI(base_url=config.META_BASE_URL, api_key=key, max_retries=0, http_client=http)
            try:
                listing = await client.models.list()
                out["status"] = 200
                out["models"] = sorted(m.id for m in listing.data)
            except openai.APIStatusError as e:
                out["status"] = e.status_code
                out["error_code"] = getattr(e, "code", None) if isinstance(getattr(e, "code", None), str) else None
                out["response_headers"] = _headers(e.response.headers)
            except Exception as e:  # noqa: BLE001
                out["error"] = type(e).__name__
        else:
            r = await http.get(config.META_BASE_URL + "/models")
            out["status"] = r.status_code
            out["response_headers"] = _headers(r.headers)
            try:
                err = r.json().get("error", {})
                out["error_code"] = err.get("code") if isinstance(err.get("code"), str) else None
            except Exception:  # noqa: BLE001
                pass
    except Exception as e:  # noqa: BLE001
        out["error"] = type(e).__name__
    finally:
        await http.aclose()
    out.update(seen)
    return out


async def collect() -> dict:
    base = urlsplit(config.META_BASE_URL)
    host = base.hostname
    loop = asyncio.get_running_loop()
    dns, tls = await asyncio.gather(loop.run_in_executor(None, _dns, host), loop.run_in_executor(None, _tls, host))
    with_key, without_key = await asyncio.gather(_models_request(True), _models_request(False))
    return {
        "machine": {"hostname": socket.gethostname(), "system": platform.platform(), "python": platform.python_version(),
                    "openai_sdk": openai.__version__, "httpx": httpx.__version__},
        "configured": {"base_url": config.META_BASE_URL, "host": host, "path": base.path,
                       "models_url": config.META_BASE_URL + "/models",
                       "responses_url": config.META_BASE_URL + "/responses",
                       "asr_url": config.META_BASE_URL + "/asr/transcribe",
                       "language_model": config.SPARK_MODEL, "sam_model": config.SAM_MODEL},
        "environment": _env(),
        "dns": dns,
        "tls": tls,
        "models_request_with_key": with_key,
        "models_request_without_key": without_key,
    }


if __name__ == "__main__":
    print(json.dumps(asyncio.run(collect()), indent=2, sort_keys=True))
