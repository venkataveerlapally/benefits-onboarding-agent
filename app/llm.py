"""Thin LLM wrapper with a free-first provider order.

Providers (set LLM_PROVIDER, or leave as "auto"):
  ollama     - FREE. Open models (Llama, Qwen, Mistral...) running locally via https://ollama.com.
               No API key, no account, data never leaves your machine.
  anthropic  - Paid. Used only if you set ANTHROPIC_API_KEY and install `anthropic`.
  none       - No LLM. Deterministic fallbacks everywhere (also free).

"auto" tries Ollama first, then Anthropic (if a key is set), then falls back to no LLM.
Every caller handles a None response, so the pipeline always runs.
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from functools import lru_cache

log = logging.getLogger(__name__)
_local = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # Ollama is local: never route via a proxy

PROVIDER = os.getenv("LLM_PROVIDER", "auto").lower()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
ANTHROPIC_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")


@lru_cache(maxsize=1)
def _ollama_up() -> bool:
    try:
        with _local.open(f"{OLLAMA_URL}/api/tags", timeout=1.5) as r:
            models = [m["name"].split(":")[0] for m in json.load(r).get("models", [])]
        if OLLAMA_MODEL.split(":")[0] not in models:
            log.warning("Ollama is running but model '%s' isn't pulled. Run: ollama pull %s", OLLAMA_MODEL, OLLAMA_MODEL)
            return False
        return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _anthropic_ok() -> bool:
    if not os.getenv("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False


def active_provider() -> str | None:
    if os.getenv("DISABLE_LLM") == "1" or PROVIDER == "none":
        return None
    if PROVIDER in ("ollama", "auto") and _ollama_up():
        return "ollama"
    if PROVIDER in ("anthropic", "auto") and _anthropic_ok():
        return "anthropic"
    return None


def available() -> bool:
    return active_provider() is not None


def model_label() -> str:
    p = active_provider()
    return {"ollama": f"Ollama · {OLLAMA_MODEL} (free, local)", "anthropic": f"Claude · {ANTHROPIC_MODEL}"}.get(p, "off")


def _ollama_chat(system: str, user: str, max_tokens: int, json_mode: bool) -> str:
    body = {
        "model": OLLAMA_MODEL, "stream": False,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "options": {"temperature": 0.1, "num_predict": max_tokens},
    }
    if json_mode:
        body["format"] = "json"  # Ollama constrains output to valid JSON
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with _local.open(req, timeout=120) as r:
        return json.load(r)["message"]["content"]


def _anthropic_chat(system: str, user: str, max_tokens: int) -> str:
    import anthropic
    resp = anthropic.Anthropic().messages.create(
        model=ANTHROPIC_MODEL, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


def complete(system: str, user: str, max_tokens: int = 1500, json_mode: bool = False) -> str | None:
    """Return the model's text, or None if no LLM is available or the call fails."""
    provider = active_provider()
    if provider is None:
        return None
    try:
        if provider == "ollama":
            return _ollama_chat(system, user, max_tokens, json_mode)
        return _anthropic_chat(system, user, max_tokens)
    except Exception as e:  # never let an LLM hiccup crash onboarding
        log.warning("LLM call failed (%s), falling back to heuristics: %s", provider, e)
        return None


def complete_json(system: str, user: str, max_tokens: int = 1500) -> dict | None:
    """Ask for JSON and parse it defensively (handles ```json fences and stray prose)."""
    text = complete(system + "\nRespond with a single JSON object and nothing else.", user, max_tokens, json_mode=True)
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        log.warning("LLM returned invalid JSON")
        return None
