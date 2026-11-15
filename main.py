#!/usr/bin/env python3
"""free-model-aggregator-01 entry: one server, two client wires, ten free providers.

Serves the OpenAI Chat Completions API and the Anthropic Messages API (JSON and
SSE streaming) over ten simulated free-tier providers reached through the env at
$MAKEFUN_ENV_URL. Every answer comes from exactly one successful provider call:
the entry picks the best free model meeting each request's requirements (by the
brief's quality table), fails over when a provider refuses, and returns the
provider's answer in the wire the client used.

Python 3.13 standard library only. No clock, no unseeded randomness in any
output: fixed ids and timestamps, deterministic ordering everywhere, so a replay
leaves a byte-identical state export.
"""

import json
import os
import signal
import ssl
import sys
import threading
import time
from http.client import HTTPConnection, HTTPSConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

FIXED_CREATED = 1767225600
FIXED_CREATED_AT = "2026-01-01T00:00:00Z"
PROTOCOL_VERSION = 1
POLL_INTERVAL = 0.1
UPSTREAM_TIMEOUT = 5.0
PROBE_TIMEOUT = 5.0
WORK_TIMEOUT = 2.0
REGISTER_TIMEOUT = 5.0
MAX_BODY = 32 * 1024 * 1024
MAX_ATTEMPTS = 14

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

# Safe per-provider call plan, from the competition's draw rules (budget ranges
# in program/generator/draw.mjs): budget is drawn in [calls_min, calls_max].
# Never plan more than calls_min calls on a provider; exceed it only when no
# other provider can serve. Unknown provider ids fall back to 5.
CALLS_MIN = {
    "groq": 9, "cerebras": 5, "gemini": 6, "zai": 4, "openrouter": 10,
    "sambanova": 4, "pollinations": 10, "cloudflare": 7, "ovh": 5, "kilol": 12,
}


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


# --------------------------------------------------------------------------
# Brief parsing (mirrors starter/lib/providers.ts documentsIn / objectEnd)
# --------------------------------------------------------------------------

def _object_end(text, start):
    depth = 0
    in_str = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _documents_in(text):
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except Exception:
                value = None
            if isinstance(value, dict) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def _string_of(value):
    return value if isinstance(value, str) else ""


def _number_or_null(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return None


def _bool_or_null(value):
    return value if isinstance(value, bool) else None


def _provider_of(value):
    if not isinstance(value, dict):
        return None
    pid = _string_of(value.get("id"))
    wire = value.get("wire")
    if pid == "" or wire not in ("openai", "anthropic", "gemini"):
        return None
    auth = value.get("auth")
    if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
        auth = "bearer"
    base = ""
    for key in ("path", "base_path", "base_url", "base"):
        candidate = _string_of(value.get(key))
        if candidate != "":
            base = candidate
            break
    name = _string_of(value.get("name"))
    return {
        "id": pid,
        "name": name if name != "" else pid,
        "wire": wire,
        "auth": auth,
        "base": base,
        "key_env": _string_of(value.get("key_env")),
        "key": _string_of(value.get("key")),
    }


def _rating_of(value):
    if not isinstance(value, dict):
        return None
    model = _string_of(value.get("model"))
    provider = _string_of(value.get("provider"))
    if model == "" or provider == "":
        return None
    rating = value.get("rating_milli")
    if not isinstance(rating, (int, float)) or isinstance(rating, bool):
        rating = 0
    return {
        "model": model,
        "provider": provider,
        "rating_milli": rating,
        "context_length": _number_or_null(value.get("context_length")),
        "tools": value.get("tools") is True,
        "vision": value.get("vision") is True,
        "json": value.get("json") is True,
        "free": value.get("free") is True,
    }


def _workload_request_of(value):
    if not isinstance(value, dict):
        return None
    rid = _string_of(value.get("id"))
    if rid == "":
        return None
    requires = value.get("requires") if isinstance(value.get("requires"), dict) else {}
    out_requires = {
        "tools": requires.get("tools") is True,
        "vision": requires.get("vision") is True,
        "json": requires.get("json") is True,
    }
    min_context = requires.get("min_context")
    if isinstance(min_context, (int, float)) and not isinstance(min_context, bool):
        out_requires["min_context"] = min_context
    return {
        "id": rid,
        "api": "anthropic" if value.get("api") == "anthropic" else "openai",
        "stream": value.get("stream") is True,
        "weight": 2 if value.get("weight") == 2 else 1,
        "requires": out_requires,
    }


def read_brief(path):
    """Returns (directory_providers, ratings_rows, workload_manifest, error)."""
    if path == "":
        return [], None, None, "MAKEFUN_PROMPT_FILE is not set"
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except Exception as exc:
        return [], None, None, "cannot read %s: %s" % (path, exc)
    docs = _documents_in(text)
    directory_doc = None
    for doc in docs:
        if doc.get("schema") == PROVIDERS_SCHEMA and isinstance(doc.get("providers"), list):
            directory_doc = doc
            break
    if directory_doc is None:
        return [], None, None, "the brief at %s holds no %s document" % (path, PROVIDERS_SCHEMA)
    providers = []
    for raw in directory_doc["providers"]:
        parsed = _provider_of(raw)
        if parsed is not None:
            providers.append(parsed)
    ratings = None
    for doc in docs:
        if doc.get("schema") == RATINGS_SCHEMA and isinstance(doc.get("ratings"), list):
            rows = []
            for raw in doc["ratings"]:
                row = _rating_of(raw)
                if row is not None:
                    rows.append(row)
            ratings = rows
            break
    workload = None
    for doc in docs:
        if doc.get("schema") == WORKLOAD_SCHEMA and isinstance(doc.get("manifest"), list):
            rows = []
            for raw in doc["manifest"]:
                row = _workload_request_of(raw)
                if row is not None:
                    rows.append(row)
            workload = rows
            break
    return providers, ratings, workload, None


def rating_of_model(ratings, provider_id, model_id):
    if not ratings:
        return None
    for row in ratings:
        if row["provider"] == provider_id and row["model"] == model_id:
            return row
    return None


def model_free(model, row):
    """The quality table's row where it has one (sealed truth), else the listing."""
    if row is not None:
        return row["free"]
    return model["free"] if model["free"] is not None else False


def capabilities_of(model, row):
    """Row first, listing second; unknown stays None and never meets a requirement."""
    if row is not None:
        return {
            "context_length": row["context_length"] if row["context_length"] is not None else model["context_length"],
            "tools": row["tools"],
            "vision": row["vision"],
            "json": row["json"],
        }
    return {
        "context_length": model["context_length"],
        "tools": model["tools"],
        "vision": model["vision"],
        "json": model["json"],
    }


def meets_requirements(requires, caps):
    if requires.get("tools") is True and caps.get("tools") is not True:
        return False
    if requires.get("vision") is True and caps.get("vision") is not True:
        return False
    if requires.get("json") is True and caps.get("json") is not True:
        return False
    min_context = requires.get("min_context")
    if min_context is not None:
        context = caps.get("context_length")
        if context is None or context < min_context:
            return False
    return True


# --------------------------------------------------------------------------
# HTTP plumbing and the catalog probe
# --------------------------------------------------------------------------

def join_base(env_url, base):
    """A base path joined to the env URL; an absolute base URL is used as-is."""
    b = base.strip()
    if b.lower().startswith("http://") or b.lower().startswith("https://"):
        return b.rstrip("/")
    prefix = env_url.rstrip("/")
    if b == "":
        return prefix
    return prefix + ("" if b.startswith("/") else "/") + b.rstrip("/")


def auth_headers(auth, key):
    if key == "":
        return {}
    if auth == "bearer":
        return {"authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def http_request(url, method="GET", headers=None, body=None, timeout=5.0):
    """One HTTP request. Returns (status, headers_dict_lower, body_bytes)."""
    parts = urlsplit(url)
    host = parts.hostname
    port = parts.port or (443 if parts.scheme == "https" else 80)
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    if parts.scheme == "https":
        conn = HTTPSConnection(host, port, timeout=timeout,
                               context=ssl._create_unverified_context())
    else:
        conn = HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        data = resp.read()
        return resp.status, {k.lower(): v for k, v in resp.getheaders()}, data
    finally:
        conn.close()


class UpstreamResponse:
    """A provider response held open for incremental (SSE) reading."""

    def __init__(self, conn, resp):
        self.conn = conn
        self.resp = resp
        self.status = resp.status
        self.headers = {k.lower(): v for k, v in resp.getheaders()}

    def read_all(self):
        try:
            return self.resp.read()
        finally:
            self.conn.close()

    def iter_frames(self):
        """Yield SSE frames as text, split on blank lines, tolerating CRLF."""
        lines = []
        try:
            while True:
                raw = self.resp.readline()
                if not raw:
                    break
                text = raw.decode("utf-8", "replace")
                if text in ("\n", "\r\n", ""):
                    if lines:
                        yield "\n".join(lines)
                        lines = []
                else:
                    lines.append(text.rstrip("\r\n"))
            if lines:
                yield "\n".join(lines)
        finally:
            self.conn.close()


def http_request_stream(url, method="POST", headers=None, body=None, timeout=5.0):
    """Start a request and return an UpstreamResponse for incremental reading."""
    parts = urlsplit(url)
    host = parts.hostname
    port = parts.port or (443 if parts.scheme == "https" else 80)
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    if parts.scheme == "https":
        conn = HTTPSConnection(host, port, timeout=timeout,
                               context=ssl._create_unverified_context())
    else:
        conn = HTTPConnection(host, port, timeout=timeout)
    conn.request(method, path, body=body, headers=headers or {})
    return UpstreamResponse(conn, conn.getresponse())


def _zero_priced(value):
    pricing = value.get("pricing") if isinstance(value, dict) else None
    if not isinstance(pricing, dict):
        return None
    values = []
    for item in pricing.values():
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            values.append(item)
        elif isinstance(item, str) and item.strip() != "":
            try:
                values.append(float(item))
            except ValueError:
                pass
    if not values:
        return None
    return all(v == 0 for v in values)


def _model_in(wire, value):
    if not isinstance(value, dict):
        return None
    if wire == "gemini":
        model_id = _string_of(value.get("name"))
        if model_id.startswith("models/"):
            model_id = model_id[len("models/"):]
    else:
        model_id = _string_of(value.get("id"))
    if model_id == "":
        return None
    context = _number_or_null(value.get("context_length"))
    if context is None:
        context = _number_or_null(value.get("context_window"))
    if context is None:
        context = _number_or_null(value.get("inputTokenLimit"))
    display = _string_of(value.get("display_name"))
    if display == "":
        display = _string_of(value.get("displayName"))
    if display == "":
        display = model_id
    free = _bool_or_null(value.get("free"))
    if free is None:
        free = _zero_priced(value)
    return {
        "id": model_id,
        "display_name": display,
        "free": free,
        "context_length": context,
        "tools": _bool_or_null(value.get("tools")),
        "vision": _bool_or_null(value.get("vision")),
        "json": _bool_or_null(value.get("json")),
    }


def _models_in(wire, body):
    if not isinstance(body, dict):
        return []
    listing = body.get("models") if wire == "gemini" else body.get("data")
    if not isinstance(listing, list):
        return []
    out = []
    for raw in listing:
        parsed = _model_in(wire, raw)
        if parsed is not None:
            out.append(parsed)
    return out


def probe_provider(env_url, provider, timeout=PROBE_TIMEOUT):
    """Probe one provider's catalog route, in its own wire, tolerating failure."""
    base = join_base(env_url, provider["base"])
    url = base + "/v1beta/models" if provider["wire"] == "gemini" else base + "/v1/models"
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    try:
        status, _hdrs, data = http_request(url, "GET", headers, timeout=timeout)
        if status != 200:
            return {"provider": provider, "ok": False, "status": status, "models": [],
                    "error": "GET %s answered HTTP %s" % (url, status)}
        body = json.loads(data.decode("utf-8"))
        return {"provider": provider, "ok": True, "status": status,
                "models": _models_in(provider["wire"], body), "error": None}
    except Exception as exc:
        return {"provider": provider, "ok": False, "status": 0, "models": [],
                "error": "GET %s failed: %s" % (url, exc)}


def build_catalog(probes, ratings):
    """The --list document and the register body: the same object, byte for byte.

    Providers in brief order; models sorted by id, deduplicated. base_url is the
    directory's own path (/p/<id>), never joined to the env URL.
    """
    providers = []
    for probe in probes:
        if not probe["ok"]:
            continue
        seen = set()
        models = []
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            row = rating_of_model(ratings, probe["provider"]["id"], model["id"])
            context = row["context_length"] if row is not None and row["context_length"] is not None else model["context_length"]
            models.append({
                "id": model["id"],
                "free": model_free(model, row),
                "context_length": context,
            })
        providers.append({
            "id": probe["provider"]["id"],
            "base_url": probe["provider"]["base"],
            "key_env": probe["provider"]["key_env"],
            "wire": probe["provider"]["wire"],
            "models": models,
        })
    return {"providers": providers}


def enrich_probes(probes, ratings):
    """Attach each probed model's free flag, capabilities and rating (for routing)."""
    for probe in probes:
        if not probe["ok"]:
            continue
        for model in probe["models"]:
            row = rating_of_model(ratings, probe["provider"]["id"], model["id"])
            model["is_free"] = model_free(model, row)
            model["caps"] = capabilities_of(model, row)
            model["rating"] = row["rating_milli"] if row is not None else 0


def served_models(probes):
    """The free models GET /v1/models lists, in brief provider order, models by id."""
    out = []
    for probe in probes:
        if not probe["ok"]:
            continue
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            if not model.get("is_free"):
                continue
            out.append({"id": model["id"], "provider": probe["provider"]["id"],
                        "display_name": model["display_name"]})
    return out


# --------------------------------------------------------------------------
# Wire translation helpers
# --------------------------------------------------------------------------

def num_or_zero(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return 0


def text_of(content):
    """The text of a string or of a block array's text blocks, joined by \n."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def parse_json_obj(value):
    """A tool call's arguments as an object; {} when it is not one."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip() != "":
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def stop_list(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _image_url_to_anthropic_block(image_url):
    """An OpenAI image_url part as an Anthropic image block."""
    url = image_url.get("url", "") if isinstance(image_url, dict) else ""
    if isinstance(url, str) and url.startswith("data:"):
        mime = "image/png"
        data = url
        rest = url[len("data:"):]
        if ";base64," in rest:
            mime, data = rest.split(";base64,", 1)
        return {"type": "image", "source": {"type": "base64", "media_type": mime or "image/png", "data": data}}
    return {"type": "image", "source": {"type": "url", "url": url}}


def _anthropic_block_to_openai_part(block):
    """One Anthropic content block as an OpenAI content part; None for the rest."""
    if not isinstance(block, dict):
        return None
    if block.get("type") == "text" and isinstance(block.get("text"), str):
        return {"type": "text", "text": block["text"]}
    if block.get("type") == "image" and isinstance(block.get("source"), dict):
        source = block["source"]
        if source.get("type") == "base64" and isinstance(source.get("data"), str):
            media = source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png"
            return {"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (media, source["data"])}}
        if source.get("type") == "url" and isinstance(source.get("url"), str):
            return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def _tool_result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_of(content)
        return text if text != "" else json.dumps(content)
    if content is None:
        return ""
    return json.dumps(content)


def anthropic_to_openai(body, model):
    """An Anthropic Messages request as an OpenAI Chat Completions request.

    Mirrors starter/lib/wire.ts anthropicToOpenAI: the system prompt becomes a
    system message, image blocks become image_url data URLs, tool_use blocks
    become tool_calls, tool_result blocks become tool messages, tools and
    tool_choice are rewritten, and stream gains stream_options.include_usage.
    """
    messages = []
    system = text_of(body.get("system"))
    if system != "":
        messages.append({"role": "system", "content": system})
    source = body.get("messages") if isinstance(body.get("messages"), list) else []
    for raw in source:
        if not isinstance(raw, dict):
            continue
        assistant = raw.get("role") == "assistant"
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": "assistant" if assistant else "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            tool_calls = []
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                tool_calls.append({
                    "id": block.get("id") if isinstance(block.get("id"), str) else "",
                    "type": "function",
                    "function": {
                        "name": block.get("name") if isinstance(block.get("name"), str) else "",
                        "arguments": json.dumps(block.get("input") if isinstance(block.get("input"), dict) else {}),
                    },
                })
            text = text_of(content)
            message = {"role": "assistant", "content": text if text != "" else None}
            if tool_calls:
                message["tool_calls"] = tool_calls
            messages.append(message)
            continue
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                messages.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id") if isinstance(block.get("tool_use_id"), str) else "",
                    "content": _tool_result_text(block.get("content")),
                })
                continue
            part = _anthropic_block_to_openai_part(block)
            if part is not None:
                parts.append(part)
        if parts:
            messages.append({"role": "user", "content": parts})
    out = {"model": model, "messages": messages}
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    stops = stop_list(body.get("stop_sequences"))
    if stops:
        out["stop"] = stops
    tools = body.get("tools")
    if isinstance(tools, list):
        converted = []
        for raw in tools:
            if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
                continue
            converted.append({
                "type": "function",
                "function": {
                    "name": raw["name"],
                    "description": raw.get("description") if isinstance(raw.get("description"), str) else "",
                    "parameters": raw.get("input_schema") if isinstance(raw.get("input_schema"), dict) else {"type": "object", "properties": {}},
                },
            })
        if converted:
            out["tools"] = converted
    choice = body.get("tool_choice")
    if isinstance(choice, dict):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


DEFAULT_ANTHROPIC_MAX_TOKENS = 1024


def openai_to_anthropic_req(body, model):
    """An OpenAI Chat Completions request as an Anthropic Messages request.

    System messages become the top-level system string, image_url parts
    become image blocks, tool_calls become tool_use blocks, and tool
    messages become tool_result blocks. Anthropic requires max_tokens on
    every request, so one is always supplied: the request's own max_tokens
    when it has one, else its max_completion_tokens, else a fixed default
    of 1024. An OpenAI request that omits both caps is still a valid
    request on the OpenAI side, and routing it to an Anthropic-wire
    provider must not turn it into a 400.
    """
    system_parts = []
    messages = []
    source = body.get("messages") if isinstance(body.get("messages"), list) else []
    for raw in source:
        if not isinstance(raw, dict):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if role == "system":
            system_parts.append(text_of(content))
            continue
        if role == "user":
            blocks = []
            if isinstance(content, str):
                blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text" and isinstance(part.get("text"), str):
                        blocks.append({"type": "text", "text": part["text"]})
                    elif part.get("type") == "image_url":
                        blocks.append(_image_url_to_anthropic_block(part.get("image_url") or {}))
            if blocks:
                messages.append({"role": "user", "content": blocks})
            continue
        if role == "assistant":
            blocks = []
            text = text_of(content)
            if text != "":
                blocks.append({"type": "text", "text": text})
            calls = raw.get("tool_calls") if isinstance(raw.get("tool_calls"), list) else []
            for call in calls:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                blocks.append({
                    "type": "tool_use",
                    "id": call.get("id") if isinstance(call.get("id"), str) else "",
                    "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
                    "input": parse_json_obj(fn.get("arguments")),
                })
            if blocks:
                messages.append({"role": "assistant", "content": blocks})
            continue
        if role == "tool":
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": raw.get("tool_call_id") if isinstance(raw.get("tool_call_id"), str) else "",
                "content": text_of(content),
            }]})
    out = {"model": model, "messages": messages}
    system = "\n".join(system_parts)
    if system != "":
        out["system"] = system
    cap = None
    if isinstance(body.get("max_tokens"), (int, float)):
        cap = body["max_tokens"]
    elif isinstance(body.get("max_completion_tokens"), (int, float)):
        cap = body["max_completion_tokens"]
    out["max_tokens"] = int(cap) if cap is not None else DEFAULT_ANTHROPIC_MAX_TOKENS
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    stops = stop_list(body.get("stop"))
    if stops:
        out["stop_sequences"] = stops
    tools = body.get("tools")
    if isinstance(tools, list):
        converted = []
        for raw in tools:
            if not isinstance(raw, dict):
                continue
            fn = raw.get("function") if isinstance(raw.get("function"), dict) else {}
            if not isinstance(fn.get("name"), str):
                continue
            converted.append({
                "name": fn["name"],
                "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
                "input_schema": fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {"type": "object", "properties": {}},
            })
        if converted:
            out["tools"] = converted
    choice = body.get("tool_choice")
    if choice == "auto":
        out["tool_choice"] = {"type": "auto"}
    elif choice == "required":
        out["tool_choice"] = {"type": "any"}
    elif choice == "none":
        out["tool_choice"] = {"type": "none"}
    elif isinstance(choice, dict) and choice.get("type") == "function":
        fn = choice.get("function") if isinstance(choice.get("function"), dict) else {}
        if isinstance(fn.get("name"), str):
            out["tool_choice"] = {"type": "tool", "name": fn["name"]}
    if body.get("stream") is True:
        out["stream"] = True
    return out


def _openai_contents_for_gemini(body):
    """OpenAI messages as Gemini contents; system text returned separately."""
    system_parts = []
    contents = []
    source = body.get("messages") if isinstance(body.get("messages"), list) else []
    for raw in source:
        if not isinstance(raw, dict):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if role == "system":
            system_parts.append(text_of(content))
            continue
        if role == "assistant":
            parts = []
            text = text_of(content)
            if text != "":
                parts.append({"text": text})
            calls = raw.get("tool_calls") if isinstance(raw.get("tool_calls"), list) else []
            for call in calls:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                parts.append({"text": "Tool call: %s %s" % (
                    fn.get("name", ""), json.dumps(parse_json_obj(fn.get("arguments"))))})
            if parts:
                contents.append({"role": "model", "parts": parts})
            continue
        if role == "tool":
            text = text_of(content)
            if text != "":
                contents.append({"role": "user", "parts": [{"text": text}]})
            continue
        parts = []
        if isinstance(content, str):
            parts.append({"text": content})
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    parts.append({"text": part["text"]})
                elif part.get("type") == "image_url":
                    image_url = part.get("image_url") or {}
                    url = image_url.get("url", "") if isinstance(image_url, dict) else ""
                    if isinstance(url, str) and url.startswith("data:"):
                        rest = url[len("data:"):]
                        if ";base64," in rest:
                            mime, data = rest.split(";base64,", 1)
                            parts.append({"inlineData": {"mimeType": mime or "image/png", "data": data}})
                        else:
                            parts.append({"text": url})
                    elif isinstance(url, str) and url != "":
                        parts.append({"text": url})
        if parts:
            contents.append({"role": "user", "parts": parts})
    return "\n".join(system_parts), contents


def _anthropic_contents_for_gemini(body):
    """Anthropic messages as Gemini contents; system text returned separately."""
    system = text_of(body.get("system"))
    contents = []
    source = body.get("messages") if isinstance(body.get("messages"), list) else []
    for raw in source:
        if not isinstance(raw, dict):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if isinstance(content, str):
            contents.append({"role": "model" if role == "assistant" else "user",
                             "parts": [{"text": content}]})
            continue
        if not isinstance(content, list):
            continue
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append({"text": block["text"]})
            elif block.get("type") == "image" and isinstance(block.get("source"), dict):
                source = block["source"]
                if source.get("type") == "base64" and isinstance(source.get("data"), str):
                    media = source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png"
                    parts.append({"inlineData": {"mimeType": media, "data": source["data"]}})
                elif source.get("type") == "url" and isinstance(source.get("url"), str):
                    parts.append({"text": source["url"]})
            elif block.get("type") == "tool_use":
                parts.append({"text": "Tool call: %s %s" % (
                    block.get("name", ""), json.dumps(block.get("input") if isinstance(block.get("input"), dict) else {}))})
            elif block.get("type") == "tool_result":
                parts.append({"text": _tool_result_text(block.get("content"))})
        if parts:
            contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})
    return system, contents


def to_gemini(body, model, client_wire):
    """Either client wire as a Gemini generateContent request body."""
    if client_wire == "openai":
        system_text, contents = _openai_contents_for_gemini(body)
        tools = []
        raw_tools = body.get("tools") if isinstance(body.get("tools"), list) else []
        for raw in raw_tools:
            if not isinstance(raw, dict):
                continue
            fn = raw.get("function") if isinstance(raw.get("function"), dict) else {}
            if isinstance(fn.get("name"), str):
                tools.append({
                    "name": fn["name"],
                    "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
                    "parameters": fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {"type": "object", "properties": {}},
                })
        stops = stop_list(body.get("stop"))
        max_tokens = body.get("max_tokens")
        if max_tokens is None:
            max_tokens = body.get("max_completion_tokens")
        fmt = body.get("response_format") if isinstance(body.get("response_format"), dict) else {}
        json_mode = fmt.get("type") in ("json_object", "json_schema")
    else:
        system_text, contents = _anthropic_contents_for_gemini(body)
        tools = []
        raw_tools = body.get("tools") if isinstance(body.get("tools"), list) else []
        for raw in raw_tools:
            if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
                continue
            tools.append({
                "name": raw["name"],
                "description": raw.get("description") if isinstance(raw.get("description"), str) else "",
                "parameters": raw.get("input_schema") if isinstance(raw.get("input_schema"), dict) else {"type": "object", "properties": {}},
            })
        stops = stop_list(body.get("stop_sequences"))
        max_tokens = body.get("max_tokens")
        json_mode = False
    out = {"model": model, "contents": contents}
    if system_text != "":
        out["systemInstruction"] = {"parts": [{"text": system_text}]}
    if tools:
        out["tools"] = [{"functionDeclarations": tools}]
    generation = {}
    if isinstance(max_tokens, (int, float)):
        generation["maxOutputTokens"] = int(max_tokens)
    if stops:
        generation["stopSequences"] = stops
    if json_mode:
        generation["responseMimeType"] = "application/json"
    if generation:
        out["generationConfig"] = generation
    return out


# --------------------------------------------------------------------------
# Answer translation (provider wire -> client wire)
# --------------------------------------------------------------------------

def oa_to_anth_stop(finish):
    if finish == "length":
        return "max_tokens"
    if finish in ("tool_calls", "function_call"):
        return "tool_use"
    if finish == "content_filter":
        return "refusal"
    return "end_turn"


def anth_to_oa_stop(reason):
    if reason == "max_tokens":
        return "length"
    if reason == "tool_use":
        return "tool_calls"
    if reason == "refusal":
        return "content_filter"
    return "stop"


def gemini_to_oa_stop(finish_reason):
    if finish_reason == "MAX_TOKENS":
        return "length"
    if finish_reason == "SAFETY":
        return "content_filter"
    return "stop"


def gemini_to_anth_stop(finish_reason):
    if finish_reason == "MAX_TOKENS":
        return "max_tokens"
    if finish_reason == "SAFETY":
        return "refusal"
    return "end_turn"


def openai_to_anthropic(answer, model):
    """A Chat Completions answer as an Anthropic Messages object.

    Mirrors starter/lib/wire.ts openAIToAnthropic; the model named back is the
    serving model id.
    """
    choices = answer.get("choices") if isinstance(answer.get("choices"), list) else []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    content = []
    text = message.get("content")
    if not isinstance(text, str):
        text = text_of(message.get("content"))
    if text != "":
        content.append({"type": "text", "text": text})
    calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
            continue
        fn = call["function"]
        content.append({
            "type": "tool_use",
            "id": call.get("id") if isinstance(call.get("id"), str) else "toolu_unknown",
            "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
            "input": parse_json_obj(fn.get("arguments")),
        })
    if not content:
        content.append({"type": "text", "text": ""})
    usage = answer.get("usage") if isinstance(answer.get("usage"), dict) else {}
    upstream_id = answer.get("id") if isinstance(answer.get("id"), str) else "aggregate"
    return {
        "id": "msg_" + upstream_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": oa_to_anth_stop(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {
            "input_tokens": num_or_zero(usage.get("prompt_tokens")),
            "output_tokens": num_or_zero(usage.get("completion_tokens")),
        },
    }


def anthropic_to_openai_answer(answer, model):
    """An Anthropic Messages answer as a Chat Completions object."""
    content_blocks = answer.get("content") if isinstance(answer.get("content"), list) else []
    text_parts = []
    tool_calls = []
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": block.get("id") if isinstance(block.get("id"), str) else "toolu_unknown",
                "type": "function",
                "function": {
                    "name": block.get("name") if isinstance(block.get("name"), str) else "",
                    "arguments": json.dumps(block.get("input") if isinstance(block.get("input"), dict) else {}),
                },
            })
    message = {"role": "assistant", "content": "\n".join(text_parts)}
    if tool_calls:
        message["tool_calls"] = tool_calls
    usage = answer.get("usage") if isinstance(answer.get("usage"), dict) else {}
    prompt_tokens = num_or_zero(usage.get("input_tokens"))
    completion_tokens = num_or_zero(usage.get("output_tokens"))
    return {
        "id": "chatcmpl-sim",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": anth_to_oa_stop(answer.get("stop_reason")),
            "logprobs": None,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _gemini_text(answer):
    candidates = answer.get("candidates") if isinstance(answer.get("candidates"), list) else []
    if not candidates or not isinstance(candidates[0], dict):
        return "", None
    candidate = candidates[0]
    content = candidate.get("content") if isinstance(candidate.get("content"), dict) else {}
    parts = content.get("parts") if isinstance(content.get("parts"), list) else []
    text_parts = []
    tool_calls = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if isinstance(part.get("text"), str):
            text_parts.append(part["text"])
        elif isinstance(part.get("functionCall"), dict):
            call = part["functionCall"]
            tool_calls.append({
                "id": "call_%s" % (call.get("name", "")),
                "type": "function",
                "function": {
                    "name": call.get("name", ""),
                    "arguments": json.dumps(call.get("args") if isinstance(call.get("args"), dict) else {}),
                },
            })
    return "".join(text_parts), (tool_calls if tool_calls else None)


def _gemini_usage(answer):
    usage = answer.get("usageMetadata") if isinstance(answer.get("usageMetadata"), dict) else {}
    prompt = num_or_zero(usage.get("promptTokenCount"))
    completion = num_or_zero(usage.get("candidatesTokenCount"))
    total = num_or_zero(usage.get("totalTokenCount"))
    if total == 0:
        total = prompt + completion
    return prompt, completion, total


def gemini_to_openai(answer, model):
    """A Gemini generateContent answer as a Chat Completions object."""
    text, tool_calls = _gemini_text(answer)
    candidates = answer.get("candidates") if isinstance(answer.get("candidates"), list) else []
    finish = candidates[0].get("finishReason") if candidates and isinstance(candidates[0], dict) else None
    finish_reason = gemini_to_oa_stop(finish)
    if tool_calls:
        finish_reason = "tool_calls"
    message = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls
    prompt, completion, total = _gemini_usage(answer)
    return {
        "id": "chatcmpl-sim",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
            "logprobs": None,
        }],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
        },
    }


def gemini_to_anthropic(answer, model):
    """A Gemini generateContent answer as an Anthropic Messages object.

    Text parts become one text block; every functionCall part becomes a
    tool_use block, with the call id carried over from _gemini_text and
    the arguments parsed back to an object. When the answer carries tool
    calls the stop reason is tool_use, not end_turn, so the client knows
    to run the tools and come back. Dropping the calls would turn a
    tool-using answer into plain text and lose the work the model did.
    """
    text, tool_calls = _gemini_text(answer)
    candidates = answer.get("candidates") if isinstance(answer.get("candidates"), list) else []
    finish = candidates[0].get("finishReason") if candidates and isinstance(candidates[0], dict) else None
    prompt, completion, _total = _gemini_usage(answer)
    blocks = []
    if text != "":
        blocks.append({"type": "text", "text": text})
    for call in tool_calls or []:
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        blocks.append({
            "type": "tool_use",
            "id": call.get("id") if isinstance(call.get("id"), str) and call.get("id") != "" else "toolu_gemini",
            "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
            "input": parse_json_obj(fn.get("arguments")),
        })
    stop = gemini_to_anth_stop(finish)
    if blocks and any(b.get("type") == "tool_use" for b in blocks):
        stop = "tool_use"
    return {
        "id": "msg_sim",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": prompt, "output_tokens": completion},
    }


def translate_json_answer(prov_wire, client_wire, answer, model):
    """A provider JSON answer in the client's wire. Same wire: verbatim."""
    if prov_wire == client_wire:
        return answer
    if prov_wire == "openai":
        return openai_to_anthropic(answer, model)
    if prov_wire == "anthropic":
        return anthropic_to_openai_answer(answer, model)
    if client_wire == "openai":
        return gemini_to_openai(answer, model)
    return gemini_to_anthropic(answer, model)


# --------------------------------------------------------------------------
# SSE translation
# --------------------------------------------------------------------------

def sse_event(event, data):
    text = json.dumps(data, separators=(",", ":"))
    if event is None:
        return "data: %s" % text
    return "event: %s\ndata: %s" % (event, text)


def sse_data(data):
    return "data: %s" % data


def parse_sse_frames(data_bytes):
    """An SSE body as [(event_name_or_None, data_string)] frames."""
    text = data_bytes.decode("utf-8", "replace").replace("\r\n", "\n")
    frames = []
    for chunk in text.split("\n\n"):
        if chunk.strip() == "":
            continue
        event = None
        data_lines = []
        for line in chunk.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
        frames.append((event, "\n".join(data_lines)))
    return frames


def _openai_chunk(model, content, finish_reason):
    chunk = {
        "id": "chatcmpl-sim",
        "object": "chat.completion.chunk",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }
    if content is None:
        return chunk
    delta = {}
    if content == "" and finish_reason is None:
        delta = {"role": "assistant", "content": ""}
    else:
        delta = {"content": content}
    chunk["choices"][0]["delta"] = delta
    return chunk


def _openai_usage_chunk(model, prompt_tokens, completion_tokens):
    return {
        "id": "chatcmpl-sim",
        "object": "chat.completion.chunk",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def anthropic_events_from_openai(frames, model):
    """An OpenAI SSE stream as the Anthropic event sequence.

    Mirrors starter/lib/wire.ts anthropicEventsFromOpenAI: message_start on the
    first chunk, content_block_start/delta/stop per text block or tool call,
    then message_delta with the stop reason and usage, then message_stop. The
    stop reason is held until [DONE], because the usage chunk arrives after the
    finish chunk.
    """
    out = []
    state = {
        "started": False, "finished": False, "pending_stop": None,
        "text_index": -1, "next_index": 0, "tool_indexes": {},
        "input_tokens": 0, "output_tokens": 0,
    }

    def start_events():
        if state["started"]:
            return []
        state["started"] = True
        return [sse_event("message_start", {
            "type": "message_start",
            "message": {
                "id": "msg_stream", "type": "message", "role": "assistant",
                "model": model, "content": [], "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": state["input_tokens"], "output_tokens": 0},
            },
        })]

    def close_events():
        events = []
        if state["text_index"] != -1:
            events.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": state["text_index"]}))
            state["text_index"] = -1
        for index in sorted(state["tool_indexes"].values()):
            events.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": index}))
        state["tool_indexes"] = {}
        return events

    def finish_events(stop_reason):
        if state["finished"]:
            return []
        state["finished"] = True
        return close_events() + [
            sse_event("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": state["output_tokens"]},
            }),
            sse_event("message_stop", {"type": "message_stop"}),
        ]

    for _event, data in frames:
        if data == "[DONE]":
            out.extend(start_events())
            out.extend(finish_events(state["pending_stop"] if state["pending_stop"] is not None else "end_turn"))
            continue
        try:
            chunk = json.loads(data)
        except Exception:
            continue
        if not isinstance(chunk, dict):
            continue
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            state["input_tokens"] = num_or_zero(usage.get("prompt_tokens"))
            state["output_tokens"] = num_or_zero(usage.get("completion_tokens"))
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
        out.extend(start_events())
        content = delta.get("content")
        if isinstance(content, str) and content != "":
            if state["text_index"] == -1:
                state["text_index"] = state["next_index"]
                state["next_index"] += 1
                out.append(sse_event("content_block_start", {
                    "type": "content_block_start", "index": state["text_index"],
                    "content_block": {"type": "text", "text": ""},
                }))
            out.append(sse_event("content_block_delta", {
                "type": "content_block_delta", "index": state["text_index"],
                "delta": {"type": "text_delta", "text": content},
            }))
        calls = delta.get("tool_calls")
        if isinstance(calls, list):
            for call in calls:
                if not isinstance(call, dict):
                    continue
                upstream_index = call.get("index") if isinstance(call.get("index"), int) else len(state["tool_indexes"])
                if upstream_index not in state["tool_indexes"]:
                    if state["text_index"] != -1:
                        out.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": state["text_index"]}))
                        state["text_index"] = -1
                    index = state["next_index"]
                    state["next_index"] += 1
                    state["tool_indexes"][upstream_index] = index
                    fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                    call_id = call.get("id")
                    out.append(sse_event("content_block_start", {
                        "type": "content_block_start", "index": index,
                        "content_block": {
                            "type": "tool_use",
                            "id": call_id if isinstance(call_id, str) and call_id != "" else "toolu_%d" % index,
                            "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
                            "input": {},
                        },
                    }))
                fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                arguments = fn.get("arguments")
                if isinstance(arguments, str) and arguments != "":
                    out.append(sse_event("content_block_delta", {
                        "type": "content_block_delta", "index": state["tool_indexes"][upstream_index],
                        "delta": {"type": "input_json_delta", "partial_json": arguments},
                    }))
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None:
            state["pending_stop"] = oa_to_anth_stop(finish_reason)
    if not state["finished"]:
        out.extend(start_events())
        out.extend(finish_events(state["pending_stop"] if state["pending_stop"] is not None else "end_turn"))
    return out


def _openai_tool_chunk(model, index, call_id, name, arguments):
    """One OpenAI chat-completion chunk carrying a tool-call delta.

    The first chunk for a call carries the id, the name and an empty
    arguments string; later chunks carry only argument fragments. The
    caller fills in the finish reason on the closing chunk.
    """
    call = {"index": index, "type": "function", "function": {"arguments": arguments}}
    if call_id is not None:
        call["id"] = call_id
    if name is not None:
        call["function"]["name"] = name
    return {
        "id": "chatcmpl-sim",
        "object": "chat.completion.chunk",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{"index": 0, "delta": {"tool_calls": [call]}, "finish_reason": None}],
    }


def anthropic_sse_to_openai(frames, model):
    """An Anthropic SSE stream as OpenAI chat-completion chunks + usage + [DONE].

    Text deltas become content chunks. A tool_use block becomes an OpenAI
    tool call: content_block_start carries its id and name, and every
    input_json_delta appends its argument fragment, so the client's tool
    call is complete by the time the stream ends. When the message stops
    with tool_use, the closing chunk's finish reason is tool_calls, not
    stop, so the client knows to run the tools. Dropping the tool events
    would silently turn a tool-using answer into plain text.
    """
    out = []
    started = False
    finished = False
    input_tokens = 0
    output_tokens = 0
    stop_reason = "stop"
    tools = {}
    next_tool = 0

    def open_chunk():
        nonlocal started
        if not started:
            out.append(sse_data(json.dumps(_openai_chunk(model, "", None), separators=(",", ":"))))
            started = True

    def close_stream():
        nonlocal finished
        if finished:
            return
        open_chunk()
        finish = _openai_chunk(model, None, stop_reason)
        finish["choices"][0]["delta"] = {}
        out.append(sse_data(json.dumps(finish, separators=(",", ":"))))
        out.append(sse_data(json.dumps(_openai_usage_chunk(model, input_tokens, output_tokens), separators=(",", ":"))))
        out.append("data: [DONE]")
        finished = True

    for _event, data in frames:
        try:
            obj = json.loads(data)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        kind = obj.get("type")
        if kind == "message_start":
            message = obj.get("message") if isinstance(obj.get("message"), dict) else {}
            usage = message.get("usage") if isinstance(message.get("usage"), dict) else {}
            input_tokens = num_or_zero(usage.get("input_tokens"))
            open_chunk()
        elif kind == "content_block_start":
            block = obj.get("content_block") if isinstance(obj.get("content_block"), dict) else {}
            if block.get("type") == "tool_use":
                index = obj.get("index") if isinstance(obj.get("index"), int) else -1
                call_id = block.get("id") if isinstance(block.get("id"), str) and block.get("id") != "" else "call_%d" % next_tool
                name = block.get("name") if isinstance(block.get("name"), str) else ""
                tools[index] = {"oa_index": next_tool, "id": call_id, "name": name}
                next_tool += 1
                open_chunk()
                out.append(sse_data(json.dumps(_openai_tool_chunk(model, tools[index]["oa_index"], call_id, name, ""), separators=(",", ":"))))
        elif kind == "content_block_delta":
            delta = obj.get("delta") if isinstance(obj.get("delta"), dict) else {}
            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str) and delta["text"] != "":
                open_chunk()
                out.append(sse_data(json.dumps(_openai_chunk(model, delta["text"], None), separators=(",", ":"))))
            elif delta.get("type") == "input_json_delta" and isinstance(delta.get("partial_json"), str) and delta["partial_json"] != "":
                index = obj.get("index") if isinstance(obj.get("index"), int) else -1
                if index in tools:
                    open_chunk()
                    out.append(sse_data(json.dumps(_openai_tool_chunk(model, tools[index]["oa_index"], None, None, delta["partial_json"]), separators=(",", ":"))))
        elif kind == "message_delta":
            delta = obj.get("delta") if isinstance(obj.get("delta"), dict) else {}
            reason = delta.get("stop_reason")
            if isinstance(reason, str):
                stop_reason = anth_to_oa_stop(reason)
            usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else {}
            output_tokens = num_or_zero(usage.get("output_tokens"))
            close_stream()
    close_stream()
    return out


def gemini_sse_to_openai(frames, model):
    """A Gemini :streamGenerateContent stream as OpenAI chunks + usage + [DONE].

    Text parts become content chunks. A functionCall part becomes an
    OpenAI tool call, with the call id derived from the function name and
    the arguments serialised as one JSON string, so the client's tool call
    is complete. When the stream carried any function call, the closing
    chunk's finish reason is tool_calls, not stop, so the client runs the
    tools instead of treating the answer as plain text.
    """
    out = []
    started = False
    finished = False
    input_tokens = 0
    output_tokens = 0
    stop_reason = "stop"
    saw_tool = False
    next_tool = 0

    def open_chunk():
        nonlocal started
        if not started:
            out.append(sse_data(json.dumps(_openai_chunk(model, "", None), separators=(",", ":"))))
            started = True

    def close_stream():
        nonlocal finished
        if finished:
            return
        open_chunk()
        reason = "tool_calls" if saw_tool else stop_reason
        chunk = _openai_chunk(model, None, reason)
        chunk["choices"][0]["delta"] = {}
        out.append(sse_data(json.dumps(chunk, separators=(",", ":"))))
        out.append(sse_data(json.dumps(_openai_usage_chunk(model, input_tokens, output_tokens), separators=(",", ":"))))
        out.append("data: [DONE]")
        finished = True

    for _event, data in frames:
        try:
            obj = json.loads(data)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        candidates = obj.get("candidates")
        if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
            continue
        candidate = candidates[0]
        content = candidate.get("content") if isinstance(candidate.get("content"), dict) else {}
        parts = content.get("parts") if isinstance(content.get("parts"), list) else []
        text = "".join(part.get("text", "") for part in parts
                       if isinstance(part, dict) and isinstance(part.get("text"), str))
        if text != "":
            open_chunk()
            out.append(sse_data(json.dumps(_openai_chunk(model, text, None), separators=(",", ":"))))
        for part in parts:
            if not isinstance(part, dict):
                continue
            call = part.get("functionCall")
            if not isinstance(call, dict):
                continue
            name = call.get("name") if isinstance(call.get("name"), str) else ""
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            call_id = "call_%s" % name
            open_chunk()
            out.append(sse_data(json.dumps(
                _openai_tool_chunk(model, next_tool, call_id, name, json.dumps(args, separators=(",", ":"))),
                separators=(",", ":"))))
            next_tool += 1
            saw_tool = True
        usage = obj.get("usageMetadata")
        if isinstance(usage, dict):
            input_tokens = num_or_zero(usage.get("promptTokenCount"))
            output_tokens = num_or_zero(usage.get("candidatesTokenCount"))
        finish = candidate.get("finishReason")
        if isinstance(finish, str):
            stop_reason = gemini_to_oa_stop(finish)
            close_stream()
    close_stream()
    return out


def gemini_sse_to_anthropic(frames, model):
    """A Gemini :streamGenerateContent stream as Anthropic events.

    Text parts become one text block (index 0). A functionCall part
    becomes a tool_use block on its own index: content_block_start
    carries the id and name, an input_json_delta carries the arguments
    as one JSON string, and content_block_stop closes it. When the stream
    carried any function call, the message stops with tool_use, not
    end_turn, so the client runs the tools. Dropping the calls would turn
    a tool-using answer into plain text.
    """
    out = []
    started = False
    block_open = False
    finished = False
    output_tokens = 0
    stop_reason = "end_turn"
    saw_tool = False
    next_index = 1

    def start_message():
        return sse_event("message_start", {
            "type": "message_start",
            "message": {
                "id": "msg_stream", "type": "message", "role": "assistant",
                "model": model, "content": [], "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })

    def close_message():
        nonlocal finished, block_open
        if finished:
            return
        if not started:
            out.append(start_message())
        if block_open:
            out.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}))
            block_open = False
        reason = "tool_use" if saw_tool else stop_reason
        out.append(sse_event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        }))
        out.append(sse_event("message_stop", {"type": "message_stop"}))
        finished = True

    for _event, data in frames:
        try:
            obj = json.loads(data)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        candidates = obj.get("candidates")
        if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
            continue
        candidate = candidates[0]
        content = candidate.get("content") if isinstance(candidate.get("content"), dict) else {}
        parts = content.get("parts") if isinstance(content.get("parts"), list) else []
        text = "".join(part.get("text", "") for part in parts
                       if isinstance(part, dict) and isinstance(part.get("text"), str))
        usage = obj.get("usageMetadata")
        if isinstance(usage, dict):
            output_tokens = num_or_zero(usage.get("candidatesTokenCount"))
        if text != "":
            if not started:
                out.append(start_message())
                started = True
            if not block_open:
                out.append(sse_event("content_block_start", {
                    "type": "content_block_start", "index": 0,
                    "content_block": {"type": "text", "text": ""},
                }))
                block_open = True
            out.append(sse_event("content_block_delta", {
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": text},
            }))
        for part in parts:
            if not isinstance(part, dict):
                continue
            call = part.get("functionCall")
            if not isinstance(call, dict):
                continue
            name = call.get("name") if isinstance(call.get("name"), str) else ""
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            call_id = "call_%s" % name
            index = next_index
            next_index += 1
            if not started:
                out.append(start_message())
                started = True
            if block_open:
                out.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}))
                block_open = False
            out.append(sse_event("content_block_start", {
                "type": "content_block_start", "index": index,
                "content_block": {"type": "tool_use", "id": call_id, "name": name, "input": {}},
            }))
            out.append(sse_event("content_block_delta", {
                "type": "content_block_delta", "index": index,
                "delta": {"type": "input_json_delta", "partial_json": json.dumps(args, separators=(",", ":"))},
            }))
            out.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": index}))
            saw_tool = True
        finish = candidate.get("finishReason")
        if isinstance(finish, str):
            stop_reason = gemini_to_anth_stop(finish)
            close_message()
    close_message()
    return out


def translate_stream(prov_wire, client_wire, data_bytes, model):
    """A provider SSE body in the client's wire. Same wire: verbatim bytes."""
    if prov_wire == client_wire:
        return data_bytes
    frames = parse_sse_frames(data_bytes)
    if prov_wire == "openai":
        events = anthropic_events_from_openai(frames, model)
    elif prov_wire == "anthropic":
        events = anthropic_sse_to_openai(frames, model)
    elif client_wire == "openai":
        events = gemini_sse_to_openai(frames, model)
    else:
        events = gemini_sse_to_anthropic(frames, model)
    return ("\n\n".join(events) + "\n\n").encode("utf-8")


# --------------------------------------------------------------------------
# Routing and failover
# --------------------------------------------------------------------------

ENV_URL = ""
PROBES = []
CATALOG = {"providers": []}
RATINGS = None
WORKLOAD = None
PROVIDER_BY_ID = {}
calls_used = {}
refusals = {}
manifest_index = 0
REQUEST_LOCK = threading.Lock()


def requirements_from_body(body):
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    vision = False
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in ("image", "image_url"):
                    vision = True
    fmt = body.get("response_format") if isinstance(body.get("response_format"), dict) else {}
    return {
        "tools": isinstance(body.get("tools"), list) and len(body["tools"]) > 0,
        "vision": vision,
        "json": fmt.get("type") in ("json_object", "json_schema"),
    }


def requirements_for(body):
    """The requirements of the request now arriving: the manifest row by
    arrival order, else the body's own. The manifest is in arrival order."""
    global manifest_index
    idx = manifest_index
    manifest_index += 1
    if WORKLOAD and idx < len(WORKLOAD):
        return WORKLOAD[idx]["requires"]
    return requirements_from_body(body)


OVER_PLAN_PENALTY = 60
REFUSAL_PENALTY = 40
MAX_REFUSAL_STEPS = 2


def build_candidates(requires, json_mode):
    """Every (provider_id, model_id, rating) that could serve this request.

    A candidate is a free model on a probed provider whose capabilities meet
    the request's requirements. A JSON-mode request never goes to an
    anthropic-wire provider, because that wire cannot carry response_format.
    Nothing is filtered out here: ranking decides the order, and the serving
    loop tries candidates until one answers, so a provider is never silently
    dropped from a request it could have served.
    """
    found = []
    for probe in PROBES:
        if not probe["ok"]:
            continue
        provider = probe["provider"]
        if json_mode and provider["wire"] == "anthropic":
            continue
        for model in probe["models"]:
            if not model.get("is_free"):
                continue
            if not meets_requirements(requires, model["caps"]):
                continue
            found.append((provider["id"], model["id"], model["rating"]))
    return found


def _ranking_penalty(provider_id):
    """Rating points subtracted when ordering one provider's candidates.

    Two soft costs, both capped, both recoverable. A provider that has used
    its safe call floor (CALLS_MIN, the minimum the draw guarantees, not a
    hard cap) costs OVER_PLAN_PENALTY, because its remaining budget is
    uncertain and a fresh provider should go first when the ratings are
    close. Each recent refusal costs REFUSAL_PENALTY, up to
    MAX_REFUSAL_STEPS steps, because a provider that just refused is more
    likely to refuse again. Neither cost excludes the provider: a clearly
    better model still outranks a worse one, and a provider whose every
    rival has already failed is still tried.
    """
    cost = 0
    if calls_used.get(provider_id, 0) >= CALLS_MIN.get(provider_id, 5):
        cost += OVER_PLAN_PENALTY
    cost += REFUSAL_PENALTY * min(refusals.get(provider_id, 0), MAX_REFUSAL_STEPS)
    return cost


def rank_candidates(cands):
    """All candidates, best first, by effective rating.

    The effective rating is the quality-table rating minus the ranking
    penalty, so a provider past its safe floor or with recent refusals is
    pushed down, never removed. Ties break toward the provider that has
    spent fewer calls and refused less, then by provider and model id for
    determinism. The list is complete: the serving loop walks it until a
    provider answers, which is what makes "exceed the safe floor only when
    no other provider can serve" true in practice.
    """
    return sorted(cands, key=lambda cand: (
        -(cand[2] - _ranking_penalty(cand[0])),
        calls_used.get(cand[0], 0),
        refusals.get(cand[0], 0),
        cand[0], cand[1]))


def build_upstream_request(prov, model_id, client_wire, body, is_stream):
    """(url, headers, body_dict) for one provider call in the provider's wire."""
    wire = prov["wire"]
    base = join_base(ENV_URL, prov["base"])
    headers = {"content-type": "application/json"}
    headers.update(auth_headers(prov["auth"], prov["key"]))
    if wire == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    if is_stream:
        headers["accept"] = "text/event-stream"
    if wire == "openai":
        if client_wire == "openai":
            up = dict(body)
            up["model"] = model_id
            if is_stream:
                up["stream_options"] = {"include_usage": True}
        else:
            up = anthropic_to_openai(body, model_id)
        url = base + "/v1/chat/completions"
    elif wire == "anthropic":
        if client_wire == "anthropic":
            up = dict(body)
            up["model"] = model_id
        else:
            up = openai_to_anthropic_req(body, model_id)
        url = base + "/v1/messages"
    else:
        up = to_gemini(body, model_id, client_wire)
        if is_stream:
            url = base + "/v1beta/models/" + model_id + ":streamGenerateContent"
        else:
            url = base + "/v1beta/models/" + model_id + ":generateContent"
    return url, headers, up


def _is_concurrent_refusal(data):
    """True when a refusal body names a concurrency limit, not a budget.

    The env refuses a second in-flight call to a one-at-a-time provider
    with 429 and the code "concurrent_requests", in the wire's own error
    shape (OpenAI puts it in error.code, Anthropic in error.type, Gemini
    in error.status). That refusal spends no call and is not a strike
    against the provider: the provider is healthy, it was just busy.
    """
    try:
        text = data.decode("utf-8", "replace")
    except Exception:
        return False
    return "concurrent_requests" in text


def attempt(prov, model_id, client_wire, body, is_stream):
    """One provider call. Returns (status, content_type, body_bytes) on a
    200, else None, and the caller fails over to the next candidate.

    Budget accounting mirrors the env. A 200, a 402 or 429 failure, or a
    5xx spends one call on that provider. A 400, a 401, or a
    concurrent_requests 429 spends none: the call never reached the
    provider's budget. A network error spends none either. A success
    clears the provider's refusal count, so a provider that had transient
    failures is not penalised forever; a refusal adds one strike.
    """
    pid = prov["id"]
    url, headers, up_body = build_upstream_request(prov, model_id, client_wire, body, is_stream)
    try:
        status, _hdrs, data = http_request(
            url, "POST", headers,
            json.dumps(up_body, separators=(",", ":")).encode("utf-8"),
            timeout=UPSTREAM_TIMEOUT)
    except Exception as exc:
        log("upstream %s failed: %s" % (pid, exc))
        refusals[pid] = refusals.get(pid, 0) + 1
        return None
    if status == 200:
        calls_used[pid] = calls_used.get(pid, 0) + 1
        refusals[pid] = 0
        if is_stream:
            out = translate_stream(prov["wire"], client_wire, data, model_id)
            return (200, "text/event-stream; charset=utf-8", out)
        try:
            answer = json.loads(data.decode("utf-8"))
        except Exception:
            refusals[pid] = refusals.get(pid, 0) + 1
            return None
        if not isinstance(answer, dict):
            refusals[pid] = refusals.get(pid, 0) + 1
            return None
        translated = translate_json_answer(prov["wire"], client_wire, answer, model_id)
        return (200, "application/json",
                json.dumps(translated, separators=(",", ":")).encode("utf-8"))
    if status == 429 and _is_concurrent_refusal(data):
        return None
    if status == 402 or status == 429 or status >= 500:
        calls_used[pid] = calls_used.get(pid, 0) + 1
        refusals[pid] = refusals.get(pid, 0) + 1
        return None
    refusals[pid] = refusals.get(pid, 0) + 1
    return None


def serve(client_wire, body):
    """Serve one client request. Returns (status, content_type, body_bytes),
    or None only when no provider on the whole ranked list could serve it.

    The ranked list is complete: every free model that meets the request
    is on it, ordered by effective rating. Pass 1 tries one model per
    provider, so a failure moves to a different provider first. Pass 2
    tries a second model on a provider already tried, for failures that
    were model-specific rather than provider-wide. Because the list is
    complete, a provider that is past its safe floor or has recent
    refusals is still reached when every fresher provider has failed:
    the safe floor is exceeded only when no other provider can serve,
    which is the rule as written, not just as intended.
    """
    requires = requirements_for(body)
    is_stream = body.get("stream") is True
    if client_wire == "openai":
        fmt = body.get("response_format") if isinstance(body.get("response_format"), dict) else {}
        json_mode = fmt.get("type") in ("json_object", "json_schema")
    else:
        json_mode = False
    cands = build_candidates(requires, json_mode)
    if not cands:
        return None
    ranked = rank_candidates(cands)
    tried_providers = set()
    tried_models = set()
    attempts = 0
    for pass_no in (1, 2):
        for pid, mid, _rating in ranked:
            if attempts >= MAX_ATTEMPTS:
                break
            if pass_no == 1 and pid in tried_providers:
                continue
            if pass_no == 2 and pid not in tried_providers:
                continue
            if (pid, mid) in tried_models:
                continue
            tried_providers.add(pid)
            tried_models.add((pid, mid))
            attempts += 1
            result = attempt(PROVIDER_BY_ID[pid], mid, client_wire, body, is_stream)
            if result is not None:
                return result
        if attempts >= MAX_ATTEMPTS:
            break
    return None


# --------------------------------------------------------------------------
# HTTP server
# --------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _read_chunked_body(self):
        """Read a Transfer-Encoding: chunked request body, up to MAX_BODY.

        Each chunk is a hex size line, that many bytes, then CRLF; a
        zero-size chunk ends the body. Returns the raw bytes, or None when
        the framing is broken or the body is too large.
        """
        parts = []
        total = 0
        while True:
            size_line = self.rfile.readline(MAX_BODY + 64)
            if not size_line:
                return None
            size_text = size_line.decode("ascii", "replace").strip().split(";", 1)[0].strip()
            try:
                size = int(size_text, 16)
            except ValueError:
                return None
            if size < 0:
                return None
            if size == 0:
                self.rfile.readline(64)
                break
            total += size
            if total > MAX_BODY:
                return None
            chunk = self.rfile.read(size)
            if chunk is None or len(chunk) != size:
                return None
            parts.append(chunk)
            crlf = self.rfile.read(2)
            if crlf != b"\r\n":
                return None
        return b"".join(parts)

    def _read_body(self):
        length = self.headers.get("Content-Length")
        if length is not None:
            try:
                size = int(length)
            except ValueError:
                return None
            if size < 0 or size > MAX_BODY:
                return None
            data = self.rfile.read(size)
            if not data:
                return None
        elif (self.headers.get("Transfer-Encoding") or "").lower() == "chunked":
            data = self._read_chunked_body()
            if data is None:
                return None
        else:
            return None
        try:
            value = json.loads(data.decode("utf-8"))
        except Exception:
            return None
        return value if isinstance(value, dict) else None

    def _send(self, status, content_type, body_bytes):
        try:
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body_bytes)))
            self.end_headers()
            self.wfile.write(body_bytes)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_json(self, status, obj):
        self._send(status, "application/json",
                   json.dumps(obj, separators=(",", ":")).encode("utf-8"))

    def _send_error(self, wire, status, err_type, message):
        if wire == "anthropic":
            self._send_json(status, {"type": "error", "error": {"type": err_type, "message": message}})
        else:
            self._send_json(status, {"error": {"message": message, "type": err_type, "code": None}})

    def _models(self):
        wire = "anthropic" if isinstance(self.headers.get("x-api-key"), str) else "openai"
        models = served_models(PROBES)
        if wire == "anthropic":
            body = {
                "data": [{"type": "model", "id": m["id"], "display_name": m["display_name"],
                          "created_at": FIXED_CREATED_AT} for m in models],
                "has_more": False,
            }
        else:
            body = {
                "object": "list",
                "data": [{"id": m["id"], "object": "model", "created": FIXED_CREATED,
                          "owned_by": m["provider"]} for m in models],
            }
        self._send_json(200, body)

    def _handle_chat(self):
        body = self._read_body()
        if body is None:
            self._send_error("openai", 400, "invalid_request_error",
                             "the request body is not one JSON object")
            return
        with REQUEST_LOCK:
            result = serve("openai", body)
        if result is None:
            self._send_error("openai", 503, "no_free_model",
                             "no free model on the directory meets this request")
            return
        status, content_type, data = result
        self._send(status, content_type, data)

    def _handle_messages(self):
        body = self._read_body()
        if body is None:
            self._send_error("anthropic", 400, "invalid_request_error",
                             "the request body is not one JSON object")
            return
        with REQUEST_LOCK:
            result = serve("anthropic", body)
        if result is None:
            self._send_error("anthropic", 503, "overloaded_error",
                             "no free model on the directory meets this request")
            return
        status, content_type, data = result
        self._send(status, content_type, data)

    def do_GET(self):
        try:
            path = urlsplit(self.path).path
            if path == "/v1/models":
                self._models()
            else:
                self._send_error("openai", 404, "not_found", "no route GET %s" % path)
        except Exception as exc:
            log("GET handler failed: %s" % exc)

    def do_POST(self):
        try:
            path = urlsplit(self.path).path
            if path == "/v1/chat/completions":
                self._handle_chat()
            elif path == "/v1/messages":
                self._handle_messages()
            else:
                wire = "anthropic" if path == "/v1/messages" else "openai"
                self._send_error(wire, 404, "not_found", "no route POST %s" % path)
        except Exception as exc:
            log("POST handler failed: %s" % exc)
            try:
                wire = "anthropic" if self.path.startswith("/v1/messages") else "openai"
                if not getattr(self, "_headers_sent", False):
                    self._send_error(wire, 500, "internal_error", "the server failed to answer")
            except Exception:
                pass


# --------------------------------------------------------------------------
# Register, poll, startup
# --------------------------------------------------------------------------

def register(port):
    url = ENV_URL + "/register"
    body = json.dumps(
        {"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": CATALOG},
        separators=(",", ":")).encode("utf-8")
    status, _hdrs, data = http_request(url, "POST", {"content-type": "application/json"},
                                       body, timeout=REGISTER_TIMEOUT)
    if status != 200:
        raise RuntimeError("POST %s answered HTTP %s: %s" % (
            url, status, data[:500].decode("utf-8", "replace")))
    reply = json.loads(data.decode("utf-8"))
    if isinstance(reply, dict) and isinstance(reply.get("requests"), int):
        log("registered: the workload holds %d requests" % reply["requests"])


def poll_loop(server, stop_event):
    while not stop_event.is_set():
        try:
            status, _hdrs, data = http_request(ENV_URL + "/work", "GET", timeout=WORK_TIMEOUT)
            if status == 200:
                work = json.loads(data.decode("utf-8"))
                if isinstance(work, dict) and work.get("done") is True:
                    stop_event.set()
                    threading.Thread(target=server.shutdown, daemon=True).start()
                    return
        except Exception:
            pass
        time.sleep(POLL_INTERVAL)


def main():
    global ENV_URL, PROBES, CATALOG, RATINGS, WORKLOAD, PROVIDER_BY_ID
    ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE", "")
    providers, ratings, workload, error = read_brief(prompt_file)
    if error is not None:
        log(error)
    RATINGS = ratings
    WORKLOAD = workload
    PROBES = [probe_provider(ENV_URL, provider) for provider in providers] if ENV_URL else []
    for probe in PROBES:
        if not probe["ok"] and probe["error"]:
            log(probe["error"])
    enrich_probes(PROBES, RATINGS)
    CATALOG = build_catalog(PROBES, RATINGS)
    PROVIDER_BY_ID = {probe["provider"]["id"]: probe["provider"]
                      for probe in PROBES if probe["ok"]}
    if "--list" in sys.argv:
        # The same document the register body carries, byte for byte.
        sys.stdout.write(json.dumps(CATALOG, separators=(",", ":")) + "\n")
        sys.stdout.flush()
        sys.exit(0)
    if ENV_URL == "":
        raise RuntimeError("MAKEFUN_ENV_URL is not set")
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    server.daemon_threads = True
    port = server.server_address[1]
    if port == 8080:
        raise RuntimeError("the OS picked 8080, which the env owns on one layout: restart the trial")
    log("listening on 0.0.0.0:%d with %d answered providers and %d free models" % (
        port, len([p for p in PROBES if p["ok"]]), len(served_models(PROBES))))
    register(port)

    def _exit(signum, _frame):
        sys.exit(0)

    signal.signal(signal.SIGTERM, _exit)
    signal.signal(signal.SIGINT, _exit)
    stop_event = threading.Event()
    poll_thread = threading.Thread(target=poll_loop, args=(server, stop_event), daemon=True)
    poll_thread.start()
    server.serve_forever()
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        log("fatal: %s" % exc)
        sys.exit(1)
