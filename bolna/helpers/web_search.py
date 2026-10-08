import asyncio
import json
import os
from dataclasses import dataclass
from urllib.parse import urlparse

import aiohttp
from openai import AsyncOpenAI

from bolna.constants import is_reasoning_model
from bolna.helpers.logger_config import configure_logger
from bolna.llms.http_client_pool import get_shared_http_client

logger = configure_logger(__name__)

SEARCH_UNAVAILABLE = json.dumps({"status": "error", "message": "Search is unavailable right now."})
NO_RESULTS = json.dumps({"status": "success", "message": "The search returned no results."})

FIRECRAWL_SEARCH_URL = "https://api.firecrawl.dev/v2/search"
EXA_SEARCH_URL = "https://api.exa.ai/search"
PARALLEL_SEARCH_URL = "https://api.parallel.ai/v1/search"

PROVIDER_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "firecrawl": "FIRECRAWL_API_KEY",
    "exa": "EXA_API_KEY",
    "parallel": "PARALLEL_API_KEY",
}


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str


def _domain(url: str) -> str:
    host = urlparse(url or "").netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def format_hits(query: str, hits: list[SearchHit], max_results: int, max_snippet_chars: int) -> str:
    hits = [h for h in hits if h.snippet or h.title][:max_results]
    if not hits:
        return NO_RESULTS
    lines = [f'Search results for "{query}":']
    for i, hit in enumerate(hits, 1):
        snippet = _clip(hit.snippet, max_snippet_chars)
        title = _clip(hit.title, 120)
        source = _domain(hit.url)
        body = f"{title} - {snippet}" if title and snippet else (title or snippet)
        lines.append(f"{i}. {body}" + (f" (source: {source})" if source else ""))
    lines.append("Answer briefly using these results. Mention a source only if the caller asks.")
    return "\n".join(lines)


async def _post_json(url: str, headers: dict, body: dict, timeout_s: float) -> dict:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_s)) as session:
        async with session.post(url, headers={"Content-Type": "application/json", **headers}, json=body) as resp:
            if resp.status >= 400:
                detail = (await resp.text())[:200]
                raise RuntimeError(f"HTTP {resp.status}: {detail}")
            return await resp.json(content_type=None)


async def _search_firecrawl(query, api_key, cfg, timeout_s) -> list[SearchHit]:
    # No scrapeOptions: full-page markdown is too large for a spoken follow-up turn.
    data = await _post_json(
        FIRECRAWL_SEARCH_URL,
        {"Authorization": f"Bearer {api_key}"},
        {"query": query, "limit": cfg.max_results},
        timeout_s,
    )
    payload = data.get("data") or {}
    results = payload.get("web") if isinstance(payload, dict) else payload
    return [
        SearchHit(
            title=r.get("title") or "",
            url=r.get("url") or "",
            snippet=r.get("description") or r.get("markdown") or "",
        )
        for r in (results or [])
        if isinstance(r, dict)
    ]


async def _search_exa(query, api_key, cfg, timeout_s) -> list[SearchHit]:
    data = await _post_json(
        EXA_SEARCH_URL,
        {"x-api-key": api_key},
        {
            "query": query,
            "type": "instant",
            "numResults": cfg.max_results,
            "contents": {"highlights": {"maxCharacters": cfg.max_snippet_chars}},
        },
        timeout_s,
    )
    return [
        SearchHit(
            title=r.get("title") or "",
            url=r.get("url") or "",
            snippet=" ".join(r.get("highlights") or []) or r.get("text") or "",
        )
        for r in (data.get("results") or [])
        if isinstance(r, dict)
    ]


async def _search_parallel(query, api_key, cfg, timeout_s) -> list[SearchHit]:
    # mode "fast" stays under a second; the default "advanced" takes about 3s.
    data = await _post_json(
        PARALLEL_SEARCH_URL,
        {"x-api-key": api_key},
        {
            "search_queries": [query],
            "mode": "fast",
            "max_results": cfg.max_results,
            "advanced_settings": {"excerpt_settings": {"max_chars_per_result": cfg.max_snippet_chars}},
        },
        timeout_s,
    )
    return [
        SearchHit(
            title=r.get("title") or "",
            url=r.get("url") or "",
            snippet=" ".join(r.get("excerpts") or []),
        )
        for r in (data.get("results") or [])
        if isinstance(r, dict)
    ]


def _openai_citations(response) -> list[str]:
    urls: list[str] = []
    for item in getattr(response, "output", None) or []:
        for content in getattr(item, "content", None) or []:
            for annotation in getattr(content, "annotations", None) or []:
                url = getattr(annotation, "url", None)
                if url and url not in urls:
                    urls.append(url)
    return urls


async def _search_openai(query, api_key, cfg, timeout_s) -> list[SearchHit]:
    client = AsyncOpenAI(api_key=api_key, timeout=timeout_s, http_client=get_shared_http_client(http2=False))
    request = {
        "model": cfg.model,
        "input": query,
        "tools": [{"type": "web_search", "search_context_size": "low"}],
        "tool_choice": {"type": "web_search"},
    }
    if is_reasoning_model(cfg.model):
        request["reasoning"] = {"effort": "low"}
    response = await client.responses.create(**request)
    answer = getattr(response, "output_text", None) or ""
    if not answer.strip():
        return []
    citations = _openai_citations(response)
    hits = [SearchHit(title="", url=citations[0] if citations else "", snippet=answer)]
    hits.extend(SearchHit(title=_domain(u), url=u, snippet="") for u in citations[1 : cfg.max_results])
    return hits


_ADAPTERS = {
    "openai": _search_openai,
    "firecrawl": _search_firecrawl,
    "exa": _search_exa,
    "parallel": _search_parallel,
}


def resolve_api_key(cfg, fallback_openai_key: str | None = None) -> str | None:
    if cfg.api_key:
        return cfg.api_key
    if cfg.provider == "openai" and fallback_openai_key:
        return fallback_openai_key
    return os.getenv(PROVIDER_KEY_ENV[cfg.provider])


async def run_web_search(query: str, cfg, *, fallback_openai_key: str | None = None, run_id=None) -> str:
    """Return text for the tool result. Never raises: failures become an apology-friendly error string."""
    query = (query or "").strip()
    adapter = _ADAPTERS.get(cfg.provider)
    if not query or adapter is None:
        return SEARCH_UNAVAILABLE

    api_key = resolve_api_key(cfg, fallback_openai_key)
    if not api_key:
        logger.warning(f"web_search skipped: no API key for provider={cfg.provider} {run_id=}")
        return SEARCH_UNAVAILABLE

    timeout_s = cfg.timeout_seconds

    try:
        hits = await asyncio.wait_for(adapter(query, api_key, cfg, timeout_s), timeout=timeout_s)
    except asyncio.TimeoutError:
        logger.warning(f"web_search timed out provider={cfg.provider} timeout_s={timeout_s} {run_id=}")
        return SEARCH_UNAVAILABLE
    except Exception as e:
        logger.error(f"web_search failed provider={cfg.provider} error={type(e).__name__}: {e} {run_id=}")
        return SEARCH_UNAVAILABLE

    logger.info(
        f"web_search provider={cfg.provider} query={query!r} hits={len(hits)} "
        f"sources={[_domain(h.url) for h in hits if h.url]} {run_id=}"
    )
    return format_hits(query, hits, cfg.max_results, cfg.max_snippet_chars)
