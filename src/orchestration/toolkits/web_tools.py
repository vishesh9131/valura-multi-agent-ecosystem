"""Web fetch + lightweight search — shared by ProductRecommendationAgent and MCP web_server."""
from __future__ import annotations

import html as html_module
import ipaddress
import logging
import re
import socket
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlparse, urljoin

import httpx

from ...config import Settings, get_settings

logger = logging.getLogger(__name__)

_DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

_MAX_REDIRECTS = 5
_BODY_READ_CHUNK = 65536


def _strip_tags(blob: str, max_chars: int) -> str:
    t = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", blob)
    t = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = html_module.unescape(t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:max_chars]


def _dns_host_allowed(hostname: str) -> tuple[bool, str | None]:
    """Reject hosts that resolve only to non-global IPs (SSRF guard)."""
    try:
        infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror as exc:
        return False, f"dns_failed:{exc}"

    seen: set[str] = set()
    for info in infos:
        ip_str = info[4][0]
        if ip_str in seen:
            continue
        seen.add(ip_str)
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if not ip.is_global:
            return False, f"blocked_ip:{ip_str}"
    if not seen:
        return False, "no_addresses"
    return True, None


def _validate_https_url(url: str, settings: Settings) -> tuple[str | None, str | None]:
    """Returns (error_code, detail) or (None, normalized_url)."""
    raw = (url or "").strip()
    if not raw:
        return "empty_url", None
    parsed = urlparse(raw)
    if parsed.scheme.lower() != "https":
        return "https_only", None
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return "missing_host", None

    deny = [x.strip().lower() for x in (settings.web_fetch_host_denylist or "").split(",") if x.strip()]
    allow = [x.strip().lower() for x in (settings.web_fetch_host_allowlist or "").split(",") if x.strip()]
    if allow and not any(host == a or host.endswith("." + a) for a in allow):
        return "host_not_allowlisted", host
    if deny and any(host == d or host.endswith("." + d) for d in deny):
        return "host_denied", host

    ok, why = _dns_host_allowed(host)
    if not ok:
        return "dns_policy", why
    return None, raw


def _unwrap_ddg_redirect(href: str) -> str:
    """DDG wraps outbound links; pull uddg= target when present."""
    try:
        u = urlparse(href)
        if "duckduckgo.com" in (u.netloc or "").lower() and u.path.startswith("/l/"):
            qs = parse_qs(u.query)
            inner = (qs.get("uddg") or [None])[0]
            if inner:
                return unquote(inner)
    except Exception:
        pass
    return href


def _http_timeout(s: Settings) -> httpx.Timeout:
    """Separate connect vs read so TLS handshakes dont stall the whole slot."""
    conn = float(s.web_fetch_connect_timeout_s)
    read = float(s.web_fetch_timeout_s)
    return httpx.Timeout(connect=conn, read=read, write=min(read, 30.0), pool=5.0)


def _parse_ddg_html_results(body: str, mr: int) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for m in re.finditer(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        body,
        re.IGNORECASE | re.DOTALL,
    ):
        href = html_module.unescape(m.group(1).strip())
        title = _strip_tags(m.group(2), 240)
        url = _unwrap_ddg_redirect(href)
        if not url.startswith("https://"):
            continue
        results.append({"title": title or url, "url": url, "snippet": ""})
        if len(results) >= mr:
            break
    snippets = re.findall(
        r'class="result__snippet"[^>]*>(.*?)</',
        body,
        re.IGNORECASE | re.DOTALL,
    )
    for i, sn in enumerate(snippets):
        if i < len(results):
            results[i]["snippet"] = _strip_tags(sn, 400)
    return results


def _parse_lite_ddg_results(body: str, mr: int) -> list[dict[str, str]]:
    """lite.duckduckgo.com uses different HTML — outbound links still wrap uddg=."""
    results: list[dict[str, str]] = []
    pat = re.compile(
        r'href="(?:https?:)?//duckduckgo\.com/l/\?uddg=([^"&]+)"[^>]*>(.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )
    for m in pat.finditer(body):
        if len(results) >= mr:
            break
        raw_u = unquote(html_module.unescape(m.group(1).strip()))
        title = _strip_tags(m.group(2), 240)
        if not raw_u.startswith("https://"):
            continue
        results.append({"title": title or raw_u, "url": raw_u, "snippet": ""})
    return results


def _finalize_search(results: list[dict[str, str]], *, http_status: int | None, phase: str) -> dict[str, Any]:
    """ok only when we actually parsed links — avoids ok=true with empty sources."""
    if results:
        return {"ok": True, "results": results, "note": None}
    note = "no_hits"
    if http_status is not None and http_status != 200:
        note = f"http_status_{http_status}"
    elif phase:
        note = f"no_hits_{phase}"
    return {"ok": False, "results": [], "note": note}


def toolkit_web_search(
    query: str,
    *,
    max_results: int | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Best-effort DuckDuckGo HTML scrape. Returns {ok, results: [{title, url, snippet}], note}."""
    s = settings or get_settings()
    mr = max_results if max_results is not None else s.web_search_max_results
    mr = max(1, min(int(mr), 10))

    if not s.web_fetch_enabled:
        return {"ok": False, "results": [], "note": "web_fetch_disabled"}

    q = (query or "").strip()
    if len(q) < 2:
        return {"ok": False, "results": [], "note": "empty_query"}

    target = "https://html.duckduckgo.com/html/"
    err, _ = _validate_https_url(target, s)
    if err:
        return {"ok": False, "results": [], "note": err}

    headers = {"User-Agent": (s.web_fetch_user_agent or _DEFAULT_UA).strip() or _DEFAULT_UA}
    timeout = _http_timeout(s)
    results: list[dict[str, str]] = []
    last_status: int | None = None

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
            resp = client.post(target, data={"q": q})
            last_status = resp.status_code
            # 202 / non-200 usually means bot wall or empty shell — HTML parser would yield nothing.
            if resp.status_code == 200:
                results = _parse_ddg_html_results(resp.text, mr)

            if not results:
                lite_target = f"https://lite.duckduckgo.com/lite/?q={quote_plus(q)}"
                err_lite, lite_norm = _validate_https_url(lite_target, s)
                if not err_lite:
                    r2 = client.get(lite_norm)
                    last_status = r2.status_code
                    if r2.status_code == 200:
                        extra = _parse_lite_ddg_results(r2.text, mr)
                        seen = {x["url"] for x in results}
                        for row in extra:
                            if row["url"] not in seen:
                                seen.add(row["url"])
                                results.append(row)
                            if len(results) >= mr:
                                break
    except Exception as exc:
        logger.warning("web_search failed: %s", exc)
        return {"ok": False, "results": [], "note": f"request_error:{exc}"}

    return _finalize_search(results[:mr], http_status=last_status, phase="ddg")


def toolkit_fetch_url(
    url: str,
    *,
    max_chars: int | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """HTTPS GET with redirect validation and plain-text extraction."""
    s = settings or get_settings()
    mc = max_chars if max_chars is not None else s.web_fetch_max_chars

    if not s.web_fetch_enabled:
        return {"ok": False, "url": url, "text": "", "note": "web_fetch_disabled"}

    err, normalized = _validate_https_url(url, s)
    if err:
        return {"ok": False, "url": url, "text": "", "note": err}

    headers = {"User-Agent": (s.web_fetch_user_agent or _DEFAULT_UA).strip() or _DEFAULT_UA}
    timeout = _http_timeout(s)
    max_bytes = int(s.web_fetch_max_bytes)

    try:
        with httpx.Client(timeout=timeout, follow_redirects=False, headers=headers) as client:
            current = normalized
            for _ in range(_MAX_REDIRECTS + 1):
                err2, cur_norm = _validate_https_url(current, s)
                if err2:
                    return {"ok": False, "url": url, "text": "", "note": err2}
                resp = client.get(cur_norm)
                if resp.status_code in (301, 302, 303, 307, 308):
                    loc = resp.headers.get("location")
                    if not loc:
                        return {"ok": False, "url": url, "text": "", "note": "redirect_no_location"}
                    current = urljoin(cur_norm, loc)
                    continue
                resp.raise_for_status()
                raw_chunks: list[bytes] = []
                total = 0
                for chunk in resp.iter_bytes(_BODY_READ_CHUNK):
                    raw_chunks.append(chunk)
                    total += len(chunk)
                    if total >= max_bytes:
                        break
                raw = b"".join(raw_chunks)
                try:
                    text_blob = raw.decode(resp.encoding or "utf-8", errors="replace")
                except Exception:
                    text_blob = raw.decode("utf-8", errors="replace")
                plain = _strip_tags(text_blob, mc)
                return {"ok": True, "url": cur_norm, "text": plain, "note": None}
    except Exception as exc:
        logger.warning("fetch_url failed: %s", exc)
        return {"ok": False, "url": url, "text": "", "note": f"request_error:{exc}"}

    return {"ok": False, "url": url, "text": "", "note": "too_many_redirects"}
