"""SSRF + HTML helpers for web_tools."""
from __future__ import annotations

import pytest

from src.config import reset_settings_cache
from src.orchestration.toolkits import web_tools as wt


@pytest.fixture
def web_on(monkeypatch):
    monkeypatch.setenv("WEB_FETCH_ENABLED", "true")
    reset_settings_cache()
    yield
    monkeypatch.delenv("WEB_FETCH_ENABLED", raising=False)
    reset_settings_cache()


def test_web_search_disabled_returns_empty():
    reset_settings_cache()
    out = wt.toolkit_web_search("anything")
    assert out["ok"] is False
    assert out["note"] == "web_fetch_disabled"


def test_fetch_url_requires_https(web_on):
    out = wt.toolkit_fetch_url("http://example.com/")
    assert out["ok"] is False
    assert out["note"] == "https_only"


def test_fetch_url_blocks_loopback(web_on):
    out = wt.toolkit_fetch_url("https://127.0.0.1/")
    assert out["ok"] is False
    assert out["note"] == "dns_policy"


def test_fetch_url_allowlist_blocks_unknown_host(web_on, monkeypatch):
    monkeypatch.setenv("WEB_FETCH_HOST_ALLOWLIST", "example.org")
    reset_settings_cache()
    out = wt.toolkit_fetch_url("https://example.com/")
    assert out["ok"] is False
    assert out["note"] == "host_not_allowlisted"


def test_parse_ddg_html(web_on, monkeypatch):
    html = """
    <html><body>
    <a rel="nofollow" class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Ffund.example.com%2Fx">
      Cool Dividend Fund
    </a>
    <a class="result__snippet">Low fee index blah blah.</a>
    </body></html>
    """

    class Resp:
        status_code = 200
        text = html

        def raise_for_status(self):
            return None

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, data=None):
            assert "duckduckgo" in url
            return Resp()

    monkeypatch.setattr(wt.httpx, "Client", FakeClient)
    out = wt.toolkit_web_search("dividend ETF")
    assert out["ok"] is True
    assert len(out["results"]) >= 1
    assert out["results"][0]["url"].startswith("https://fund.example.com")


def test_web_search_falls_back_to_lite_on_202(web_on, monkeypatch):
    lite_html = (
        '<a href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fsrcexample.com%2Fpath">From lite</a>'
    )

    class Resp202:
        status_code = 202
        text = ""

    class RespLite:
        status_code = 200
        text = lite_html

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, data=None):
            assert "html.duckduckgo.com" in url
            return Resp202()

        def get(self, url):
            assert "lite.duckduckgo.com" in url
            return RespLite()

    monkeypatch.setattr(wt.httpx, "Client", FakeClient)
    out = wt.toolkit_web_search("foo query")
    assert out["ok"] is True
    assert out["results"][0]["url"] == "https://srcexample.com/path"


def test_web_search_ok_false_when_still_no_links(web_on, monkeypatch):
    class RespEmpty:
        status_code = 200
        text = "<html><body>nothing</body></html>"

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, data=None):
            return RespEmpty()

        def get(self, url):
            return RespEmpty()

    monkeypatch.setattr(wt.httpx, "Client", FakeClient)
    out = wt.toolkit_web_search("zzz")
    assert out["ok"] is False
    assert out["results"] == []


def test_strip_tags():
    raw = "<div>Hello <b>world</b> &amp; friends<script>x</script></div>"
    assert wt._strip_tags(raw, 100) == "Hello world & friends"
