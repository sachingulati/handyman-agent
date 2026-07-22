import tools

DUCKDUCKGO_FIXTURE = """
<div class="results">
  <div class="result">
    <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fone&rut=abc123">Example <b>One</b></a>
  </div>
  <div class="result">
    <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Ftwo&rut=def456">Example Two</a>
  </div>
</div>
"""


def test_parse_duckduckgo_results_extracts_url_and_title():
    results = tools.parse_duckduckgo_results(DUCKDUCKGO_FIXTURE)
    assert results == [
        {"url": "https://example.com/one", "title": "Example One"},
        {"url": "https://example.com/two", "title": "Example Two"},
    ]


def test_parse_duckduckgo_results_respects_max_results():
    results = tools.parse_duckduckgo_results(DUCKDUCKGO_FIXTURE, max_results=1)
    assert len(results) == 1
    assert results[0]["url"] == "https://example.com/one"


def test_parse_duckduckgo_results_empty_html_returns_empty_list():
    assert tools.parse_duckduckgo_results("<html></html>") == []


def test_parse_duckduckgo_results_falls_back_to_raw_href_without_uddg():
    html = """
    <a class="result__a" href="https://example.com/direct">Direct Link</a>
    """
    results = tools.parse_duckduckgo_results(html)
    assert results == [{"url": "https://example.com/direct", "title": "Direct Link"}]


def test_text_extractor_strips_tags_and_scripts():
    extractor = tools._TextExtractor()
    extractor.feed(
        "<html><body><script>ignoreMe();</script>"
        "<p>Hello <b>world</b></p><style>.x{color:red}</style></body></html>"
    )
    text = extractor.text()
    assert "Hello" in text
    assert "world" in text
    assert "ignoreMe" not in text
    assert "color:red" not in text


def test_web_fetch_extracts_text(monkeypatch):
    class FakeResponse:
        text = "<html><body><p>fetched content</p></body></html>"
        encoding = "utf-8"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: FakeResponse())
    result = tools.web_fetch("https://example.com")
    assert "fetched content" in result


def test_web_fetch_truncates_to_max_chars(monkeypatch):
    class FakeResponse:
        text = "<p>" + ("x" * 100) + "</p>"
        encoding = "utf-8"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: FakeResponse())
    result = tools.web_fetch("https://example.com", max_chars=10)
    assert len(result) == 10


def test_web_fetch_corrects_iso_8859_1_fallback_encoding(monkeypatch):
    class FakeResponse:
        encoding = "ISO-8859-1"
        apparent_encoding = "utf-8"

        def raise_for_status(self):
            pass

        @property
        def text(self):
            # requests decodes the raw bytes using whatever `.encoding` is
            # set to at the time `.text` is accessed. Simulate that here:
            # only the corrected encoding produces the properly-decoded body.
            if self.encoding == "utf-8":
                return "<html><body><p>café</p></body></html>"
            return "<html><body><p>mojibake</p></body></html>"

    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: FakeResponse())
    result = tools.web_fetch("https://example.com")
    assert "café" in result


def test_web_search_calls_duckduckgo_and_parses(monkeypatch):
    class FakeResponse:
        text = DUCKDUCKGO_FIXTURE
        encoding = "utf-8"

        def raise_for_status(self):
            pass

    captured = {}

    def fake_get(url, params=None, timeout=None, headers=None):
        captured["url"] = url
        captured["params"] = params
        return FakeResponse()

    monkeypatch.setattr(tools.requests, "get", fake_get)
    results = tools.web_search("cats")
    assert captured["url"] == "https://html.duckduckgo.com/html/"
    assert captured["params"] == {"q": "cats"}
    assert results[0]["url"] == "https://example.com/one"


def test_web_search_uses_duckduckgo_when_no_tavily_key(monkeypatch):
    class FakeResponse:
        text = DUCKDUCKGO_FIXTURE
        encoding = "utf-8"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: FakeResponse())

    def fail_if_called(*a, **k):
        raise AssertionError("Tavily must not be called when no key is provided")

    monkeypatch.setattr(tools.requests, "post", fail_if_called)

    results = tools.web_search("cats")
    assert results[0]["url"] == "https://example.com/one"


def test_web_search_uses_tavily_when_key_provided(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "results": [
                    {"url": "https://example.com/tavily-one", "title": "Tavily One", "content": "..."},
                    {"url": "https://example.com/tavily-two", "title": "Tavily Two", "content": "..."},
                ]
            }

    captured = {}

    def fail_if_called(*a, **k):
        raise AssertionError("DuckDuckGo must not be called when a Tavily key is provided")

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr(tools.requests, "get", fail_if_called)
    monkeypatch.setattr(tools.requests, "post", fake_post)

    results = tools.web_search("cats", tavily_api_key="tvly-test-key")

    assert captured["url"] == "https://api.tavily.com/search"
    assert captured["headers"]["Authorization"] == "Bearer tvly-test-key"
    assert captured["json"] == {"query": "cats"}
    assert results == [
        {"url": "https://example.com/tavily-one", "title": "Tavily One"},
        {"url": "https://example.com/tavily-two", "title": "Tavily Two"},
    ]


def test_web_search_tavily_respects_max_results(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "results": [
                    {"url": "https://example.com/1", "title": "One"},
                    {"url": "https://example.com/2", "title": "Two"},
                    {"url": "https://example.com/3", "title": "Three"},
                ]
            }

    monkeypatch.setattr(tools.requests, "post", lambda *a, **k: FakeResponse())
    results = tools.web_search("cats", max_results=1, tavily_api_key="tvly-test-key")
    assert len(results) == 1
    assert results[0]["url"] == "https://example.com/1"


def test_web_search_corrects_iso_8859_1_fallback_encoding(monkeypatch):
    class FakeResponse:
        encoding = "ISO-8859-1"
        apparent_encoding = "utf-8"

        def raise_for_status(self):
            pass

        @property
        def text(self):
            if self.encoding == "utf-8":
                return DUCKDUCKGO_FIXTURE
            return "<html><body>mojibake</body></html>"

    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: FakeResponse())
    results = tools.web_search("cats")
    assert results[0]["url"] == "https://example.com/one"
