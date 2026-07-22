import re
import subprocess
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests

from handyman import procutil


class PathJailViolation(Exception):
    pass


def resolve_in_jail(working_dir: str, relative_path: str) -> Path:
    root = Path(working_dir).resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise PathJailViolation(
            f"path '{relative_path}' escapes working_dir '{working_dir}'"
        ) from None
    return candidate


def read_file(working_dir: str, path: str) -> str:
    target = resolve_in_jail(working_dir, path)
    if not target.exists():
        raise FileNotFoundError(f"no such file: {path}")
    return target.read_text(encoding="utf-8")


def write_file(working_dir: str, path: str, content: str) -> str:
    target = resolve_in_jail(working_dir, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} bytes to {path}"


def edit_file(working_dir: str, path: str, old_str: str, new_str: str) -> str:
    target = resolve_in_jail(working_dir, path)
    if not target.exists():
        raise FileNotFoundError(f"no such file: {path}")
    text = target.read_text(encoding="utf-8")
    count = text.count(old_str)
    if count == 0:
        raise ValueError(f"old_str not found in {path}")
    if count > 1:
        raise ValueError(f"old_str is not unique in {path} ({count} occurrences)")
    target.write_text(text.replace(old_str, new_str, 1), encoding="utf-8")
    return f"edited {path}"


def run_bash(working_dir: str, command: str, timeout: int = 60) -> dict:
    root = Path(working_dir).resolve()
    argv, use_shell = procutil.shell_command(command)
    process = subprocess.Popen(
        argv,
        shell=use_shell,
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **procutil.process_group_kwargs(),
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return {"stdout": stdout, "stderr": stderr, "return_code": process.returncode}
    except subprocess.TimeoutExpired:
        procutil.kill_process_tree(process.pid)
        stdout, stderr = process.communicate()
        stderr = (stderr or "") + f"\n[timed out after {timeout}s]"
        return {"stdout": stdout or "", "stderr": stderr, "return_code": -1}


from html.parser import HTMLParser


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            stripped = data.strip()
            if stripped:
                self._chunks.append(stripped)

    def text(self) -> str:
        return "\n".join(self._chunks)


def web_fetch(url: str, max_chars: int = 8000) -> str:
    resp = requests.get(url, timeout=15, headers={"User-Agent": "gemma-agent/0.1"})
    resp.raise_for_status()
    if resp.encoding == "ISO-8859-1":
        resp.encoding = resp.apparent_encoding
    extractor = _TextExtractor()
    extractor.feed(resp.text)
    return extractor.text()[:max_chars]


_DUCKDUCKGO_RESULT_PATTERN = re.compile(
    r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL
)


def _resolve_duckduckgo_href(href: str) -> str:
    """Decode a DuckDuckGo result href into its real destination URL.

    DuckDuckGo's HTML endpoint wraps result links in a protocol-relative
    redirect (`//duckduckgo.com/l/?uddg=<url-encoded-destination>&rut=...`)
    rather than linking directly. Extract the `uddg` param and use it as
    the real URL; fall back to the href as-is if the markup doesn't match
    the expected redirect shape.
    """
    absolute = "https:" + href if href.startswith("//") else href
    query = urlparse(absolute).query
    uddg = parse_qs(query).get("uddg")
    if uddg:
        return uddg[0]
    return href


def parse_duckduckgo_results(html: str, max_results: int = 5) -> list[dict]:
    results = []
    for match in _DUCKDUCKGO_RESULT_PATTERN.finditer(html):
        if len(results) >= max_results:
            break
        href, title_html = match.group(1), match.group(2)
        url = _resolve_duckduckgo_href(href)
        title = re.sub(r"<[^>]+>", "", title_html).strip()
        results.append({"url": url, "title": title})
    return results


def web_search(query: str, max_results: int = 5, tavily_api_key: str | None = None) -> list[dict]:
    """Search the web. Uses Tavily (more reliable, needs a key) when
    tavily_api_key is set; falls back to free DuckDuckGo scraping otherwise."""
    if tavily_api_key:
        return _web_search_tavily(query, max_results, tavily_api_key)
    return _web_search_duckduckgo(query, max_results)


def _web_search_tavily(query: str, max_results: int, api_key: str) -> list[dict]:
    resp = requests.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"query": query},
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    return [{"url": r["url"], "title": r["title"]} for r in results[:max_results]]


def _web_search_duckduckgo(query: str, max_results: int) -> list[dict]:
    resp = requests.get(
        "https://html.duckduckgo.com/html/",
        params={"q": query},
        timeout=15,
        headers={"User-Agent": "gemma-agent/0.1"},
    )
    resp.raise_for_status()
    if resp.encoding == "ISO-8859-1":
        resp.encoding = resp.apparent_encoding
    return parse_duckduckgo_results(resp.text, max_results=max_results)
