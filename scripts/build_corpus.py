"""Rebuild the versioned corpus from FastAPI's MIT-licensed source documentation."""

import argparse
import json
import re
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "support_rag" / "data"
REVISION = "325fd16d32fd815633d6caf3d0d75aa7f30c51c9"
VERSION = "0.121.3"
REPOSITORY = "https://github.com/fastapi/fastapi"
RAW = f"https://raw.githubusercontent.com/fastapi/fastapi/{REVISION}"
DOCS = "https://fastapi.tiangolo.com"

PAGES = {
    "tutorial/dependencies/index": "Dépendances",
    "tutorial/dependencies/sub-dependencies": "Dépendances",
    "tutorial/dependencies/dependencies-in-path-operation-decorators": "Dépendances",
    "tutorial/dependencies/dependencies-with-yield": "Dépendances",
    "tutorial/testing": "Tests",
    "advanced/testing-dependencies": "Tests",
    "advanced/async-tests": "Tests",
    "tutorial/handling-errors": "Erreurs",
    "tutorial/body": "Validation",
    "tutorial/query-params-str-validations": "Validation",
    "tutorial/dependencies/global-dependencies": "Dépendances",
}

# Keep the pinned heading as the section ID. Some live pages moved older text to a parent section.
LIVE_ANCHORS = {
    "requestvalidationerror-vs-validationerror": "override-request-validation-exceptions",
    "pydantic-v1-regex-instead-of-pattern": "add-regular-expressions",
    "required-can-be-none": "required-parameters-and-none",
}

HEADING = re.compile(r"^#{1,6} (.+?) \{ #([\w-]+) \}$", re.MULTILINE)
INCLUDE = re.compile(
    r"^\s*\{(?P<kind>\*|!>)\s+\.\./\.\./(?P<path>docs_src/[\w/]+\.py)"
    r"(?P<options>[^\n}]+)\}[ \t]*$",
    re.MULTILINE,
)
LINE_RANGE = re.compile(r"ln\[(\d+):(\d+)\]")
TAB = re.compile(r"^//// tab \| [^\n]+\n(.*?)^////[ \t]*$", re.MULTILINE | re.DOTALL)


def download(path: str, cache: Path | None) -> str:
    cached = cache / path.replace("/", "_") if cache else None
    if cached and cached.is_file():
        return cached.read_text(encoding="utf-8")
    with urlopen(f"{RAW}/{path}", timeout=25) as response:
        content = response.read().decode("utf-8")
    content = content.replace("\r\n", "\n")
    if cached:
        cached.write_text(content, encoding="utf-8")
    return content


def make_readable(text: str, cache: Path | None) -> str:
    def expand_example(match: re.Match) -> str:
        source = download(match["path"], cache).rstrip()
        if line_range := LINE_RANGE.search(match["options"]):
            start, end = map(int, line_range.groups())
            source = "\n".join(source.splitlines()[start - 1 : end])
        if match["kind"] == "!>":
            return source
        return f"\n\n```python\n{source}\n```"

    parts = []
    cursor = 0
    # Keep the first example variant in each consecutive group of documentation tabs.
    for index, tab in enumerate(TAB.finditer(text)):
        between = text[cursor : tab.start()]
        parts.append(between)
        if index == 0 or between.strip():
            parts.append(tab[1])
        cursor = tab.end()
    parts.append(text[cursor:])
    text = "".join(parts)
    text = INCLUDE.sub(expand_example, text)
    text = re.sub(r"^/{3,4}(?: .*)?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^<img [^>]+>\s*$", "", text, flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def split_page(page: str, topic: str, original: str, cache: Path | None) -> list[dict[str, str]]:
    upstream_path = f"docs/en/docs/{page}.md"
    canonical_path = page.removesuffix("/index")
    leaf = canonical_path.rsplit("/", 1)[-1]
    headings = list(HEADING.finditer(original))
    if not headings:
        raise ValueError(f"No headings found in {upstream_path}")
    page_title = headings[0][1].replace("**", "")
    sections = []
    for heading, following in zip(headings, [*headings[1:], None], strict=True):
        end = following.start() if following else len(original)
        text = make_readable(original[heading.end() : end].strip(), cache)
        if not text:
            continue
        if "{* ../../docs_src" in text or "{!> ../../docs_src" in text:
            raise ValueError(f"Unresolved example in {page}#{heading[2]}")
        title = heading[1].replace("**", "").replace("`", "")
        anchor = heading[2]
        is_intro = heading is headings[0]
        line = original.count("\n", 0, heading.start()) + 1
        sections.append(
            {
                "id": f"{leaf}/{'intro' if is_intro else anchor}",
                "title": page_title if is_intro else f"{page_title} · {title}",
                "url": f"{DOCS}/{canonical_path}/#{LIVE_ANCHORS.get(anchor, anchor)}",
                "source_url": f"{REPOSITORY}/blob/{REVISION}/{upstream_path}#L{line}",
                "text": text,
                "topic": topic,
            }
        )
    return sections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, help="Optional local cache of upstream files")
    args = parser.parse_args()
    cache = args.cache / REVISION if args.cache else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)

    sections = []
    for page, topic in PAGES.items():
        original = download(f"docs/en/docs/{page}.md", cache)
        sections.extend(split_page(page, topic, original, cache))
    if len({section["id"] for section in sections}) != len(sections):
        raise ValueError("Corpus contains duplicate section IDs")

    corpus = {
        "revision": REVISION,
        "version": VERSION,
        "source": REPOSITORY,
        "license": "MIT",
        "notice": "LICENSE.fastapi.txt",
        "sections": sections,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "LICENSE.fastapi.txt").write_text(
        download("LICENSE", cache), encoding="utf-8", newline="\n"
    )
    (OUT / "corpus.json").write_text(
        json.dumps(corpus, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"FastAPI {VERSION} · {len(PAGES)} pages · {len(sections)} sections")


if __name__ == "__main__":
    main()
