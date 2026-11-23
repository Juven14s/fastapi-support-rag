"""Compare retrieval rankings and evaluate the local RAG API against annotated questions."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import unicodedata
from http.client import HTTPException
from pathlib import Path
from time import perf_counter
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
MODES = ("lexical", "dense", "hybrid")


def post(base_url: str, route: str, payload: dict, timeout: float) -> tuple[dict, float]:
    request = Request(
        f"{base_url.rstrip('/')}{route}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = perf_counter()
    with urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise ValueError("Expected an API response object.")
    return result, round((perf_counter() - started) * 1000, 2)


def folded(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold())
    return "".join(character for character in value if not unicodedata.combining(character))


def marker_match(answer: str, groups: list[list[str]]) -> bool:
    normalized = folded(answer)
    return all(any(folded(marker) in normalized for marker in group) for group in groups)


def evaluate_retrieval(args: argparse.Namespace, case: dict, mode: str) -> dict:
    result, latency = post(
        args.url,
        "/v1/retrieve",
        {"question": case["question"], "mode": mode, "limit": args.k},
        args.timeout,
    )
    ids = [hit["id"] for hit in result["hits"][: args.k]]
    rank = next((index for index, hit in enumerate(ids, 1) if hit in case["gold"]), None)
    return {"ids": ids, "first_expected_rank": rank, "client_latency_ms": latency}


def evaluate_answer(args: argparse.Namespace, case: dict, corpus_urls: dict[str, str]) -> dict:
    result, latency = post(args.url, "/v1/answer", {"question": case["question"]}, args.timeout)
    status = result["status"]
    if status not in {"answered", "abstained"}:
        raise ValueError(f"Unexpected answer status: {status}")
    citations = result["citations"]
    invalid = [
        citation["id"]
        for citation in citations
        if citation["id"] not in corpus_urls or corpus_urls[citation["id"]] != citation["url"]
    ]
    cited = {citation["id"] for citation in citations} - set(invalid)
    return {
        "status": status,
        "answer": result["answer"],
        "citation_ids": [citation["id"] for citation in citations],
        "invalid_citation_ids": invalid,
        "expected_citation": status == "answered" and bool(cited.intersection(case["gold"])),
        "all_markers_found": (
            status == "answered"
            and bool(case["gold"])
            and marker_match(result["answer"], case["markers"])
        ),
        "model_called": result["model_called"],
        "client_latency_ms": latency,
    }


def score(successes: int, total: int) -> dict:
    return {"successes": successes, "total": total, "rate": successes / total if total else None}


def latencies(rows: list[dict]) -> dict | None:
    times = sorted(row["client_latency_ms"] for row in rows if "client_latency_ms" in row)
    if not times:
        return None
    return {
        "measured": len(times),
        "median": round(statistics.median(times), 1),
        "p95": times[math.ceil(len(times) * 0.95) - 1],
    }


def summarize(cases: list[dict], modes: list[str], k: int, retrieval_only: bool) -> dict:
    answerable = [case for case in cases if case["gold"]]
    unanswerable = [case for case in cases if not case["gold"]]
    retrieval = {}
    for mode in modes:
        rows = [case["retrieval"][mode] for case in answerable]
        ranks = [row["first_expected_rank"] for row in rows if row.get("first_expected_rank")]
        retrieval[mode] = {
            "hit_at_k": score(len(ranks), len(rows)),
            "mrr_at_k": sum(1 / rank for rank in ranks) / len(rows) if rows else None,
            "errors": sum("error" in row for row in rows),
            "client_latency_ms": latencies(rows),
        }
    report = {
        "selected_cases": len(cases),
        "answerable_cases": len(answerable),
        "unanswerable_cases": len(unanswerable),
        "k": k,
        "retrieval": retrieval,
    }
    if retrieval_only:
        return report
    ordinary = [case["answer"] for case in answerable]
    unsupported = [case["answer"] for case in unanswerable]
    answers = ordinary + unsupported
    report["answers"] = {
        "answered_when_answerable": score(
            sum(row.get("status") == "answered" for row in ordinary), len(ordinary)
        ),
        "expected_citation": score(
            sum(bool(row.get("expected_citation")) for row in ordinary), len(ordinary)
        ),
        "all_answer_markers": score(
            sum(bool(row.get("all_markers_found")) for row in ordinary), len(ordinary)
        ),
        "abstained_when_unsupported": score(
            sum(
                row.get("status") == "abstained" and not row["citation_ids"] for row in unsupported
            ),
            len(unsupported),
        ),
        "answers_with_invalid_citations": sum(
            bool(row.get("invalid_citation_ids")) for row in answers
        ),
        "model_calls": sum(row.get("model_called") is True for row in answers),
        "errors": sum("error" in row for row in answers),
        "client_latency_ms": latencies(answers),
    }
    return report


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8076")
    parser.add_argument("--split", choices=("dev", "test", "all"), help="Defaults to test.")
    parser.add_argument("--mode", action="append", choices=MODES)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run only cases tagged smoke.")
    parser.add_argument("--output", type=Path, help="Also write the complete case report as JSON.")
    args = parser.parse_args()
    if args.split is None:
        args.split = "dev" if args.smoke else "test"
    if args.k < 1 or args.timeout <= 0:
        parser.error("--k and --timeout must be positive")
    return args


def main() -> int:
    args = parse_args()
    corpus = read_json(ROOT / "support_rag" / "data" / "corpus.json")
    corpus_urls = {section["id"]: section["url"] for section in corpus["sections"]}
    fixture = read_json(ROOT / "evaluation" / "cases.json")
    cases = [
        case
        for case in fixture["cases"]
        if (args.split == "all" or case["split"] == args.split)
        and (not args.smoke or case.get("smoke"))
    ]
    if not cases:
        print("No evaluation cases selected.", file=sys.stderr)
        return 2
    unknown = {gold for case in cases for gold in case["gold"]} - corpus_urls.keys()
    if unknown:
        print(f"Expected section IDs missing from corpus: {sorted(unknown)}", file=sys.stderr)
        return 2

    modes = list(dict.fromkeys(args.mode or MODES))
    for case in cases:
        case["retrieval"] = {}
        if case["gold"]:
            for mode in modes:
                try:
                    case["retrieval"][mode] = evaluate_retrieval(args, case, mode)
                except (OSError, HTTPException, ValueError, KeyError, TypeError) as error:
                    case["retrieval"][mode] = {"error": str(error)}
        if not args.retrieval_only:
            try:
                case["answer"] = evaluate_answer(args, case, corpus_urls)
            except (OSError, HTTPException, ValueError, KeyError, TypeError) as error:
                case["answer"] = {"error": str(error)}
        answer = case.get("answer", {})
        status = answer.get("status", "error") if not args.retrieval_only else "retrieval"
        print(f"{case['id']} {status}", file=sys.stderr, flush=True)

    summary = summarize(cases, modes, args.k, args.retrieval_only)
    summary["split"] = args.split
    summary["smoke"] = args.smoke
    summary["corpus_revision"] = corpus["revision"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report = json.dumps({"summary": summary, "cases": cases}, ensure_ascii=False, indent=2)
        args.output.write_text(report + "\n", encoding="utf-8")
    errors = sum(mode["errors"] for mode in summary["retrieval"].values())
    errors += summary.get("answers", {}).get("errors", 0)
    return int(errors > 0)


if __name__ == "__main__":
    raise SystemExit(main())
