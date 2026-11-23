import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from support_rag.generation import ABSTENTION, LocalModel, checked_citations
from support_rag.models import Answer, Question, SearchHit, SearchRequest, SearchResponse
from support_rag.retrieval import Corpus, E5Embedding, Index

logger = logging.getLogger(__name__)
PACKAGE = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    corpus = Corpus.read(PACKAGE / "data" / "corpus.json")
    cache = Path(os.getenv("RAG_EMBEDDING_CACHE", ".cache/embeddings"))
    app.state.index = Index(corpus, E5Embedding(cache), cache)
    app.state.model = LocalModel(
        os.getenv("RAG_MODEL_URL", "http://127.0.0.1:8080/v1"),
        os.getenv("RAG_MODEL_NAME", "qwen3-4b-instruct"),
        float(os.getenv("RAG_MODEL_TIMEOUT_SECONDS", "120")),
    )
    yield


app = FastAPI(title="FastAPI Support RAG", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=PACKAGE / "static", check_dir=False), name="static")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(PACKAGE / "static" / "index.html")


@app.get("/healthz")
def health(request: Request):
    index = request.app.state.index
    return {
        "status": "ok",
        "corpus_version": index.corpus.version,
        "sections": len(index.corpus.sections),
        "passages": len(index.passages),
    }


@app.post("/v1/retrieve", response_model=SearchResponse)
def retrieve(body: SearchRequest, request: Request):
    started = perf_counter()
    matches = request.app.state.index.search(body.question, body.mode, body.limit)
    return SearchResponse(
        hits=[
            SearchHit(
                id=match.passage.section.id,
                title=match.passage.section.title,
                url=match.passage.section.url,
                score=match.score,
            )
            for match in matches
        ],
        latency_ms=round((perf_counter() - started) * 1000, 2),
    )


@app.post("/v1/answer", response_model=Answer)
def answer(body: Question, request: Request):
    started = perf_counter()
    matches = request.app.state.index.search(body.question, "hybrid", 5)
    try:
        draft = request.app.state.model.generate(body.question, matches)
        citations = checked_citations(draft, matches)
    except httpx.HTTPError as error:
        logger.warning("Local generation unavailable (%s)", type(error).__name__)
        raise HTTPException(503, "Le modèle local est indisponible.") from error
    except (ValueError, TypeError, KeyError, IndexError) as error:
        logger.warning("Local generation emitted an invalid response (%s)", type(error).__name__)
        citations = []
    return Answer(
        status="answered" if citations else "abstained",
        answer=draft.answer.strip() if citations else ABSTENTION,
        citations=citations,
        latency_ms=round((perf_counter() - started) * 1000, 2),
        model_called=True,
    )
