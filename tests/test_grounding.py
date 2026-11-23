from pathlib import Path

import numpy as np

from support_rag.generation import checked_citations
from support_rag.models import Draft
from support_rag.retrieval import Corpus, Index, Match, Passage, Section

SECTION = Section(
    id="testing/using-testclient",
    title="Testing · Using TestClient",
    url="https://fastapi.tiangolo.com/tutorial/testing/#using-testclient",
    text="Create a TestClient by passing your FastAPI application to it.",
    topic="testing",
)


def test_citations_are_resolved_to_an_exact_upstream_excerpt():
    draft = Draft(
        status="answered",
        answer="Passe l’application FastAPI à TestClient.",
        evidence=["S1.1"],
    )
    match = Match(Passage(SECTION, SECTION.text), score=1.0, similarity=1.0)
    [citation] = checked_citations(draft, [match])
    assert citation.id == SECTION.id
    assert citation.url == SECTION.url
    assert citation.excerpt == SECTION.text


def test_unknown_source_and_invented_excerpt_id_are_rejected():
    match = Match(Passage(SECTION, SECTION.text), score=1.0, similarity=1.0)
    for evidence in ("S9.1", "S1.999"):
        draft = Draft(status="answered", answer="Une réponse.", evidence=[evidence])
        assert not checked_citations(draft, [match])


def test_context_includes_the_section_introduction_when_search_finds_a_late_chunk():
    late = "client = TestClient(app)"
    section = SECTION.model_copy(update={"text": f"{SECTION.text}\n\n{late}"})
    match = Match(Passage(section, late), score=1.0, similarity=1.0)
    draft = Draft(status="answered", answer="Une réponse.", evidence=["S1.1"])
    assert SECTION.text in checked_citations(draft, [match])[0].excerpt


def test_lexical_search_returns_unique_document_sections():
    class FixedEmbedding:
        def encode(self, texts):
            return np.ones((len(texts), 1), dtype=np.float32)

    long_section = SECTION.model_copy(update={"text": SECTION.text * 80})
    index = Index(Corpus(version="0", revision="test", sections=[long_section]), FixedEmbedding())
    assert len(index.passages) > 1
    assert [item.passage.section.id for item in index.search("TestClient", "lexical", 5)] == [
        SECTION.id
    ]
    assert not index.search("Tableaux de bord financiers", "lexical", 5)


def test_versioned_corpus_has_unique_sections_and_official_links():
    path = Path(__file__).resolve().parents[1] / "support_rag/data/corpus.json"
    corpus = Corpus.read(path)
    assert len(corpus.revision) == 40
    assert any(section.id.startswith("testing-dependencies/") for section in corpus.sections)
