import asyncio
from pathlib import Path

import httpx
import numpy as np

from support_rag.api import app
from support_rag.documented import (
    GLOBAL_ID,
    PARAMETER_SOURCE_ID,
    PATTERN_ID,
    SOURCE_ID,
    YIELD_SCOPE_ID,
    documented_cleanup_order,
    documented_setting,
)
from support_rag.generation import available_evidence, checked_citations
from support_rag.models import Citation, Draft
from support_rag.retrieval import Corpus, Index, Match, Passage, Section
from support_rag.scope import requests_private_facts

SECTION = Section(
    id="testing/using-testclient",
    title="Testing · Using TestClient",
    url="https://fastapi.tiangolo.com/tutorial/testing/#using-testclient",
    text="Create a TestClient by passing your FastAPI application to it.",
    topic="testing",
)
SOURCES = {
    "S1.1": Citation(id=SECTION.id, title=SECTION.title, url=SECTION.url, excerpt=SECTION.text)
}


class FixedEmbedding:
    def encode(self, texts):
        return np.ones((len(texts), 1), dtype=np.float32)


def test_citations_are_resolved_to_an_exact_upstream_excerpt():
    draft = Draft(
        status="answered",
        answer="Passe l’application FastAPI à TestClient.",
        evidence=["S1.1"],
    )
    [citation] = checked_citations(draft, SOURCES, "Comment utiliser TestClient ?")
    assert citation.id == SECTION.id
    assert citation.url == SECTION.url
    assert citation.excerpt == SECTION.text


def test_unknown_source_and_invented_excerpt_id_are_rejected():
    for evidence in ("S9.1", "S1.999"):
        draft = Draft(status="answered", answer="Une réponse.", evidence=[evidence])
        assert not checked_citations(draft, SOURCES, "Comment tester une route ?")


def test_multiple_excerpts_from_one_section_are_one_citation():
    draft = Draft(
        status="answered",
        answer="Passe l’application FastAPI à TestClient.",
        evidence=["S1.1", "S1.2", "S1.3", "S1.4"],
    )
    sources = {"S1.1": SOURCES["S1.1"]}
    for number in range(2, 5):
        sources[f"S1.{number}"] = SOURCES["S1.1"].model_copy(
            update={"excerpt": "Test configuration."}
        )
    assert checked_citations(draft, sources, "Comment utiliser TestClient ?") == [SOURCES["S1.1"]]


def test_shortlist_keeps_relevant_text_when_search_finds_a_late_chunk():
    late = "client = TestClient(app)"
    section = SECTION.model_copy(update={"text": f"{SECTION.text}\n\n{late}"})
    match = Match(Passage(section, late), score=1.0, similarity=1.0)
    index = Index(Corpus(version="0", revision="test", sections=[section]), FixedEmbedding())
    evidence = available_evidence(index, "Comment utiliser TestClient ?", [match])
    assert SECTION.text in evidence["S1.1"].excerpt
    assert all(len(value.excerpt) <= 420 for value in evidence.values())


def test_unsupported_literals_cannot_be_validated_by_an_existing_citation():
    for text in ("Tous les tests ont réussi à 100 %.", "Utilise `app.invented_method`."):
        draft = Draft(status="answered", answer=text, evidence=["S1.1"])
        assert not checked_citations(draft, SOURCES, "Comment utiliser TestClient ?")


def test_private_data_request_abstains_before_model_and_retrieval():
    async def request_without_index():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.post(
                "/v1/answer",
                json={"question": "Combien de tests sont passés sur notre serveur hier ?"},
            )

    response = asyncio.run(request_without_index())
    assert response.status_code == 200
    assert response.json()["status"] == "abstained"
    assert response.json()["model_called"] is False
    assert not requests_private_facts("Comment tester mon API sans démarrer mon serveur ?")
    assert requests_private_facts("Quel est le jeton de mon instance de préproduction ?")


def test_borderline_match_gets_a_reviewed_answer(monkeypatch):
    index = Index(Corpus(version="0", revision="test", sections=[SECTION]), FixedEmbedding())
    match = Match(Passage(SECTION, SECTION.text), score=0.1, similarity=0.812)
    monkeypatch.setattr(index, "search", lambda *_args: [match])

    class ReviewModel:
        def generate(self, question, evidence, *, max_tokens):
            return Draft(
                status="answered", answer="Lance une application distante.", evidence=["S1.1"]
            )

        def review(self, question, proposal, evidence, *, max_tokens):
            assert proposal == "Lance une application distante."
            assert evidence["S1.1"].excerpt == SECTION.text
            return Draft(status="answered", answer="Passe ton application.", evidence=["S1.1"])

    monkeypatch.setattr(app.state, "index", index, raising=False)
    monkeypatch.setattr(app.state, "model", ReviewModel(), raising=False)

    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.post(
                "/v1/answer", json={"question": "Comment utiliser TestClient ?"}
            )

    response = asyncio.run(request()).json()
    assert response["answer"] == "Passe ton application."
    assert response["model_attempts"] == 2


def test_lexical_search_returns_unique_document_sections():
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


def test_cleanup_order_is_sourced_and_respects_active_and_passive_relations():
    path = Path(__file__).resolve().parents[1] / "support_rag/data/corpus.json"
    section = next(item for item in Corpus.read(path).sections if item.id == SOURCE_ID)
    match = Match(Passage(section, section.text), score=1.0, similarity=1.0)
    for question in (
        "Avec yield, quel ordre de nettoyage "
        "si la dépendance Service utilise la dépendance Connexion ?",
        "Avec yield, qui ferme en premier "
        "si la dépendance Connexion est utilisée par la dépendance Service ?",
    ):
        answer, citation = documented_cleanup_order(question, [match])
        assert "`Service` avant `Connexion`" in answer
        assert citation.excerpt in section.text
    assert documented_cleanup_order(question, []) is None


def test_documented_config_and_version_preserve_scope_and_quoted_proof():
    path = Path(__file__).resolve().parents[1] / "support_rag/data/corpus.json"
    sections = {
        item.id: item for item in Corpus.read(path).sections if item.id in {GLOBAL_ID, PATTERN_ID}
    }
    global_section = sections[GLOBAL_ID]
    global_match = Match(Passage(global_section, global_section.text), score=1.0, similarity=1.0)
    answer, reference = documented_setting(
        "Comment appliquer une dépendance à toutes les routes d'une application FastAPI ?",
        [global_match],
    )
    assert "FastAPI(dependencies=" in answer
    assert reference.excerpt in global_section.text
    version_section = sections[PATTERN_ID]
    version_match = Match(Passage(version_section, version_section.text), score=1.0, similarity=1.0)
    answer, reference = documented_setting(
        "Mon code Pydantic v1 utilise Query(regex=...). "
        "Je le mets à jour, quel argument utiliser ?",
        [version_match],
    )
    assert "utilise `pattern`" in answer
    assert reference.excerpt in version_section.text


def test_default_scope_cites_cleanup_after_response():
    path = Path(__file__).resolve().parents[1] / "support_rag/data/corpus.json"
    section = next(item for item in Corpus.read(path).sections if item.id == YIELD_SCOPE_ID)
    match = Match(Passage(section, section.text), score=1.0, similarity=1.0)
    answer, citation = documented_setting(
        "À quel moment le bloc après yield s’exécute-t-il sans préciser de `scope` ?", [match]
    )
    assert "une fois la réponse envoyée" in answer
    assert citation.excerpt in section.text
    assert documented_setting(
        'Quand se ferme la dépendance yield si je fixe scope="function" ?', [match]
    ) is None


def test_scalar_source_with_body_respects_explicit_annotations():
    path = Path(__file__).resolve().parents[1] / "support_rag/data/corpus.json"
    section = next(item for item in Corpus.read(path).sections if item.id == PARAMETER_SOURCE_ID)
    match = Match(Passage(section, section.text), score=1.0, similarity=1.0)
    question = (
        "Une route reçoit un modèle Pydantic et un paramètre str absent du chemin. "
        "D’où vient le paramètre str ?"
    )
    answer, citation = documented_setting(question, [match])
    assert "paramètres de requête de l’URL" in answer
    assert citation.excerpt in section.text
    assert documented_setting(f"{question} Le paramètre est déclaré via Header().", [match]) is None
