import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from math import log
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol
from urllib.parse import urlparse

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray
from pydantic import BaseModel, field_validator
from tokenizers import Tokenizer

from support_rag.models import SearchMode

EMBEDDING_REPO = "Xenova/multilingual-e5-small"
EMBEDDING_REVISION = "761b726dd34fb83930e26aab4e9ac3899aa1fa78"
CHUNK_SIZE = 1400
CHUNK_OVERLAP = 220
STOP_WORDS = frozenset(
    "a an and are as at be by can do for from how i if in is it of on or that the this to use "
    "was what when where which will with you your "
    "ai au aux avec ce ces cette comment dans de des du elle en est et il je la le les leur "
    "lors me mes ne nos ou par pas pour peut puis que quel quelle quelles quels qui sa sans se "
    "ses si son sur un une vos vous".split()
)
TECHNICAL_TERMS = (
    (r"\b(?:inject\w*|dependances?)\b", "dependency injection Depends"),
    (r"\b(?:remplac\w*|surcharg\w*|simul\w*|substitu\w*)\b", "override dependency"),
    (r"\b(?:services? externes?)\b", "external service"),
    (r"\b(?:sous[- ](?:appel|dependance)s?)\b", "sub-dependency"),
    (r"\b(?:partag\w*|reutilis\w*|une seule fois)\b", "reuse cached same dependency"),
    (r"\b(?:introuvable|404)\b", "raise HTTPException status code not found"),
    (r"\b(?:reponse|erreurs?)\b", "response error"),
    (r"\b(?:en[- ]tetes?)\b", "headers"),
    (r"\b(?:corps)\b", "request body"),
    (r"\b(?:caracteres|longueur)\b", "string length"),
)


class Section(BaseModel):
    id: str
    title: str
    url: str
    text: str
    topic: str

    @field_validator("url")
    @classmethod
    def check_url(cls, value: str) -> str:
        url = urlparse(value)
        if url.scheme != "https" or url.netloc != "fastapi.tiangolo.com":
            raise ValueError("A citation must link to the official FastAPI documentation.")
        return value


class Corpus(BaseModel):
    revision: str
    version: str
    sections: list[Section]

    @classmethod
    def read(cls, path: Path):
        corpus = cls.model_validate_json(path.read_text(encoding="utf-8"))
        if not corpus.sections or len({section.id for section in corpus.sections}) != len(
            corpus.sections
        ):
            raise ValueError("The corpus needs nonempty sections with unique IDs.")
        if any(not section.text.strip() for section in corpus.sections):
            raise ValueError("The corpus contains an empty section.")
        return corpus


@dataclass(frozen=True)
class Passage:
    section: Section
    text: str


@dataclass(frozen=True)
class Match:
    passage: Passage
    score: float
    similarity: float


class Embeddings(Protocol):
    def encode(self, texts: list[str]) -> NDArray[np.float32]: ...


def section_passages(section: Section) -> list[Passage]:
    text = section.text.strip()
    passages = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            boundary = text.rfind("\n", end - 250, end)
            if boundary > start:
                end = boundary
        passages.append(Passage(section, text[start:end].strip()))
        if end == len(text):
            break
        start = end - CHUNK_OVERLAP
        boundary = text.find("\n", start, end)
        if boundary != -1 and boundary < end - 50:
            start = boundary + 1
    return passages


def folded_text(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in folded if not unicodedata.combining(char))


def terms(text: str) -> list[str]:
    folded = folded_text(text)
    words = re.findall(r"[a-z0-9_]{2,}", folded)
    headers = re.findall(r"\b[a-z][a-z0-9]*(?:-[a-z0-9]+)+\b", folded)
    return [word for word in words if word not in STOP_WORDS] + headers


def bilingual_query(question: str) -> str:
    folded = folded_text(question)
    expansion = [english for pattern, english in TECHNICAL_TERMS if re.search(pattern, folded)]
    if "external service" in expansion and re.search(r"\btest\w*\b", folded):
        expansion.append("testing dependency overrides app.dependency_overrides")
    return " ".join([question, *expansion])


class E5Embedding:
    def __init__(self, cache: Path):
        local = Path(os.getenv("RAG_EMBEDDING_DIR", cache / "e5"))
        offline = os.getenv("RAG_OFFLINE", "0") == "1"

        def artifact(name: str) -> str:
            if (local / name).is_file():
                return str(local / name)
            return hf_hub_download(
                EMBEDDING_REPO,
                name,
                revision=EMBEDDING_REVISION,
                cache_dir=cache / "hf",
                local_files_only=offline,
                token=False,
            )

        self.tokenizer = Tokenizer.from_file(artifact("tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id("<pad>"), pad_token="<pad>")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            artifact("onnx/model_quantized.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {entry.name for entry in self.session.get_inputs()}

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        batches = []
        for start in range(0, len(texts), 8):
            encoded = self.tokenizer.encode_batch(texts[start : start + 8])
            mask = np.asarray([item.attention_mask for item in encoded], dtype=np.int64)
            available = {
                "input_ids": np.asarray([item.ids for item in encoded], dtype=np.int64),
                "attention_mask": mask,
                "token_type_ids": np.asarray([item.type_ids for item in encoded], dtype=np.int64),
            }
            [states] = self.session.run(
                ["last_hidden_state"], {name: available[name] for name in self.input_names}
            )
            expanded = mask[:, :, None]
            pooled = (states * expanded).sum(axis=1) / expanded.sum(axis=1)
            pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
            batches.append(pooled.astype(np.float32))
        return np.concatenate(batches)


class Index:
    def __init__(self, corpus: Corpus, embedding: Embeddings, cache: Path | None = None):
        self.corpus = corpus
        self.embedding = embedding
        self.passages = [item for section in corpus.sections for item in section_passages(section)]
        texts = [f"{p.section.title}\n{p.text}" for p in self.passages]
        self.bags = [Counter(terms(text)) for text in texts]
        lengths = [sum(bag.values()) for bag in self.bags]
        average = max(sum(lengths) / len(lengths), 1)
        self.length_norm = [1.2 * (0.25 + 0.75 * size / average) for size in lengths]
        doc_freq = Counter(term for bag in self.bags for term in bag)
        total = len(self.bags)
        self.idf = {
            term: log(1 + (total - count + 0.5) / (count + 0.5)) for term, count in doc_freq.items()
        }
        self.vectors = self._vectors(texts, cache)

    def _vectors(self, texts: list[str], cache: Path | None) -> NDArray[np.float32]:
        data = [f"passage: {text}" for text in texts]
        if cache is None:
            return self.embedding.encode(data)
        signature = sha256("\0".join([EMBEDDING_REVISION, *data]).encode()).hexdigest()[:24]
        cache.mkdir(parents=True, exist_ok=True)
        destination = cache / f"index-{signature}.npy"
        if destination.exists():
            return np.load(destination, allow_pickle=False)
        vectors = self.embedding.encode(data)
        with TemporaryDirectory(dir=cache) as folder:
            temporary = Path(folder) / "index.npy"
            np.save(temporary, vectors, allow_pickle=False)
            temporary.replace(destination)
        return vectors

    def _lexical(self, question: str) -> list[float]:
        query = set(terms(question))
        return [
            sum(
                self.idf[word] * 2.2 * bag[word] / (bag[word] + norm) for word in query & bag.keys()
            )
            for bag, norm in zip(self.bags, self.length_norm, strict=True)
        ]

    def _best_sections(self, scores) -> dict[str, int]:
        best = {}
        for index in sorted(range(len(scores)), key=lambda i: -scores[i]):
            if scores[index] > 0:
                best.setdefault(self.passages[index].section.id, index)
        return best

    def search(self, question: str, mode: SearchMode, limit: int) -> list[Match]:
        if mode == "hybrid":
            question = bilingual_query(question)
        lexical_scores = self._lexical(question) if mode != "dense" else []
        lexical = self._best_sections(lexical_scores)
        if mode == "lexical":
            first = list(lexical.values())[:limit]
            return [Match(self.passages[i], lexical_scores[i], 0.0) for i in first]

        vectors = self.embedding.encode([f"query: {question}"])
        scores = self.vectors @ vectors[0]
        dense = self._best_sections(scores)
        if mode == "dense":
            first = list(dense.values())[:limit]
            return [Match(self.passages[i], float(scores[i]), float(scores[i])) for i in first]

        ranking = {key: 2 / (18 + rank) for rank, key in enumerate(dense, start=1)}
        for rank, key in enumerate(lexical, start=1):
            if key in ranking:
                ranking[key] += 1 / (18 + rank)
        top = sorted(ranking, key=ranking.__getitem__, reverse=True)[:limit]
        return [
            Match(self.passages[dense[key]], ranking[key], float(scores[dense[key]])) for key in top
        ]
