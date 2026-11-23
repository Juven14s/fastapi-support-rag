from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

SearchMode = Literal["lexical", "dense", "hybrid"]
Prompt = Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=1000)]


class Question(BaseModel):
    question: Prompt


class SearchRequest(Question):
    mode: SearchMode = "hybrid"
    limit: int = Field(default=5, ge=1, le=20)


class SearchHit(BaseModel):
    id: str
    title: str
    url: str
    score: float


class SearchResponse(BaseModel):
    hits: list[SearchHit]
    latency_ms: float


class Citation(BaseModel):
    id: str
    title: str
    url: str
    excerpt: str


class Answer(BaseModel):
    status: Literal["answered", "abstained"]
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    latency_ms: float
    model_called: bool
    model_attempts: int = 0


class Draft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["answered", "abstained"]
    answer: str
    evidence: list[str]

    @model_validator(mode="after")
    def evidence_required(self):
        if self.status == "answered" and (not self.answer.strip() or not self.evidence):
            raise ValueError("An answer requires at least one piece of evidence.")
        return self
