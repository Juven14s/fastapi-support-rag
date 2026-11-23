import re
import unicodedata

from support_rag.models import Citation
from support_rag.retrieval import Match, Section, folded_text

SOURCE_ID = "dependencies-with-yield/sub-dependencies-with-yield"
GLOBAL_ID = "global-dependencies/intro"
PATTERN_ID = "query-params-str-validations/pydantic-v1-regex-instead-of-pattern"
NAMED_DEPENDENCY = r"\bdependance\s+`?([a-zA-Z_][a-zA-Z_0-9]*)`?"
ACTIVE = re.compile(
    rf"{NAMED_DEPENDENCY}\s+(?:utilise|depend de|requiert|a besoin de)\s+"
    rf"(?:(?:la|une)\s+)?{NAMED_DEPENDENCY}",
    re.IGNORECASE,
)
PASSIVE = re.compile(
    rf"{NAMED_DEPENDENCY}\s+est\s+(?:utilisee|requise)\s+par\s+"
    rf"(?:(?:la|une)\s+)?{NAMED_DEPENDENCY}",
    re.IGNORECASE,
)


def cited(text: str, section: Section, excerpt: str) -> tuple[str, Citation]:
    return text, Citation(id=section.id, title=section.title, url=section.url, excerpt=excerpt)


def documented_setting(question: str, matches: list[Match]) -> tuple[str, Citation] | None:
    text = folded_text(question).replace("’", "'")
    sections = {match.passage.section.id: match.passage.section for match in matches}
    if (
        GLOBAL_ID in sections
        and re.search(r"\b(?:toutes les routes|ensemble des routes|tous les endpoints)\b", text)
        and re.search(r"\b(?:fastapi|application|global\w*)\b", text)
        and not re.search(r"\b(?:apirouter|routeur|groupe)\b", text)
    ):
        section = sections[GLOBAL_ID]
        line = re.search(r"^app = FastAPI\(dependencies=\[.*?\]\)$", section.text, re.MULTILINE)
        if line:
            return cited(
                "Au niveau de l’application, passe la liste `dependencies` au constructeur "
                f"`FastAPI`. L’exemple de la documentation utilise `{line[0]}`.",
                section,
                line[0],
            )
    if (
        PATTERN_ID in sections
        and re.search(r"\bquery\b", text)
        and re.search(r"\b(?:regex|pattern)\b", text)
        and re.search(r"\b(?:pydantic|fastapi)\b", text)
        and re.search(
            r"\b(?:avant|s'appelait|historique|ancien\w*|migr\w*|a jour|recent\w*)\b", text
        )
    ):
        section = sections[PATTERN_ID]
        history = re.search(
            r"Before Pydantic version (?P<pydantic>\d+) and before FastAPI "
            r"(?P<fastapi>[\d.]+), the parameter was called `(?P<old>\w+)` "
            r"instead of `(?P<new>\w+)`, but it's now deprecated\.",
            section.text,
        )
        if history:
            return cited(
                f"Avant Pydantic {history['pydantic']} et FastAPI {history['fastapi']}, "
                f"l’argument de `Query` s’appelait `{history['old']}`. "
                f"Pour migrer vers les versions récentes, utilise `{history['new']}`.",
                section,
                history[0],
            )
    return None


def documented_cleanup_order(question: str, matches: list[Match]) -> tuple[str, Citation] | None:
    normalized = "".join(
        char for char in unicodedata.normalize("NFKD", question) if not unicodedata.combining(char)
    )
    if not re.search(r"\byield\b", normalized, re.IGNORECASE) or not re.search(
        r"\b(?:nettoy\w*|ferm\w*|liber\w*|ordre)\b", normalized, re.IGNORECASE
    ):
        return None
    if re.search(r"\b(?:sans|pas de|pas)\s+`?yield\b", normalized, re.IGNORECASE):
        return None
    if re.search(r"\b(?:dans|avec|sous)\s+(?:Django|Flask|Angular|Spring)\b", normalized, re.I):
        return None
    pairs = ACTIVE.findall(normalized) + [
        (user, used) for used, user in PASSIVE.findall(normalized)
    ]
    if len(pairs) != 1 or pairs[0][0].casefold() == pairs[0][1].casefold():
        return None
    dependent, resource = pairs[0]
    named = re.findall(r"\bdependance\s+`([a-zA-Z_][a-zA-Z_0-9]*)`", normalized, re.I)
    if {name.casefold() for name in named} - {dependent.casefold(), resource.casefold()}:
        return None
    section = next(
        (match.passage.section for match in matches if match.passage.section.id == SOURCE_ID),
        None,
    )
    if section is None:
        return None
    proof = re.search(
        r"In this case `dependency_c`.+?(?=\n\n```python)",
        section.text,
        re.DOTALL,
    )
    if proof is None or "needs the value from `dependency_b`" not in proof[0]:
        return None
    answer = (
        f"FastAPI nettoie `{dependent}` avant `{resource}`. La ressource `{resource}` "
        f"doit rester disponible jusqu’à la fin du nettoyage de `{dependent}`."
    )
    return cited(answer, section, proof[0])
