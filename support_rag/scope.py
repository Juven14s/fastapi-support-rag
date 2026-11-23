import re

from support_rag.retrieval import folded_text

PRIVATE_CONTEXT = re.compile(
    r"\b(?:mon|ma|mes|notre|nos|leur|leurs)\s+"
    r"(?:serveurs?|clusters?|bases?(?: de donnees)?|environnements?|instances?|"
    r"deployements?|pipelines?|projets?|applications?|apis?|equipes?|depots?|"
    r"(?:(?:derniers?|derniere?) )?(?:tests?|builds?|logs?|journaux|metriques))\b|"
    r"\b(?:chez nous|en interne|de mon equipe|de notre equipe)\b"
)
PRIVATE_FACTS = re.compile(
    r"\b(?:quel(?:le|les|s)?\s+(?:est (?:le|la|mon|ma|notre)\s+)?"
    r"(?:pourcentage|taux|secret|token|jeton|mot de passe|cle|adresse|valeur|version|"
    r"configuration|identifiant|score|latence|resultat(?:s)?)|"
    r"combien (?:de|d['’]|ont|a |sont)|"
    r"quels? (?:tests|builds|deploiements?) (?:ont|a |sont)|"
    r"(?:donne|affiche|liste|retrouve|lis) (?:le[s]?|la|mon|ma|mes|notre|nos) "
    r"(?:vraie? |actuel(?:le)? )?(?:mots? de passe|secrets?|tokens?|cles?|"
    r"resultats?|logs?|journaux|identifiants?|configuration))\b"
)
OBSERVED_PERIOD = re.compile(
    r"\b(?:hier|ce matin|actuellement|la semaine derniere|le mois dernier|cette nuit)\b"
)


def requests_private_facts(question: str) -> bool:
    text = " ".join(folded_text(question).split())
    return bool(
        PRIVATE_FACTS.search(text)
        and (PRIVATE_CONTEXT.search(text) or OBSERVED_PERIOD.search(text))
    )
