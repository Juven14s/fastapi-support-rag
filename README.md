# FastAPI Support RAG

Cet assistant répond en français à des questions sur les dépendances, les tests, la gestion des erreurs et la validation dans FastAPI. Il retrouve des passages de la documentation officielle, puis un modèle local rédige la réponse. Chaque citation affichée contient un extrait vérifié dans le texte indexé et un lien vers la section. Si la réponse manque de preuves, l’application s’abstient.

> Comment tester une route qui appelle un service externe sans déclencher le vrai appel dans chaque test ?

## Démarrer

Avec Docker Desktop démarré, lance le projet depuis sa racine. Les services sont accessibles en local. Le premier lancement télécharge les poids publics de Qwen et du modèle d’embeddings. Aucune clé ni aucun service payant n’est nécessaire.

```powershell
./scripts/docker.ps1 compose up --build --wait --wait-timeout 480
```

- [Interface de support](http://127.0.0.1:8076)
- [Documentation interactive de l’API](http://127.0.0.1:8076/docs)

Si le modèle du projet `enterprise-agentic-analytics` tourne déjà sur cette machine, ce lancement utilise la même instance Qwen et économise la mémoire.

```powershell
./scripts/docker.ps1 compose -f compose.yaml -f compose.shared-model.yaml up --build -d --wait --wait-timeout 180 app
```

## Du document à la réponse

1. Le corpus contient 111 sections de dix pages de FastAPI 0.121.3. Le générateur résout les exemples de code au même commit que la documentation et conserve les URL de section. [Instantané officiel](https://github.com/fastapi/fastapi/tree/325fd16d32fd815633d6caf3d0d75aa7f30c51c9/docs/en/docs)
2. Une recherche BM25 par mots-clés sert de référence. Le modèle multilingue E5 calcule les embeddings en local sur CPU avec ONNX. Le mode hybride ajoute des termes techniques anglais à la question et combine les rangs des deux recherches. [Modèle E5](https://huggingface.co/intfloat/multilingual-e5-small) · [Poids ONNX utilisés](https://huggingface.co/Xenova/multilingual-e5-small/tree/761b726dd34fb83930e26aab4e9ac3899aa1fa78)
3. Qwen reçoit les extraits numérotés et renvoie une réponse avec les identifiants des passages justificatifs. L’API accepte uniquement les identifiants transmis au modèle et affiche les extraits originaux et leurs liens officiels.

Le corpus provient de la documentation FastAPI sous licence MIT. Le texte de la licence et l’attribution sont inclus dans [LICENSE.fastapi.txt](support_rag/data/LICENSE.fastapi.txt).

## Vérifier les résultats

Les 32 questions d’évaluation sont divisées en 12 cas de développement et 20 cas de test. Le runner mesure séparément si le passage attendu apparaît dans les premiers résultats, si la réponse cite une source attendue, si des termes essentiels sont présents et si le système s’abstient hors corpus. La présence de termes sert d’indicateur automatique. Une lecture humaine reste nécessaire pour évaluer l’exactitude complète et la qualité des explications.

Sur les 9 questions de développement qui ont une réponse dans le corpus, le passage attendu est apparu dans les cinq premiers résultats 5 fois avec BM25, 5 fois avec les embeddings seuls et 9 fois avec la recherche hybride. Ces cas ont servi au réglage.

Le premier essai de bout en bout a renvoyé les deux citations attendues pour deux questions documentées et une abstention pour une question sur SvelteKit. Sur ces trois appels locaux, la latence médiane était de 31 s. Les 20 questions du test final sont réservées à l’évaluation de la version stabilisée.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/evaluate.py --split dev --retrieval-only
.\.venv\Scripts\python.exe scripts/evaluate.py --smoke
.\.venv\Scripts\python.exe scripts/evaluate.py --split test --output artifacts/evaluation-test.json
```
