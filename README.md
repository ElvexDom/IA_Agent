# IA Agentic — LangChain / LlamaIndex

Travaux pratiques du support **DEV IA 2025 — IA Agentic (Antony Schutz)**.
Ce README suit l'avancement réel du dépôt par rapport au sommaire du PDF.

## Modèle utilisé

Les scripts du dossier `chatbots/` suivent le PDF tel quel (**Ollama** en local).
Les scripts à la racine (`agent_tools.py`, `agent_evaluateur_sav.py`, à venir) utilisent
**Groq** (`ChatGroq`, modèle `openai/gpt-oss-120b`) à la place d'Ollama, trop lourd sur la machine de dev.

Clé API attendue dans un fichier `.env` à la racine (non versionné, voir `.gitignore`).
Un modèle sans secret est fourni dans `.env.example` :
```
cp .env.example .env   # puis y coller sa vraie clé GROQ_API_KEY
```

## Avancement

| Chapitre (page PDF) | Statut | Fichier(s) |
|---|---|---|
| Fondamentaux (p.5) | ✅ | — (exploration Ollama, pas de script dédié conservé) |
| Chatbot — stateless / stateful (p.6) | ✅ | `chatbots/01_chatbot_stateless.py`, `chatbots/02_chatbot_stateful.py` |
| Chatbot LangChain — LCEL (p.8) | ✅ | `chatbots/03_chatbot_langchain_lcel.py` |
| Optimisation de la mémoire (p.10) | ✅ | `chatbots/04_chatbot_memoire_strategies.py` (full buffer, sliding window, summary buffer, moving summary, compactage) |
| Persistance et bases vectorielles (p.12) | ✅ | `chatbots/05_persistance_chroma.py` (Chroma local), `chatbots/06_persistance_pgvector.py` (PGVector distant) |
| RAG (p.15) | ✅ | `chatbots/07_rag_langchain_chroma.py` (menu du restaurant), `chatbots/08_rag_llamaindex_faiss.py` (référentiel Simplon) |
| Agents et Outils (p.20) | ✅ | `agent_tools.py` |
| Nœuds d'évaluation (p.27) | ✅ | `agent_evaluateur_sav.py` + `db_init_boutique.py` |
| **Nœud de Protection — Guardrail (p.33)** | ✅ | `agent_evaluateur_sav.py` (nœud `guardrail` en entrée du graphe) |
| Human-in-the-loop (p.38) | ⏭️ à faire | — |
| CheckPoint / Time Travel (p.43) | ⏳ | — |
| Multi-Agent & Orchestration (p.45) | ⏳ | — |
| Connecteur MCP (p.58) | ⏳ | — |

## Structure du dépôt

```
.
├── .env                        # GROQ_API_KEY (non versionné, voir .gitignore)
├── .env.example                # Modèle de .env sans secret (versionné)
├── .gitignore                  # Exclut .env, *.db, memoire_chatbot/, __pycache__/, ...
├── agent_tools.py              # Agents et Outils : ReAct + tools + StateGraph (Groq)
├── agent_evaluateur_sav.py     # Agent SAV : guardrail + agent + juge (Groq)
├── db_init_boutique.py         # Génère boutique.db (users/orders) pour l'agent SAV
├── boutique.db                 # Base sqlite générée
├── graph.mmd                   # Visualisation mermaid du graphe de agent_tools.py
├── menu.txt                    # Source du RAG "menu de restaurant"
├── memoire_chatbot/            # Persistance Chroma (générée par 05_persistance_chroma.py)
├── data/                       # Sources documentaires pour les RAG et le pseudo-RAG SAV
│   ├── IA-E1.pdf, IA-E2-E3.md, IA-E4-E5.txt   # référentiel Simplon (RAG LlamaIndex)
│   └── politiques_sav.txt                     # politique de retour (lue par query_knowledge_base)
└── chatbots/                   # Exercices Chatbot → RAG, dans l'ordre du cours
    ├── 01_chatbot_stateless.py
    ├── 02_chatbot_stateful.py
    ├── 03_chatbot_langchain_lcel.py
    ├── 04_chatbot_memoire_strategies.py
    ├── 05_persistance_chroma.py
    ├── 06_persistance_pgvector.py
    ├── 07_rag_langchain_chroma.py
    └── 08_rag_llamaindex_faiss.py
```

## Installation

```
pip install langchain-core langchain-groq langchain-ollama langchain-chroma \
            langchain-community langchain-text-splitters langchain-postgres \
            langgraph llama-index llama-index-llms-ollama llama-index-embeddings-ollama \
            llama-index-vector-stores-faiss faiss-cpu ollama pydantic python-dotenv
```

Les scripts `chatbots/0*` nécessitent en plus [Ollama](https://ollama.com/) installé et lancé en local
(`ollama run llama3.2:3b`), sauf s'ils sont adaptés à Groq.

`boutique.db` et `memoire_chatbot/` sont générés localement (non versionnés, voir `.gitignore`) :
```
python db_init_boutique.py   # recrée boutique.db
```
`memoire_chatbot/` se recrée automatiquement au premier lancement de `chatbots/05_persistance_chroma.py`.

## Notes

- Windows/PowerShell : certains scripts impriment des caractères Unicode (✅, emojis) qui font planter
  la console par défaut (`cp1252`). Si besoin, ajouter `sys.stdout.reconfigure(encoding="utf-8")`
  en tête de script (déjà fait dans `agent_evaluateur_sav.py`).
- `chatbots/07_rag_langchain_chroma.py` et `08_rag_llamaindex_faiss.py` référencent leurs ressources
  via des chemins relatifs (`../menu.txt`, `../data`) : les lancer en étant positionné dans `chatbots/`.
