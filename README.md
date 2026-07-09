# IA Agentic — LangChain / LlamaIndex

Travaux pratiques du support **DEV IA 2025 — IA Agentic (Antony Schutz)**.
Ce README suit l'avancement réel du dépôt par rapport au sommaire du PDF.

## Modèle utilisé

Les scripts du dossier `chatbots/` suivent le PDF tel quel (**Ollama** en local).
Tous les scripts à la racine (`agent_tools.py`, `agent_evaluateur_sav.py`, `agent_hitl_fichiers.py`,
`agent_checkpoint_timetravel.py`, `agent_multiagent_peer2peer.py`, `agent_multiagent_superviseur.py`,
`agent_mcp_client.py`) utilisent **Groq** (`ChatGroq`, modèle `openai/gpt-oss-120b`) à la place
d'Ollama, trop lourd sur la machine de dev.

Clé API attendue dans un fichier `.env` à la racine (non versionné, voir `.gitignore`).
Un modèle sans secret est fourni dans `.env.example` :
```
cp .env.example .env   # puis y coller sa vraie clé GROQ_API_KEY
```

## Avancement — procédure terminée (13/13 chapitres)

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
| Nœud de Protection — Guardrail (p.33) | ✅ | `agent_evaluateur_sav.py` (nœud `guardrail` en entrée du graphe) |
| Human-in-the-loop (p.38) | ✅ | `agent_hitl_fichiers.py` + dossier `workspace/` |
| CheckPoint / Time Travel (p.43) | ✅ | `agent_checkpoint_timetravel.py` |
| Multi-Agent & Orchestration (p.45) | ✅ | `agent_multiagent_peer2peer.py`, `agent_multiagent_superviseur.py` + `db_init_cinema.py` |
| Connecteur MCP (p.58) | ✅* | `mcp_server.py` + `agent_mcp_client.py` |

\* Le PDF s'arrête au titre de ce chapitre (page 58, non rédigé). Implémentation de ma propre initiative,
avec `langchain-mcp-adapters` : un serveur MCP (`mcp_server.py`, protocole stdio) expose des outils, et
un agent LangGraph (`agent_mcp_client.py`) les découvre et les utilise dynamiquement à l'exécution.

## Structure du dépôt

```
.
├── .env                             # GROQ_API_KEY (non versionné, voir .gitignore)
├── .env.example                     # Modèle de .env sans secret (versionné)
├── .gitignore                       # Exclut .env, *.db, memoire_chatbot/, .venv/, workspace généré, __pycache__/, ...
├── requirements.txt                 # Dépendances Python (pip install -r requirements.txt)
├── agent_tools.py                   # Agents et Outils : ReAct + tools + StateGraph (Groq)
├── agent_evaluateur_sav.py          # Agent SAV : guardrail + agent + juge (Groq)
├── db_init_boutique.py              # Génère boutique.db (users/orders) pour l'agent SAV
├── boutique.db                      # Base sqlite générée (non versionnée)
├── agent_hitl_fichiers.py           # Human-in-the-loop : agent de gestion de fichiers (Groq)
├── agent_checkpoint_timetravel.py   # CheckPoint / Time Travel : historique + fork sur agent_hitl_fichiers
├── workspace/                       # Fichiers de test pour les 2 scripts ci-dessus (auto-générés, voir Notes)
├── agent_multiagent_peer2peer.py    # Multi-Agent Peer-to-Peer : analyst → writer (Groq)
├── agent_multiagent_superviseur.py  # Multi-Agent Superviseur Central : manager + analyst + writer (Groq)
├── db_init_cinema.py                # Génère cinema.db (films/réalisateurs) pour les 2 agents ci-dessus
├── cinema.db                        # Base sqlite générée (non versionnée)
├── mcp_server.py                    # Connecteur MCP : serveur d'outils (protocole stdio)
├── agent_mcp_client.py              # Connecteur MCP : agent LangGraph qui consomme le serveur MCP (Groq)
├── graph.mmd                        # Visualisation mermaid du graphe de agent_tools.py
├── menu.txt                         # Source du RAG "menu de restaurant"
├── memoire_chatbot/                 # Persistance Chroma (générée par 05_persistance_chroma.py, non versionnée)
├── data/                            # Sources documentaires pour les RAG et le pseudo-RAG SAV
│   ├── IA-E1.pdf, IA-E2-E3.md, IA-E4-E5.txt   # référentiel Simplon (RAG LlamaIndex)
│   └── politiques_sav.txt                     # politique de retour (lue par query_knowledge_base)
└── chatbots/                        # Exercices Chatbot → RAG, dans l'ordre du cours
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
python -m venv .venv
.venv\Scripts\activate        # PowerShell : .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Les scripts `chatbots/0*` nécessitent en plus [Ollama](https://ollama.com/) installé et lancé en local
(`ollama run llama3.2:3b`), sauf s'ils sont adaptés à Groq.

`boutique.db`, `cinema.db` et `memoire_chatbot/` sont générés localement (non versionnés, voir `.gitignore`) :
```
python db_init_boutique.py   # recrée boutique.db
python db_init_cinema.py     # recrée cinema.db
```
`memoire_chatbot/` se recrée automatiquement au premier lancement de `chatbots/05_persistance_chroma.py`.

## Lancer chaque exercice

```
python agent_tools.py                        # Agents et Outils
python agent_evaluateur_sav.py                # Nœuds d'évaluation + Guardrail
python agent_hitl_fichiers.py                 # Human-in-the-loop
python agent_checkpoint_timetravel.py         # CheckPoint / Time Travel
python agent_multiagent_peer2peer.py          # Multi-Agent Peer-to-Peer
python agent_multiagent_superviseur.py        # Multi-Agent Superviseur Central
python agent_mcp_client.py                    # Connecteur MCP (démarre mcp_server.py automatiquement)
```

## Notes

- Windows/PowerShell : certains scripts impriment des caractères Unicode (✅, emojis) qui font planter
  la console par défaut (`cp1252`). Tous les scripts à la racine appellent
  `sys.stdout.reconfigure(encoding="utf-8")` en tête de fichier pour éviter ce plantage.
- `workspace/` (utilisé par `agent_hitl_fichiers.py` et `agent_checkpoint_timetravel.py`) est
  recréé automatiquement si besoin par `assurer_fichiers_de_test()` (dans `agent_hitl_fichiers.py`) :
  le dossier peut être vidé sans risque, il se repeuple tout seul au lancement suivant.
- `chatbots/07_rag_langchain_chroma.py` et `08_rag_llamaindex_faiss.py` référencent leurs ressources
  via des chemins relatifs (`../menu.txt`, `../data`) : les lancer en étant positionné dans `chatbots/`.
- `agent_mcp_client.py` démarre `mcp_server.py` comme sous-processus (protocole MCP en stdio) :
  aucune action manuelle requise, mais les deux fichiers doivent rester dans le même dossier.
