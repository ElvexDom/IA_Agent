import difflib
import json
import operator
import sqlite3
import sys
from typing import Annotated, Literal, Sequence

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)


# ── Outil de l'analyste (identique au peer-to-peer) ──
@tool
def rechercher_films_par_realisateur(nom_realisateur: str) -> str:
    """Recherche en base les films d'un réalisateur. Tolère les fautes de frappe et les noms
    partiels (juste le nom de famille, sans le prénom)."""
    conn = sqlite3.connect("cinema.db")
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT director FROM movies")
    directors = [row[0] for row in cursor.fetchall()]

    matches = [d for d in directors if nom_realisateur.lower() in d.lower()]
    if not matches:
        matches = difflib.get_close_matches(nom_realisateur, directors, n=1, cutoff=0.6)
    if not matches:
        conn.close()
        return f"Aucun réalisateur trouvé pour '{nom_realisateur}'."

    director = matches[0]
    cursor.execute("SELECT title, year FROM movies WHERE director = ? ORDER BY year", (director,))
    films = cursor.fetchall()
    conn.close()
    films_str = ", ".join(f"{title} ({year})" for title, year in films)
    return f"Réalisateur trouvé : {director}. Films : {films_str}"


analyst_tools = [rechercher_films_par_realisateur]
analyst_llm = llm.bind_tools(analyst_tools)


# ── État partagé ──
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# ── Le Manager (superviseur) ──
SUPERVISOR_PROMPT = """Tu es le chef d'orchestre d'une équipe de 2 agents : Analyst (cherche
les films d'un réalisateur en base) et Writer (rédige le tweet final une fois les films connus).

RÈGLES :
- Si aucun film n'a encore été trouvé pour la demande du client, réponds {{"next": "Analyst"}}.
- Si les films ont été trouvés mais qu'aucun tweet n'a encore été rédigé, réponds {{"next": "Writer"}}.
- Si un tweet final citant tous les films a déjà été rédigé, réponds {{"next": "FINISH"}}.
Réponds UNIQUEMENT avec un objet JSON, rien d'autre.

Historique de l'équipe :
{history}
"""


def supervisor_node(state: AgentState):
    messages = state["messages"]

    # 🧠 RECONSTRUCTION PROPRE DE L'HISTORIQUE (Anti-collision de tokens)
    history_lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            history_lines.append(f"- CLIENT (Demande initiale) : {m.content}")
        elif isinstance(m, AIMessage):
            content = (m.content or "").strip()
            if '"next"' in content:
                history_lines.append(f"- SUPERVISEUR (Ordre précédent) : {content}")
            elif "#" in content:
                history_lines.append(f"- WRITER (Tweet final rédigé) : {content}")
            elif content:
                history_lines.append(f"- ANALYST (Films extraits) : {content}")
    history_text = "\n".join(history_lines)

    response = llm.invoke([SystemMessage(content=SUPERVISOR_PROMPT.format(history=history_text))])
    return {"messages": [response]}


def routing_supervisor(state: AgentState) -> Literal["Analyst", "Writer", "FINISH"]:
    """Aiguillage conditionnel dynamique basé sur la décision du Manager."""
    last_message = state["messages"][-1].content.strip()
    start_idx = last_message.find("{")
    end_idx = last_message.rfind("}")
    if start_idx != -1 and end_idx != -1:
        try:
            decision = json.loads(last_message[start_idx:end_idx + 1])
            destination = decision.get("next", "FINISH")
            if destination in ["Analyst", "Writer"]:
                return destination
        except Exception:
            pass
    return "FINISH"


# ── Agent 1 : l'analyste ──
ANALYST_PROMPT = SystemMessage(
    content="""Tu es un analyste cinéma. Ta seule tâche est de trouver, via l'outil
rechercher_films_par_realisateur, la liste des films demandés par l'utilisateur.
Utilise toujours l'outil, n'invente jamais de films ni de dates."""
)


def analyst_node(state: AgentState):
    messages = state["messages"]
    # En Superviseur : on cache au raisonnement de l'analyste les ordres JSON du manager.
    clean_history = [
        m for m in messages
        if not (isinstance(m, AIMessage) and '"next"' in (m.content or ""))
    ]
    response = analyst_llm.invoke([ANALYST_PROMPT] + clean_history)
    return {"messages": [response]}


def should_continue_analyst(state: AgentState):
    """En Superviseur : l'analyste ne décide jamais de la suite, il retourne au manager."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "analyst_tools"
    return "manager"


# ── Agent 2 : le rédacteur ──
WRITER_PROMPT = SystemMessage(
    content="""Tu es un community manager. À partir de la liste de films trouvée par
l'analyste, rédige un tweet accrocheur. Tu DOIS impérativement citer TOUS les films de
la liste dans ta réponse, et ajouter quelques hashtags pertinents."""
)


def writer_node(state: AgentState):
    messages = state["messages"]
    user_msg = [m for m in messages if isinstance(m, HumanMessage)][0]

    # Le message juste avant est celui du manager ({"next": "Writer"}), on scanne donc
    # l'historique à l'envers pour sauter le manager et retrouver le vrai rapport de l'analyste.
    raw_data = ""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content and '"next"' not in m.content:
            raw_data = m.content
            break

    prompt = [
        WRITER_PROMPT,
        HumanMessage(content=f"Demande initiale : {user_msg.content}\nFilms trouvés : {raw_data}"),
    ]
    response = llm.invoke(prompt)
    return {"messages": [response]}


# ── Assemblage du graph ──
workflow = StateGraph(AgentState)
workflow.add_node("manager", supervisor_node)
workflow.add_node("analyst", analyst_node)
workflow.add_node("analyst_tools", ToolNode(analyst_tools))
workflow.add_node("writer", writer_node)

workflow.add_edge(START, "manager")
workflow.add_conditional_edges(
    "manager",
    routing_supervisor,
    {"Analyst": "analyst", "Writer": "writer", "FINISH": END},
)
workflow.add_conditional_edges(
    "analyst",
    should_continue_analyst,
    {"analyst_tools": "analyst_tools", "manager": "manager"},
)
workflow.add_edge("analyst_tools", "analyst")
workflow.add_edge("writer", "manager")

app = workflow.compile()


def extraire_tweet_final(messages) -> str:
    """Le dernier message est l'accusé FINISH du manager : on cherche le vrai tweet en amont."""
    for m in reversed(messages):
        content = (getattr(m, "content", "") or "").strip()
        if isinstance(m, AIMessage) and content and '"next"' not in content:
            return content
    return "(aucun tweet trouvé)"


if __name__ == "__main__":
    queries = [
        "Fais-moi un tweet sur les films de Martin Scorsese.",
        "Fais un tweet sur les films de Tarantina.",  # faute de frappe volontaire
    ]
    for query in queries:
        print(f"\n{'=' * 60}\nDEMANDE : {query}")
        response = app.invoke({"messages": [HumanMessage(content=query)]}, {"recursion_limit": 25})
        print(f"TWEET : {extraire_tweet_final(response['messages'])}")
