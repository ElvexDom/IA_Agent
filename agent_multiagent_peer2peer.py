import difflib
import operator
import sqlite3
import sys
from typing import Annotated, Sequence

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)


# ── Outil de l'analyste ──
@tool
def rechercher_films_par_realisateur(nom_realisateur: str) -> str:
    """Recherche en base les films d'un réalisateur. Tolère les fautes de frappe et les noms
    partiels (juste le nom de famille, sans le prénom)."""
    conn = sqlite3.connect("cinema.db")
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT director FROM movies")
    directors = [row[0] for row in cursor.fetchall()]

    # 1. Correspondance partielle (ex: "Nolan" contenu dans "Christopher Nolan")
    matches = [d for d in directors if nom_realisateur.lower() in d.lower()]

    # 2. Sinon, tolérance aux fautes de frappe (ex: "Christofer Nolen")
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


# ── État partagé Peer-to-Peer ──
class AgentState(TypedDict):
    """L'état partagé par nos agents Peer-to-Peer."""
    messages: Annotated[Sequence[BaseMessage], operator.add]


# ── Agent 1 : l'analyste ──
ANALYST_PROMPT = SystemMessage(
    content="""Tu es un analyste cinéma. Ta seule tâche est de trouver, via l'outil
rechercher_films_par_realisateur, la liste des films demandés par l'utilisateur.
Utilise toujours l'outil, n'invente jamais de films ni de dates."""
)


def analyst_node(state: AgentState):
    messages = state["messages"]
    response = analyst_llm.invoke([ANALYST_PROMPT] + list(messages))
    return {"messages": [response]}


def should_continue_analyst(state: AgentState):
    """Routeur conditionnel pour la micro-boucle de l'analyste."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "analyst_tools"
    return "writer"


# ── Agent 2 : le rédacteur ──
WRITER_PROMPT = SystemMessage(
    content="""Tu es un community manager. À partir de la liste de films fournie par
l'analyste, rédige un tweet accrocheur. Tu DOIS impérativement citer TOUS les films de
la liste dans ta réponse, et ajouter quelques hashtags pertinents."""
)


def writer_node(state: AgentState):
    messages = state["messages"]
    # En Peer-to-Peer : le message juste avant est forcément celui de l'analyste.
    user_msg = [m for m in messages if isinstance(m, HumanMessage)][0]
    analyst_output = messages[-1].content
    prompt = [
        WRITER_PROMPT,
        HumanMessage(
            content=f"Demande initiale : {user_msg.content}\nFilms trouvés par l'analyste : {analyst_output}"
        ),
    ]
    response = llm.invoke(prompt)
    return {"messages": [response]}


# ── Assemblage du graph ──
workflow = StateGraph(AgentState)
workflow.add_node("analyst", analyst_node)
workflow.add_node("analyst_tools", ToolNode(analyst_tools))
workflow.add_node("writer", writer_node)

workflow.add_edge(START, "analyst")
workflow.add_conditional_edges(
    "analyst",
    should_continue_analyst,
    {"analyst_tools": "analyst_tools", "writer": "writer"},
)
workflow.add_edge("analyst_tools", "analyst")
workflow.add_edge("writer", END)

app = workflow.compile()


if __name__ == "__main__":
    queries = [
        "Fais-moi un tweet sur les films de Christopher Nolan.",
        "Fais un tweet sur les films de Christofer Nolen.",  # faute de frappe volontaire
        "Fais un tweet sur les films de Tarantino.",  # nom de famille seul
    ]
    for query in queries:
        print(f"\n{'=' * 60}\nDEMANDE : {query}")
        response = app.invoke({"messages": [HumanMessage(content=query)]})
        print(f"TWEET : {response['messages'][-1].content}")
