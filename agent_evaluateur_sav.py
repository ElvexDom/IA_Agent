import sys
import sqlite3
from datetime import date, datetime
from typing import Literal

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)


# ── Outils ──
@tool
def get_order_info(user_id: int, order_id: int) -> str:
    """Récupère le produit, sa catégorie et la date d'une commande, pour un utilisateur et un ID de commande donnés."""
    conn = sqlite3.connect("boutique.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT product_name, category, order_date FROM orders WHERE id = ? AND user_id = ?",
        (order_id, user_id),
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        return f"Aucune commande {order_id} trouvée pour l'utilisateur {user_id}."
    product_name, category, order_date = row
    return f"Produit: {product_name}, Catégorie: {category}, Date de commande: {order_date}"


@tool
def jours_ecoules(date_commande: str) -> int:
    """Calcule le nombre de jours écoulés entre une date de commande (format YYYY-MM-DD) et aujourd'hui."""
    d = datetime.strptime(date_commande, "%Y-%m-%d").date()
    return (date.today() - d).days


@tool
def query_knowledge_base(query: str) -> str:
    """Renseigne la politique de retour produit par catégorie (pseudo-RAG : lecture du document interne)."""
    with open("data/politiques_sav.txt", encoding="utf-8") as f:
        return f.read()


tools = [get_order_info, jours_ecoules, query_knowledge_base]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)


# ── État du graphe ──
class EvaluationVerdict(BaseModel):
    grade: Literal["OUI", "NON"] = Field(description="Verdict final")
    critique: str = Field(description="Pourquoi la réponse est refusée")


class MyGraphState(MessagesState):
    verdict: EvaluationVerdict


# ── Agent principal ──
SYSTEM_PROMPT = SystemMessage(
    content="""Tu es un agent SAV chargé de statuer sur les demandes de retour produit.
Pour chaque demande tu DOIS, dans l'ordre, utiliser les outils :
1. get_order_info pour connaître le produit, sa catégorie et sa date de commande.
2. jours_ecoules pour connaître le nombre de jours écoulés depuis la commande.
3. query_knowledge_base pour connaître le délai de retour autorisé pour cette catégorie.
Puis conclus clairement si le retour est ACCEPTÉ ou REFUSÉ, en citant les chiffres exacts renvoyés par les outils.
N'invente jamais de données, base-toi uniquement sur les résultats des outils."""
)


def call_model(state: MyGraphState):
    messages = state["messages"]
    response = llm_with_tools.invoke([SYSTEM_PROMPT] + list(messages))
    return {"messages": [response]}


def should_continue(state: MyGraphState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "evaluator"


# ── Nœud d'évaluation (le Juge) ──
evaluator_llm = llm.with_structured_output(EvaluationVerdict)


def quality_control_node(state: MyGraphState):
    """Audite la réponse de l'agent en se basant uniquement sur les résultats des outils."""
    messages = state["messages"]
    last_agent_message = messages[-1].content
    audit_prompt = SystemMessage(
        content=f"""Tu es un auditeur qualité neutre. Ta seule source de vérité est
l'historique des messages (résultats des outils) qui te sera fourni après ce message.

ÉTAPES D'AUDIT :
1. Vérifie que l'agent a bien appelé get_order_info, jours_ecoules et query_knowledge_base.
2. Vérifie que le nombre de jours écoulés cité correspond à celui renvoyé par l'outil jours_ecoules.
3. Vérifie que le délai autorisé cité correspond à la politique de la catégorie renvoyée par query_knowledge_base.
4. Vérifie que la conclusion (ACCEPTÉ/REFUSÉ) est cohérente avec la comparaison jours écoulés vs délai autorisé.

VERDICT :
- Grade "OUI" seulement si toutes les étapes sont respectées et la conclusion est correcte.
- Grade "NON" si l'agent a sauté un outil, inventé un chiffre, ou si la conclusion est incohérente.

RÉPONSE DE L'AGENT À AUDITER :
"{last_agent_message}"
"""
    )
    verdict = evaluator_llm.invoke([audit_prompt] + list(messages))
    return {"verdict": verdict}


def route_after_eval(state: MyGraphState):
    verdict = state.get("verdict")
    if verdict and verdict.grade == "OUI":
        return END
    return "agent"


# ── Graphe ──
workflow = StateGraph(MyGraphState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("evaluator", quality_control_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")
workflow.add_conditional_edges("evaluator", route_after_eval)

app = workflow.compile()


if __name__ == "__main__":
    query = "Bonjour, je suis le client 1, je voudrais retourner ma commande 101, est-ce possible ?"
    inputs = {"messages": [HumanMessage(content=query)]}
    response = app.invoke(inputs)
    print(f"\nRÉPONSE FINALE : {response['messages'][-1].content}")
