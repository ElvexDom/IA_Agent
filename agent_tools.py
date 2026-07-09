import sys

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
import urllib.request
import urllib.parse
import json

load_dotenv()

# ── LLM ──
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

# ── Outils ──
@tool
def calculateur_tva(prix_ht: float) -> float:
    """Calcule le prix TTC à partir du prix HT en ajoutant une TVA de 20%.
    Utile pour les calculs financiers précis."""
    return prix_ht * 1.20

@tool
def recherche_web(query: str) -> str:
    """Recherche des informations en temps réel sur internet.
    Utile pour trouver des prix, des actualités, des données récentes."""
    try:
        query_encodee = urllib.parse.quote(query)
        url = f"https://fr.wikipedia.org/w/api.php?action=query&list=search&srsearch={query_encodee}&format=json&utf8=1"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read())
            resultats = data["query"]["search"]
            if resultats:
                return resultats[0]["snippet"]
            return "Aucun résultat trouvé sur Wikipedia pour cette requête."
    except Exception as e:
        return f"Erreur lors de la recherche : {str(e)}. Impossible de trouver l'information."

# ── Tools greffés au LLM ──
tools = [recherche_web, calculateur_tva]
llm_with_tools = llm.bind_tools(tools)

# ── Agent ──
def node_agent(state: MessagesState):
    """Le cerveau de l'agent — reçoit l'historique et décide quoi faire."""
    response = llm_with_tools.invoke(state["messages"])
    print(f"DEBUG tool_calls: {response.tool_calls}")
    return {"messages": [response]}

# ── Graphe ──
workflow = StateGraph(MessagesState)
workflow.add_node("agent", node_agent)
workflow.add_node("tools", ToolNode(tools))
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")
agent_app = workflow.compile()

# ── Test ──
system_prompt = SystemMessage(content="""Tu es un assistant qui DOIT obligatoirement utiliser des outils.
RÈGLES ABSOLUES :
- Tu DOIS appeler recherche_web pour toute information factuelle
- Tu DOIS appeler calculateur_tva pour tout calcul de prix
- Il est INTERDIT de répondre sans utiliser au moins un outil
- Il est INTERDIT d'inventer des chiffres""")

query = "Quel est le prix actuel de l'action NVIDIA en temps réel ? Et si j'en achète 10, combien ça fait TTC ?"

inputs = {"messages": [system_prompt, HumanMessage(content=query)]}
response = agent_app.invoke(inputs)
print(f"RÉPONSE : {response['messages'][-1].content}")

# ── Visualisation ──
mermaid_code = agent_app.get_graph().draw_mermaid()
with open("graph.mmd", "w", encoding="utf-8") as f:
    f.write(mermaid_code)
print("Graphe sauvegardé dans graph.mmd ✓")
print("\n", mermaid_code)