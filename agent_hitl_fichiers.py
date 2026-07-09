import os
import sys
from typing import Literal

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

WORKSPACE_DIR = "workspace"

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)


# ── Outils ──
@tool
def list_files() -> str:
    """Liste les fichiers présents dans le dossier de travail."""
    files = os.listdir(WORKSPACE_DIR)
    if not files:
        return "Le dossier est vide."
    return "Fichiers présents : " + ", ".join(files)


@tool
def delete_file(filename: str) -> str:
    """Supprime définitivement un fichier du dossier de travail. Action irréversible."""
    path = os.path.join(WORKSPACE_DIR, filename)
    if not os.path.exists(path):
        return f"Le fichier {filename} n'existe pas."
    os.remove(path)
    return f"Le fichier {filename} a été supprimé définitivement."


tools = [list_files, delete_file]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)


# ── Agent ──
SYSTEM_PROMPT = SystemMessage(
    content="""Tu es un agent de gestion de fichiers.
Tu peux lister les fichiers (list_files) ou en supprimer un (delete_file).
Utilise toujours les outils, n'invente jamais de nom de fichier."""
)


def agent_node(state: MessagesState):
    messages = state["messages"]
    response = llm_with_tools.invoke([SYSTEM_PROMPT] + list(messages))
    return {"messages": [response]}


def route_after_agent(state: MessagesState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "__end__"


# ── Graphe avec point de pause avant les outils ──
workflow = StateGraph(MessagesState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", route_after_agent)
workflow.add_edge("tools", "agent")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory, interrupt_before=["tools"])


# ── Gestion temporelle du flux (le Runner) ──
def run_interaction(user_prompt: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    print(f"\n--- UTILISATEUR : {user_prompt} ---")

    for _ in app.stream({"messages": [HumanMessage(content=user_prompt)]}, config, stream_mode="values"):
        pass

    state_info = app.get_state(config)
    while state_info.next:
        last_message = state_info.values["messages"][-1]
        tool_call = last_message.tool_calls[0]
        tool_name = tool_call["name"]

        if tool_name == "delete_file":
            filename = tool_call["args"].get("filename", "?")
            print(f"[HITL DÉCLENCHÉ] L'agent veut exécuter delete_file sur : {filename}")
            decision = input("Confirmer la suppression ? (oui/non) : ")

            if decision.strip().lower() == "oui":
                print("Accord donné. Reprise du graphe (le fichier va être réellement supprimé)...")
                for _ in app.stream(None, config, stream_mode="values"):
                    pass
            else:
                print("Suppression refusée par l'administrateur.")
                # as_node="tools" : on fait croire au graphe que le nœud "tools" a produit
                # ce message, sans jamais exécuter réellement delete_file.
                app.update_state(
                    config,
                    {
                        "messages": [
                            ToolMessage(
                                content="La suppression a été refusée par l'administrateur, votre fichier est en sécurité.",
                                tool_call_id=tool_call["id"],
                            )
                        ]
                    },
                    as_node="tools",
                )
                for _ in app.stream(None, config, stream_mode="values"):
                    pass
        else:
            print("Lecture seule. Auto-validation...")
            for _ in app.stream(None, config, stream_mode="values"):
                pass

        state_info = app.get_state(config)

    final_message = app.get_state(config).values["messages"][-1]
    print(f"RÉPONSE FINALE : {final_message.content}")


def assurer_fichiers_de_test():
    """Recrée les fichiers de démo si absents, pour que le script reste rejouable."""
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    fixtures = {
        "notes.txt": "Ceci est une note de test.\n",
        "brouillon.txt": "Brouillon a supprimer.\n",
    }
    for filename, content in fixtures.items():
        path = os.path.join(WORKSPACE_DIR, filename)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)


if __name__ == "__main__":
    assurer_fichiers_de_test()
    run_interaction("Liste les fichiers du dossier de travail.", thread_id="session_liste")
    run_interaction("Supprime le fichier brouillon.txt", thread_id="session_suppression")
