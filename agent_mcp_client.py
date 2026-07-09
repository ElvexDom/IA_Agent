import asyncio
import sys

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

SYSTEM_PROMPT = SystemMessage(
    content="""Tu es un assistant qui DOIT utiliser les outils fournis par le serveur MCP
pour tout calcul ou conversion. N'invente jamais un résultat toi-même."""
)


async def build_app():
    # Le client MCP démarre mcp_server.py comme sous-processus et communique
    # avec lui via stdio (protocole MCP), exactement comme un connecteur externe.
    client = MultiServerMCPClient(
        {
            "outils_demo": {
                "command": sys.executable,
                "args": ["mcp_server.py"],
                "transport": "stdio",
            }
        }
    )
    tools = await client.get_tools()
    print(f"Outils récupérés depuis le serveur MCP : {[t.name for t in tools]}")

    llm_with_tools = llm.bind_tools(tools)

    def node_agent(state: MessagesState):
        response = llm_with_tools.invoke([SYSTEM_PROMPT] + list(state["messages"]))
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", node_agent)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    return workflow.compile()


async def main():
    app = await build_app()
    query = (
        "Un client paie 250 USD pour un article HT. Calcule le prix TTC, "
        "puis convertis ce montant TTC en euros."
    )
    print(f"\nQUESTION : {query}")
    response = await app.ainvoke({"messages": [HumanMessage(content=query)]})
    print(f"RÉPONSE : {response['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(main())
