from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- Initialisation du modèle et du prompt ---
llm = ChatOllama(model="llama3.2:3b", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Tu es un assistant technique pro. Réponds de manière concise."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# --- Chaîne LCEL ---
chain = prompt | llm

def chatbot_langchain():
    print("🤖 Chatbot LangChain démarré. (Tape 'exit', 'stop' ou 'fin' pour quitter)\n")
    
    history = []  # Liste d'objets HumanMessage / AIMessage
    
    while True:
        user_input = input("\nClient : ")
        
        # Condition d'arrêt
        if user_input.lower().strip() in ["exit", "stop", "fin"]:
            print("🤖 Bot : À bientôt !")
            break
        
        # Appel à la chaîne avec l'historique
        response = chain.invoke({
            "input": user_input,
            "chat_history": history
        })
        
        bot_reply = response.content
        
        # Empilement dans l'historique avec les objets typés
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=bot_reply))
        
        print(f"🤖 Bot : {bot_reply}")

if __name__ == "__main__":
    chatbot_langchain()