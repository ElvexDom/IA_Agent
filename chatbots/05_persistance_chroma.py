from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- Config ---
model_name = "llama3.2:3b"
DB_PATH = "../memoire_chatbot"
WINDOW_SIZE = 5

# --- Modèle + Embeddings ---
llm = ChatOllama(model=model_name, temperature=0)
embeddings = OllamaEmbeddings(model=model_name)

# --- Vector Store (Chroma DB) ---
vectorstore = Chroma(
    collection_name="historique_global",
    embedding_function=embeddings,
    persist_directory=DB_PATH
)

# --- Prompt strict ---
prompt = ChatPromptTemplate.from_messages([
    ("system", """Tu es un assistant avec une mémoire persistante.

Voici ce que tu sais sur l'utilisateur grâce à tes archives :
{archives}

RÈGLES :
1. Tu AS une mémoire — ne dis JAMAIS le contraire.
2. Si une info est dans les archives, réponds directement sans hésiter.
3. N'utilise JAMAIS les phrases : "je ne connais pas", "chaque conversation est nouvelle", "je ne garde pas d'informations".
4. Sois naturel, comme un ami qui se souvient de ses conversations passées.
"""),
    MessagesPlaceholder(variable_name="session_history"),
    ("human", "{input}")
])

chain = prompt | llm

# --- Session en cours (Window Memory) ---
session_history = []

print("Chatbot démarré. Tape 'quit' pour quitter.\n")

while True:
    user_input = input("Vous : ").strip()
    if user_input.lower() == "quit":
        break

    # 1. Récupérer les souvenirs pertinents depuis Chroma
    docs_du_passe = vectorstore.similarity_search(user_input, k=3)
    context_archives = "\n".join([doc.page_content for doc in docs_du_passe])

    # 2. Injection explicite des archives dans le message
    if context_archives:
        enriched_input = f"{user_input}\n\n[INFO ARCHIVES : {context_archives}]"
    else:
        enriched_input = user_input

    # 3. Appel au LLM
    response = chain.invoke({
        "input": enriched_input,
        "session_history": session_history,
        "archives": context_archives
    })

    print(f"Assistant : {response.content}\n")

    # 4. Mise à jour Window Memory (court terme)
    session_history.append(HumanMessage(content=user_input))
    session_history.append(AIMessage(content=response.content))
    if len(session_history) > WINDOW_SIZE * 2:
        session_history = session_history[-WINDOW_SIZE * 2:]

    # 5. Sauvegarder l'échange dans Chroma (long terme)
    echange_formate = (
        f"L'utilisateur a demandé : {user_input} | "
        f"L'assistant a répondu : {response.content}"
    )
    vectorstore.add_texts(texts=[echange_formate])