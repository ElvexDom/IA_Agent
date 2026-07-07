from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. Ingestion ---
loader = TextLoader("../menu.txt", encoding="utf-8")
documents = loader.load()

# --- 2. Chunking ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"✅ {len(chunks)} chunks créés")

# --- 3. Embedding + Vector Store ---
embeddings = OllamaEmbeddings(model="llama3.2:3b")
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

# --- 4. Retriever ---
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- 5. Prompt ---
template = """Tu es un serveur virtuel sympathique et professionnel du restaurant "Le Glitch Gourmand".
Réponds à la question du client en utilisant uniquement les informations du menu ci-dessous.
Si la réponse n'est pas dans le menu, réponds poliment que tu ne sais pas ou que ce n'est pas disponible.
Ne jamais inventer un plat ou un prix qui n'existe pas dans le menu.

Menu (Contexte) :
{context}

Question du client : {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# --- LLM ---
llm = ChatOllama(model="llama3.2:3b", temperature=0)

# --- Formatage ---
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

# --- Pipeline RAG ---
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- Chatbot interactif ---
print("\n🍽️  Bienvenue au Glitch Gourmand ! Comment puis-je vous aider ?")
print("(tapez 'quit' pour partir)\n")

while True:
    question = input("Client : ").strip()
    if question.lower() == "quit":
        print("Serveur : Merci de votre visite, à bientôt ! 👋")
        break
    if not question:
        continue

    response = rag_chain.invoke(question)
    print(f"Serveur : {response}\n")