from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
import faiss

# Modèle
Settings.llm = Ollama(model="llama3.2:3b", request_timeout=300)
Settings.embed_model = OllamaEmbedding(model_name="qwen3-embedding:0.6b")

# Chargement des documents
print("Chargement des documents...")
documents = SimpleDirectoryReader("../data").load_data()
print(f"{len(documents)} documents chargés ✓")

# FAISS
d = 1024  # dimension pour llama3.2:3b
faiss_index = faiss.IndexFlatL2(d)
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Index
print("Création de l'index (peut être long)...")
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
print("Index créé ✓")

# Moteur de recherche
query_engine = index.as_query_engine()

# Question
print("Prêt ! Lancement de la requête...")
response = query_engine.query("Quelles sont les compétences requises pour le développeur IA ?")
print(response)