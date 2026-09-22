import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate import WeaviateVectorStore
import weaviate

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large:latest")
COLLECTION_NAME = os.getenv("WEAVIATE_COLLECTION_NAME", "AgenticDocuments")

# Connect to Weaviate in Embedded mode natively via Python
client = weaviate.connect_to_embedded()

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

docs_to_insert = []
docs_folder = "./docs"

print("--- READING DOCUMENTS FOLDER ---")

for file_name in os.listdir(docs_folder):
  if file_name.endswith(".pdf"):
    pdf_path = os.path.join(docs_folder, file_name)
    print(f"Processing: {file_name}...")

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    for idx, page in enumerate(pages):
      page.metadata["source_file"] = file_name
      page.metadata["page_number"] = idx + 1

    docs_to_insert.extend(pages)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs_to_insert)

print(
    f"--- UPLOADING {len(chunks)} CHUNKS TO WEAVIATE (EMBEDDED + OLLAMA) IN SAFE BATCHES ---"
)

vector_store = WeaviateVectorStore(
    client=client,
    index_name=COLLECTION_NAME,
    text_key="text",
    embedding=embeddings,
)

# 1. Reducimos el tamaño del lote a 25 para evitar que Ollama colapse en el endpoint de tokenización
BATCH_SIZE = 25

try:
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        print(f"Uploading batch {i} to {min(i + BATCH_SIZE, len(chunks))}...")
        
        vector_store.add_documents(batch)
        
        # Pequeña pausa para permitir que el socket local de Ollama respire
        time.sleep(0.5)
        
    print("Ingestion completed successfully!")
    
except Exception as e:
    print(f"An error occurred during chunk insertion: {e}")
    raise e

finally:
    client.close()