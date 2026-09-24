import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_classic.retrievers import EnsembleRetriever

from huggingface_hub import InferenceClient


# CONFIGURATION

load_dotenv()

DATA_PATH = Path("data")
CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
LLM_MODEL = "Qwen/Qwen3-8B"
HF_TOKEN = os.getenv("HF_TOKEN")

# 1. LOAD DOCUMENTS

print("\n1. Loading PDF documents...") 

documents = [] 

pdf_files = list(DATA_PATH.glob("*.pdf")) 

if not pdf_files: 
    raise FileNotFoundError( 
        f"No PDF files were found in the folder: {DATA_PATH}" 
    ) 

for pdf_file in pdf_files: 
    print(f"Loading: {pdf_file.name}") 
    loader = PyPDFLoader(str(pdf_file)) 
    pdf_documents = loader.load() 
    documents.extend(pdf_documents) 
    print(f" Loaded {len(pdf_documents)} pages") 
    print("\nPDF loading completed.") 
    print(f"Total PDFs: {len(pdf_files)}") 
    print(f"Total pages: {len(documents)}")


# 2. SPLIT DOCUMENTS INTO CHUNKS

print("\n2. Splitting document into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

docs_processed = text_splitter.split_documents(documents)
print(f"Created {len(docs_processed)} chunks")


# 3. BM25 RETRIEVER

print("\n3. Creating BM25 retriever...")
bm25_retriever = BM25Retriever.from_documents(
    docs_processed
)

bm25_retriever.k = 5


# 4. EMBEDDINGS

print("\n4. Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)


# 5. CHROMA VECTOR STORE

print("\n5. Creating/loading Chroma vector store...")

if Path(CHROMA_PATH).exists():

    print("Existing Chroma database found.")
    print("Loading existing vector store...")

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

else:

    print("No Chroma database found.")
    print("Creating embeddings for document chunks...")
    print("This may take some time on the first run.")

    vectorstore = Chroma.from_documents(
        documents=docs_processed,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print("Chroma database created.")


vector_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 8}
)


# 6. HYBRID RETRIEVER

print("\n6. Creating hybrid retriever...")

hybrid_retriever = EnsembleRetriever(
    retrievers=[
        bm25_retriever,
        vector_retriever
    ],
    weights=[
        0.4,
        0.6
    ]
)


# 7. HUGGING FACE CLIENT

print("\n7. Creating Hugging Face client...", flush=True)

client = InferenceClient(
    api_key=HF_TOKEN,
    timeout=60
)

print("Hugging Face client created.", flush=True)

# User question loop
while True:

    # 1. Ask the user for a question
    query = input("\nEnter your question (or 'exit'): ")

    if query.lower() == "exit":
        break

    # 2. Retrieve relevant documents
    print("\nRetrieving relevant documents...", flush=True)

    retrieved_docs = hybrid_retriever.invoke(query)

    print(f"Retrieved {len(retrieved_docs)} documents", flush=True)

    # 3. Build context
    context_parts = []

    for i, doc in enumerate(retrieved_docs, start=1):

        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")

        if isinstance(page, int):
            page = page + 1

        context_parts.append(
            f"[Source {i} | File: {source} | Page: {page}]\n"
            f"{doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    # 4. Build the prompt
    system_prompt = """
You are a question-answering assistant.

Answer the user's question using ONLY the provided context.

Rules:
1. Do not use outside knowledge.
2. Do not invent or assume information.
3. If the context does not contain enough information to answer,
   say: "I don't know based on the documents provided."
4. Keep the answer concise and directly answer the question.
5. Cite the relevant source number and page when available.
"""

    context_message = f"""
Context:

{context}

Question:

{query}
"""

    # 5. Send request to Hugging Face
    print("\nSending request to Hugging Face...", flush=True)

    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": context_message
            }
        ],
        max_tokens=1024,
        temperature=0.1
    )

    print("Response received.", flush=True)

    # 6. Get answer
    answer = completion.choices[0].message.content

    # 7. Display answer
    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
        
    print(answer)


    # 8. Display sources
    print("\n" + "=" * 60)
    print("RETRIEVED SOURCES")
    print("=" * 60)

    for i, doc in enumerate(retrieved_docs, start=1):

        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")

        if isinstance(page, int):
            page = page + 1

        print(f"\nSource {i}")
        print(f"File: {source}")
        print(f"Page: {page}")

        print(
            "Content: "
            + doc.page_content[:300].replace("\n", " ")
        )

 