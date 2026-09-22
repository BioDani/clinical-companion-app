import os
import warnings

# 1. Suppress noisy Weaviate logs, telemetry, and Python socket warnings (Note: must be uppercase "ERROR")
os.environ["WEAVIATE_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore", category=ResourceWarning)

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_weaviate import WeaviateVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Any
import weaviate

load_dotenv()

# Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large:latest")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma4:e2b")
COLLECTION_NAME = os.getenv("WEAVIATE_COLLECTION_NAME", "AgenticDocuments")

# Define State for LangGraph
class State(TypedDict):
    question: str
    documents: List[Any]
    answer: str

# Initialize Weaviate and Ollama components
client = weaviate.connect_to_embedded()
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
llm = ChatOllama(model=LLM_MODEL)

# Initialize LangChain Vector Store wrapper
vector_store = WeaviateVectorStore(
    client=client,
    index_name=COLLECTION_NAME,
    text_key="text",
    embedding=embeddings,
)

# Setup retriever (fetches top 3 relevant chunks)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Define Nodes for LangGraph workflow
def retrieve_node(state: State):
    question = state["question"]
    retrieved_docs = retriever.invoke(question)
    return {"documents": retrieved_docs}

def generate_node(state: State):
    question = state["question"]
    docs = state["documents"]
    
    context_texts = []
    for doc in docs:
        source = doc.metadata.get("source_file", "Unknown file")
        page = doc.metadata.get("page_number", "Unknown page")
        context_texts.append(f"[Source: {source}, Page: {page}]\n{doc.page_content}")
    
    formatted_context = "\n\n---\n\n".join(context_texts)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant specialized in clinical and project documentation. Answer the question accurately using ONLY the provided context below. Always mention the source files and page numbers used to form your answer.\n\nContext:\n{context}"),
        ("human", "{question}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": formatted_context, "question": question})
    
    return {"answer": answer}

# Build LangGraph Workflow
workflow = StateGraph(State)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# Interactive Continuous Chat Loop
if __name__ == "__main__":
    try:
        print("\n=== CLINICAL COMPANION RAG AGENT ===")
        print("Type 'exit' or 'quit' to end the conversation.\n")
        
        while True:
            user_question = input("You: ").strip()
            
            if user_question.lower() in ["exit", "quit"]:
                print("\nGoodbye!")
                break
                
            if not user_question:
                continue
                
            inputs = {"question": user_question}
            output = app.invoke(inputs)
            
            print(f"\nAssistant:\n{output['answer']}\n")
            
            print("--- Sources Used ---")
            sources = set()
            for doc in output.get("documents", []):
                src = doc.metadata.get("source_file", "N/A")
                pg = doc.metadata.get("page_number", "N/A")
                sources.add(f"- File: {src} (Page {pg})")
            
            for source in sources:
                print(source)
            print("-" * 50 + "\n")
            
    finally:
        client.close()