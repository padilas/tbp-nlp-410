from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from typing import List

from rag import create_rag_chain, get_retriever
from guardrails import is_safe_query

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    history: List[ChatMessage] = []

class AdminRetrieveRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"message": "RAG Backend is running. Access /api/chat via POST."}

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    # 0. Validasi keamanan pertanyaan (Guardrails)
    is_safe, warning_msg = is_safe_query(req.question)
    if not is_safe:
        return {"answer": warning_msg}

    # 1. Format history menjadi string
    history_str = ""
    for msg in req.history:
        role_name = "Siswa" if msg.role == "user" else "Asisten"
        history_str += f"{role_name}: {msg.content}\n"

    # 2. Ambil dokumen relevan
    retriever = get_retriever()
    if not retriever:
        return {"answer": "Error: Database dokumen belum siap atau dokumen belum diproses."}

    docs = retriever.invoke(req.question)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Ekstrak metadata source
    extracted_sources = []
    seen = set()
    for doc in docs:
        source_name = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", 0) + 1
        identifier = f"{source_name}_{page}"
        if identifier not in seen:
            seen.add(identifier)
            extracted_sources.append({"file": source_name, "page": page})

    # 3. Jalankan chain RAG
    chain = create_rag_chain()
    if not chain:
        return {"answer": "Error: Gagal memuat model AI atau RAG chain."}

    try:
        response = chain.invoke({
            "context": context,
            "chat_history": history_str,
            "question": req.question
        })
        return {"answer": response, "sources": extracted_sources}
    except Exception as e:
        return {"answer": f"Terjadi kesalahan saat memproses jawaban: {str(e)}"}

@app.get("/api/admin/config")
def admin_config():
    import rag
    return {
        "llm_provider": rag.LLM_PROVIDER,
        "openrouter_model": rag.OPENROUTER_MODEL,
        "ollama_model": rag.OLLAMA_MODEL,
        "chunk_size": rag.CHUNK_SIZE,
        "chunk_overlap": rag.CHUNK_OVERLAP,
        "embedding_model": rag.EMBEDDING_MODEL_NAME,
        "vector_weight": 0.8, # dari class HybridRetriever default
        "bm25_weight": 0.2
    }

@app.post("/api/admin/retrieve")
def admin_retrieve(req: AdminRetrieveRequest):
    retriever = get_retriever()
    if not retriever:
        raise HTTPException(status_code=500, detail="Database belum siap.")
    
    docs = retriever.invoke(req.query)
    results = []
    for rank, doc in enumerate(docs, 1):
        content = doc.page_content
        # bersihkan prefix 'passage: ' jika ada
        if content.startswith("passage: "):
            content = content[len("passage: "):]
            
        results.append({
            "rank": rank,
            "source": os.path.basename(doc.metadata.get("source", "Unknown")),
            "page": doc.metadata.get("page", 0) + 1,
            "content": content
        })
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
