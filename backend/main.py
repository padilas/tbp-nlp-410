from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from rag import create_rag_chain, get_retriever

app = FastAPI()

# Konfigurasi CORS agar frontend React bisa memanggil API ini
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], # URL Vite default
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Definisi struktur data request dari frontend
class ChatMessage(BaseModel):
    role: str # "user" atau "bot"
    content: str

class ChatRequest(BaseModel):
    question: str
    history: List[ChatMessage] = []

@app.get("/")
def read_root():
    return {"message": "RAG Backend is running. Access /api/chat via POST."}

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    # 1. Format history menjadi string
    history_str = ""
    for msg in req.history:
        role_name = "Siswa" if msg.role == "user" else "Asisten"
        history_str += f"{role_name}: {msg.content}\n"

    # 2. Ambil dokumen relevan
    retriever = get_retriever()
    if not retriever:
        return {"answer": "Error: Database dokumen belum siap atau dokumen belum diproses (retriever gagal diload)."}

    docs = retriever.invoke(req.question)
    context = "\n\n".join([doc.page_content for doc in docs])

    # 3. Jalankan chain RAG
    chain = create_rag_chain()
    if not chain:
        return {"answer": "Error: Gagal memuat model AI atau RAG chain."}

    try:
        # Panggil AI (menunggu stream selesai dan kumpulkan sebagai string)
        # Catatan: Walaupun model diset streaming=True di rag.py, 
        # metode .invoke() akan mengumpulkan semua stream hingga selesai dan mengembalikan teks utuh.
        response = chain.invoke({
            "context": context,
            "chat_history": history_str,
            "question": req.question
        })
        return {"answer": response}
    except Exception as e:
        return {"answer": f"Terjadi kesalahan saat memproses jawaban: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
