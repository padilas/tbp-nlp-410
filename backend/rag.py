import os
import inspect

if not hasattr(inspect, "formatargspec"):
    def formatargspec(args, varargs=None, varkw=None, defaults=None,
                      kwonlyargs=None, kwonlydefaults=None, annotations={},
                      formatarg=str,
                      formatvarargs=lambda name: '*' + name,
                      formatvarkw=lambda name: '**' + name,
                      formatvalue=lambda value: '=' + repr(value),
                      formatreturns=lambda text: ' -> ' + text,
                      formatannotation=lambda annotation: str(annotation)):
        specs = []
        if args:
            for i, arg in enumerate(args):
                spec = formatarg(arg)
                if defaults and i >= len(args) - len(defaults):
                    spec += formatvalue(defaults[i - len(args) + len(defaults)])
                specs.append(spec)
        if varargs:
            specs.append(formatvarargs(varargs))
        if kwonlyargs:
            for arg in kwonlyargs:
                spec = formatarg(arg)
                if kwonlydefaults and arg in kwonlydefaults:
                    spec += formatvalue(kwonlydefaults[arg])
                specs.append(spec)
        if varkw:
            specs.append(formatvarkw(varkw))
        return '(' + ', '.join(specs) + ')'
    inspect.formatargspec = formatargspec

import pickle
from typing import List, Any
from pydantic import Field
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

# env
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(root_dir, ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
else:
    load_dotenv()

# db
DB_DIR = os.path.join(os.path.dirname(__file__), "vectorstore")
os.makedirs(DB_DIR, exist_ok=True)
CHUNKS_PKL_PATH = os.path.join(DB_DIR, "chunks.pkl")

# Config env var
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").lower()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434") 
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Chunk param, di set default ke 1000 dgn overlap 200
# Chunk paragraf untuk dokumen yg dipake di sini kurang cocok krn malah konteksnya ilang
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"

import re

def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    lines = text.split("\n")
    cleaned_lines = []
    current_line = ""
    
    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
        is_list_item = line_str.startswith(('●', '-', '*', '•', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '0.', 'a)', 'b)', 'c)', 'd)', 'e)', 'f)'))
        
        if is_list_item:
            if current_line:
                cleaned_lines.append(current_line.strip())
            current_line = line_str
        elif current_line and not current_line.endswith((".", "!", "?", ":")):
            current_line += " " + line_str
        else:
            if current_line:
                cleaned_lines.append(current_line.strip())
            current_line = line_str
            
    if current_line:
        cleaned_lines.append(current_line.strip())
    result = "\n".join(cleaned_lines)
    result = re.sub(r" +", " ", result)
    return result

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

# RRF Hybrid Retriever pakai FAISS untuk dense dan BM25 untuk sparse (keyword)
class HybridRetriever(BaseRetriever):
    vector_retriever: Any = Field(description="The FAISS vector store retriever")
    bm25_retriever: Any = Field(description="The BM25 keyword retriever")
    weight_vector: float = 0.8
    weight_bm25: float = 0.2
    top_k: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        vector_query = f"query: {query}"
        vector_docs = self.vector_retriever.invoke(vector_query)
        bm25_docs = self.bm25_retriever.invoke(query)

        # rank dari vector dan bm25, pakai rrf
        rrf_scores = {}

        def accumulate_rrf(docs, weight):
            for rank, doc in enumerate(docs):
                content_key = doc.page_content.strip()
                if content_key not in rrf_scores:
                    rrf_scores[content_key] = {"doc": doc, "score": 0.0}
                rrf_scores[content_key]["score"] += weight * (1.0 / (rank + 60))
                
        accumulate_rrf(vector_docs, self.weight_vector)
        accumulate_rrf(bm25_docs, self.weight_bm25)

        sorted_items = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
        retrieved_docs = [item["doc"] for item in sorted_items[:self.top_k]]
        cleaned_docs = []
        for doc in retrieved_docs:
            content = doc.page_content
            if content.startswith("passage: "):
                content = content[len("passage: "):]
            cleaned_docs.append(Document(page_content=content, metadata=doc.metadata))
            
        return cleaned_docs

def get_llm():
    if LLM_PROVIDER == "openrouter" and OPENROUTER_API_KEY:
        return ChatOpenAI(
            model=OPENROUTER_MODEL,
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.0,
            max_tokens=1024,
            streaming=True,
            default_headers={
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "Academic QA Bot"
            }
        )
    else:
        return ChatOllama(
            base_url=OLLAMA_HOST,
            model=OLLAMA_MODEL,
            temperature=0.0,
            streaming=True
        )

def process_docs(file_path: str) -> str:
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    else:
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()

    full_text = " ".join([doc.page_content for doc in docs]).lower()
    ai_keywords = ["kecerdasan buatan", "artificial intelligence", "machine learning", "deep learning", 
                   "neural network", "jaringan saraf", "algoritma", "data", "komputer", "teknologi", "robot", "ann", "jst"]
    match_count = sum(1 for kw in ai_keywords if kw in full_text)
    
    if match_count < 3:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
        raise ValueError("Dokumen yang diunggah tidak berkaitan dengan topik Kecerdasan Buatan.")
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    chunks = text_splitter.split_documents(docs)
    for chunk in chunks:
        chunk.page_content = f"passage: {chunk.page_content}"
    
    # update faiss db
    embeddings = get_embeddings()
    if os.path.exists(os.path.join(DB_DIR, "index.faiss")):
        vectorstore = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
        vectorstore.add_documents(chunks)
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(DB_DIR)
    
    # update chunks pkl (untuk bm25 atau retriever yg sparse)
    existing_chunks = []
    if os.path.exists(CHUNKS_PKL_PATH):
        try:
            with open(CHUNKS_PKL_PATH, "rb") as f:
                existing_chunks = pickle.load(f)
        except Exception:
            existing_chunks = []
            
    existing_chunks.extend(chunks)
    with open(CHUNKS_PKL_PATH, "wb") as f:
        pickle.dump(existing_chunks, f)
        
    return f"Successfully processed {len(chunks)} chunks from the document. \nSaved to FAISS and BM25 store."

def get_retriever():
    embeddings = get_embeddings()
    has_faiss = os.path.exists(os.path.join(DB_DIR, "index.faiss"))
    has_pkl = os.path.exists(CHUNKS_PKL_PATH)
    
    if has_faiss and has_pkl:
        try:
            vectorstore = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
            vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
            with open(CHUNKS_PKL_PATH, "rb") as f:
                chunks = pickle.load(f)

            bm25_retriever = BM25Retriever.from_documents(chunks)
            bm25_retriever.k = 10
            return HybridRetriever(
                vector_retriever=vector_retriever,
                bm25_retriever=bm25_retriever,
                top_k=5
            )
        except Exception as e:
            print(f"[Error loading retrievers: {e}]")
            return None
    return None

# prompt
ACADEMIC_PROMPT_TEMPLATE = """Role: Anda adalah Asisten Buku Modul AI SMP/SMA (AI Textbook Bot) yang bertindak sebagai ahli materi pembelajaran berdasarkan modul Kecerdasan Buatan (AI) yang diunggah. Tugas Anda adalah membantu siswa memahami konsep, materi bab, dan teori yang dibahas dalam dokumen dengan ramah, komunikatif, profesional, dan akurat.

Riwayat Percakapan Sebelumnya:
{chat_history}

Context (Konteks Modul):
{context}

Constraints (Batasan Ketat):
1. Jawablah pertanyaan HANYA berdasarkan informasi yang terdapat dalam Context (Konteks Modul) di atas.
2. Jika informasi untuk menjawab pertanyaan TIDAK ADA di dalam Context, jawablah dengan: "Maaf, informasi tersebut tidak ditemukan dalam modul pembelajaran yang diunggah. Silakan merujuk ke buku referensi utama atau tanyakan kepada Guru pengampu mata pelajaran." Jangan mencoba mengarang jawaban atau berhalusinasi.
3. Jangan pernah berasumsi atau memberikan teori/penjelasan di luar data yang disediakan di Context.
4. JANGAN PERNAH menyebutkan nomor dokumen (seperti '[Dokumen X]' atau '(Dokumen X)') ataupun nama file modul di dalam jawaban Anda kepada siswa. Jawablah secara langsung dan natural.

Contoh Tanya Jawab:
Contoh 1:
Pertanyaan: Apa definisi kecerdasan buatan menurut bab pendahuluan?
Context: [Dokumen 1]
Sumber: BUKU AI SMA Kelas 10 Semester 1.pdf (Halaman 5)
Isi: Kecerdasan Buatan (Artificial Intelligence) didefinisikan sebagai simulasi proses kecerdasan manusia oleh mesin, khususnya sistem komputer, yang mencakup pembelajaran, penalaran, dan koreksi diri.
Jawaban: Berdasarkan bab pendahuluan modul, Kecerdasan Buatan (AI) adalah simulasi proses kecerdasan manusia oleh mesin (terutama sistem komputer) yang mencakup kemampuan belajar, menalar, dan melakukan koreksi diri.

Contoh 2:
Pertanyaan: Bagaimana sejarah penemuan algoritma backpropagation?
Context: [Dokumen 1]
Sumber: BUKU AI SMA Kelas 11 Semester 2.pdf (Halaman 40)
Isi: Dokumen ini hanya menjelaskan struktur jaringan saraf tiruan (neural networks) dan fungsi aktivasi sigmoid.
Jawaban: Maaf, informasi mengenai sejarah penemuan algoritma backpropagation tidak ditemukan dalam modul pembelajaran yang diunggah. Silakan merujuk ke buku referensi utama atau tanyakan kepada Guru pengampu mata pelajaran.

Pertanyaan: {question}
Jawaban:"""

def create_rag_chain():
    retriever = get_retriever()
    if not retriever:
        return None
        
    prompt = PromptTemplate(
        template=ACADEMIC_PROMPT_TEMPLATE,
        input_variables=["context", "chat_history", "question"]
    )
    
    llm = get_llm()
    return prompt | llm | StrOutputParser()