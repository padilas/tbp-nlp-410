# Academic QA Bot - RAG Assistant (Materi AI SMP/SMA)

Proyek ini adalah ekosistem aplikasi asisten tanya jawab akademik (*Academic QA Bot*) berbasis *Retrieval-Augmented Generation* (RAG) untuk membantu siswa tingkat SMP/SMA mempelajari materi Kecerdasan Buatan (Artificial Intelligence).

---

## 📂 Struktur Direktori Proyek

*   **`backend/`**: Folder server web berbasis FastAPI yang menyediakan layanan RAG dan deteksi keamanan.
    *   **[main.py](file:///D:/Collage/6th%20Term/NLP/tbp-004-010/tbp-nlp-410/backend/main.py)**: Entry point API server FastAPI.
    *   **[rag.py](file:///D:/Collage/6th%20Term/NLP/tbp-004-010/tbp-nlp-410/backend/rag.py)**: Orkestrasi RAG (FAISS + BM25 Hybrid Retriever, Text Splitter, HuggingFace embeddings, dan LangChain pipeline).
    *   **[guardrails.py](file:///D:/Collage/6th%20Term/NLP/tbp-004-010/tbp-nlp-410/backend/guardrails.py)**: Modul filter keamanan query siswa (Prompt Injection, kasar, SARA, kecurangan akademis).
    *   **[ingest.py](file:///D:/Collage/6th%20Term/NLP/tbp-004-010/tbp-nlp-410/backend/ingest.py)**: Script offline untuk memproses buku modul PDF ke Vector Database FAISS.
    *   **[evaluator.py](file:///D:/Collage/6th%20Term/NLP/tbp-004-010/tbp-nlp-410/backend/evaluator.py)**: Script evaluasi kuantitatif otomatis (ROUGE-L, BERTScore, Hit Rate, MRR, Faithfulness).
    *   **[ragas_evaluator.py](file:///D:/Collage/6th%20Term/NLP/tbp-004-010/tbp-nlp-410/backend/ragas_evaluator.py)**: Script evaluasi kualitatif otomatis (Faithfulness, Answeer Relevancy, Context Recall & Precision).
    *   **[test_retrieval.py](file:///D:/Collage/6th%20Term/NLP/tbp-004-010/tbp-nlp-410/backend/test_retrieval.py)**: Script terminal interaktif untuk menguji relevansi retrieval.
*   **`frontend/`**: Aplikasi client antarmuka chatting interaktif berbasis React + Vite (TypeScript + Vanilla CSS).
    *   **[ChatInterface.tsx](file:///D:/Collage/6th%20Term/NLP/tbp-004-010/tbp-nlp-410/frontend/src/components/ChatInterface.tsx)**: Komponen antarmuka chat dengan manajemen multi-percakapan.
*   **`modul/`**: Direktori penyimpanan buku kurikulum AI sekolah menengah (.pdf) yang diindeks oleh sistem RAG.

---

## 🚀 Panduan Ringkas Cara Menjalankan Aplikasi

### 1. Setup Backend
1. Masuk ke folder backend dan pasang library dependensi:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
2. Salin berkas `.env` di root proyek ke direktori `backend/` dan isi `OPENROUTER_API_KEY` milik Anda.
3. Jalankan indexasi dokumen modul:
   ```bash
   python ingest.py
   ```
4. Aktifkan server FastAPI:
   ```bash
   python main.py
   ```
   *Server akan mendengarkan di http://127.0.0.1:8000.*

### 2. Setup Frontend
1. Masuk ke folder frontend dan pasang package NodeJS:
   ```bash
   cd ../frontend
   npm install
   ```
2. Jalankan aplikasi web client:
   ```bash
   npm run dev
   ```
   *Frontend akan dapat diakses secara lokal di browser melalui http://localhost:5173.*

### 3. Jalankan Pengujian Evaluasi Kuantitatif
Untuk memverifikasi metrik ROUGE, BERTScore, Faithfulness, Hit Rate, dan MRR secara otomatis, jalankan:
```bash
cd ../backend
python evaluator.py # untuk kuantitatif
python ragas_evaluator.py # untuk kualitatif
```
Hasil ringkasan skor rata-rata pencarian dan jawaban model akan ditampilkan langsung di terminal Anda.
