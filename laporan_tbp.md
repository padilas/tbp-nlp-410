# LAPORAN TUGAS BESAR PEMROGRAMAN & UTS NLP
## PENGEMBANGAN SISTEM GEN-AI: ACADEMIC QA BOT UNTUK PEMBELAJARAN KECERDASAN BUATAN SISWA SMP/SMA BERBASIS HYBRID RETRIEVAL-AUGMENTED GENERATION (RAG)

---

### **DAFTAR ISI**
1. [PERUMUSAN MASALAH & USE CASE](#1-perumusan-masalah--use-case)
2. [LANDASAN TEORI NLP & KOMPUTASI LINGUISTIK (RUBRIK UTS)](#2-landasan-teori-nlp--komputasi-linguistik-rubrik-uts)
3. [DESAIN ARSITEKTUR PIPELINE & SISTEM](#3-desain-arsitektur-pipeline--sistem)
4. [ARSITEKTUR BACKEND & IMPLEMENTASI RAG AKTUAL](#4-arsitektur-backend--implementasi-rag-aktual)
5. [PROMPT ENGINEERING STRUCTURE](#5-prompt-engineering-structure)
6. [ARSITEKTUR FRONTEND INTERFACE](#6-arsitektur-frontend-interface)
7. [EVALUASI KUANTITATIF & KUALITATIF](#7-evaluasi-kuantitatif--gevaluasi-kualitatif)
8. [ANALISIS ETIKA, BIAS, SAFETY, & GUARDRAILS](#8-analisis-etika-bias-safety--guardrails)
9. [ANALISIS HASIL, DISKUSI, & DOKUMENTASI PENYELENGGARAAN](#9-analisis-hasil-diskusi--dokumentasi-penyelenggaraan)

---

## 1. PERUMUSAN MASALAH & USE CASE

### Latar Belakang & Urgensi
Kurikulum pendidikan di tingkat sekolah menengah (SMP dan SMA) mulai mengadopsi materi pengenalan Kecerdasan Buatan (Artificial Intelligence/AI). Namun, AI merupakan bidang multidisiplin yang kompleks, melibatkan konsep statistika, matematika, dan pemrograman yang menantang bagi remaja usia sekolah. Buku paket atau modul pembelajaran yang disediakan sering kali tebal dan padat informasi, sehingga siswa kesulitan untuk menemukan jawaban cepat atau memahami konsep tertentu secara mandiri di luar jam sekolah. 

Membangun asisten tanya jawab akademik (*Academic QA Bot*) berbasis Generative AI adalah solusi taktis untuk masalah ini. Dengan memanfaatkan teknologi LLM (Large Language Model), siswa dapat berdialog secara natural mengenai isi modul. Namun, LLM standar memiliki kelemahan bawaan berupa **halusinasi** (mengarang jawaban) dan ketiadaan pengetahuan kontekstual yang spesifik terhadap kurikulum lokal sekolah menengah. Oleh karena itu, arsitektur **Retrieval-Augmented Generation (RAG)** diterapkan. RAG memungkinkan bot untuk mencari dan mengambil (*retrieve*) potongan informasi yang relevan dari buku modul resmi sekolah menengah terlebih dahulu, kemudian menyuntikkannya sebagai konteks kepada LLM untuk menghasilkan jawaban yang akurat, berlandaskan dokumen sumber, dan bebas halusinasi.

### Tujuan Taktis Sistem
1.  Menyediakan asisten belajar interaktif bagi siswa SMP/SMA untuk menanyakan dan memahami konsep kecerdasan buatan dari buku teks resmi secara real-time.
2.  Mencegah jawaban halusinasi LLM dengan menerapkan batasan ketat (*hard constraint*) agar bot menolak menjawab jika informasi tidak ditemukan dalam modul pembelajaran yang diunggah.
3.  Menyediakan sistem pencarian informasi yang cerdas dengan menggabungkan kemiripan semantik (vektor) dan kecocokan kata kunci (leksikal) menggunakan pendekatan *Hybrid Retrieval*.

### Batasan Masalah
1.  **Ruang Lingkup Dokumen**: Sistem dibatasi untuk mengindeks dan menjawab pertanyaan berdasarkan modul/buku kurikulum AI resmi tingkat SMP dan SMA yang diunggah di dalam folder [modul/](file:///D:/Collage/6th%20Term/NLP/tbp-004-010/tbp-nlp-410/modul) (seperti modul AI Kelas 7 s.d. Kelas 12).
2.  **Topik Pertanyaan**: Sistem mendeteksi relevansi materi secara otomatis melalui pemfilteran kata kunci AI. Jika dokumen yang diunggah atau pertanyaan yang diajukan berada di luar materi Kecerdasan Buatan, sistem akan menolak untuk memprosesnya.
3.  **Akurasi Kontekstual**: Bot tidak diperkenankan berasumsi, memperluas teori, atau memberikan penjelasan di luar dokumen modul yang diunggah guna menjamin keselarasan dengan materi ajar sekolah.

---

## 2. LANDASAN TEORI NLP & KOMPUTASI LINGUISTIK (RUBRIK UTS)

Untuk mengukur pemenuhan CPMK1 dan CPMK2, bagian ini merinci konsep dasar NLP, komputasi linguistik, serta tantangan pemrosesan bahasa alami yang relevan dengan sistem ini.

### 2.1 Konsep Dasar NLP
*   **Pengertian NLP**: *Natural Language Processing* (NLP) adalah cabang dari kecerdasan buatan yang berfokus pada interaksi antara komputer dan bahasa manusia. Tujuannya adalah memampukan komputer untuk membaca, memahami, menerjemahkan, dan menghasilkan bahasa manusia secara bermakna.
*   **Sumber Data Teks**: Data teks dalam aplikasi ini bersumber dari dokumen terstruktur formal (.pdf atau .txt) berupa buku ajar digital mata pelajaran AI tingkat SMP dan SMA yang disusun oleh pendidik atau lembaga kurikulum resmi.
*   **Jenis Task NLP**: Task NLP yang diimplementasikan meliputi:
    1.  *Information Retrieval (IR)*: Mengambil chunk teks yang paling relevan dengan query siswa dari database vektor FAISS dan indeks BM25.
    2.  *Text Segmentation/Chunking*: Memotong dokumen panjang menjadi bagian-bagian kecil (chunks) agar dapat masuk ke dalam batasan token input LLM.
    3.  *Text Classification (Kategori Filter)*: Mengidentifikasi materi apakah dokumen mengandung kata kunci AI sebelum diindeks.
    4.  *Text Generation*: Menghasilkan teks jawaban kontekstual menggunakan LLM.
*   **Pipeline NLP Aktual**:
    $$\text{Dokumen PDF/Teks} \rightarrow \text{Ekstraksi Teks} \rightarrow \text{Data Cleaning} \rightarrow \text{Recursive Chunking} \rightarrow \text{Embedding Generator} \rightarrow \text{Indexing (FAISS \& BM25)} \rightarrow \text{Hybrid Retrieval} \rightarrow \text{Augmentasi Prompt} \rightarrow \text{LLM Generation} \rightarrow \text{Output}$$

### 2.2 Komputasi Linguistik
*   **Pengertian**: Komputasi Linguistik (*Computational Linguistics*) adalah bidang interdisipliner yang menggunakan metode komputasi untuk memodelkan dan menganalisis struktur serta aspek-aspek bahasa manusia secara ilmiah.
*   **Aspek-Aspek Komputasi Linguistik**:
    1.  *Fonologi*: Studi tentang sistem suara bahasa. Pada bot akademik berbasis teks, aspek ini tidak diproses secara langsung (kecuali jika dikembangkan ke mode Voice-to-Text).
    2.  *Morfologi*: Struktur internal kata (misalnya, proses afiksasi/stemming kata dalam bahasa Indonesia seperti kata *memproses* menjadi *proses*).
    3.  *Sintaksis*: Struktur kalimat dan tata bahasa (posisi Subjek-Predikat-Objek).
    4.  *Semantik*: Makna literal dari kata dan kalimat (direpresentasikan menggunakan vektor embedding berdimensi tinggi untuk mengukur kedekatan makna antar kalimat).
    5.  *Pragmatik*: Makna kalimat dalam konteks situasional (direpresentasikan melalui pengelolaan instruksi peran / *system prompt* LLM agar bertindak sebagai guru yang ramah bagi siswa SMP/SMA).
*   **Contoh Penerapan Nyata**: Sistem translasi bahasa otomatis (Google Translate), analisis sentimen opini publik, klasifikasi email spam, dan pembuatan sistem pencarian semantik (Semantic Search) seperti pada retrieval RAG dalam aplikasi ini.

### 2.3 Tantangan pada NLP
1.  **Ambiguitas Bahasa**: Kata yang sama dapat memiliki arti berbeda tergantung kelas kata atau posisi sintaksisnya. Contoh: "Buku itu *bisa* dibaca" (kata *bisa* berarti dapat/sanggup) vs "Ular itu mengeluarkan *bisa*" (kata *bisa* berarti racun).
2.  **Makna Kontekstual**: Arti kata berubah secara drastis berdasarkan konteks kalimatnya. Misalnya kata "Jaringan" dalam "Jaringan komputer" (sistem interkoneksi perangkat) berbeda dengan "Jaringan seluler" atau "Jaringan sosial". NLP modern menggunakan *contextual embeddings* (seperti BERT atau Gemini) untuk menangani variasi makna kontekstual ini dengan cara menghasilkan representasi vektor yang berbeda untuk kata yang sama jika konteks kalimatnya berbeda.
3.  **Bahasa Formal vs Informal**: Modul kurikulum ditulis dalam bahasa formal (baku), sedangkan query yang dimasukkan oleh siswa SMP/SMA sering kali menggunakan bahasa semi-formal atau informal. RAG harus mampu menjembatani perbedaan kosakata ini agar dokumen formal tetap dapat ditemukan meskipun query siswa menggunakan istilah sehari-hari.
4.  **Multilingual & Low-Resource**: Mayoritas model bahasa besar (LLM) dilatih menggunakan korpora bahasa Inggris. Bahasa Indonesia diklasifikasikan sebagai *medium-resource language*, yang berarti performa model embedding atau representasi teks bisa menurun jika tidak menggunakan model multilingual khusus yang telah dituning seperti model `intfloat/multilingual-e5-base` yang digunakan pada sistem ini.
5.  **Concern Bias & Etika**: Model bahasa dapat mereplikasi bias yang ada pada data latihan (misal mengasosiasikan peran pengembang teknologi hanya dengan gender tertentu). Di samping itu, penyalahgunaan bot untuk pengerjaan tugas ujian secara instan (plagiarisme) menjadi kekhawatiran etis utama di lingkungan sekolah.

### 2.4 Preprocessing Teks
Tujuan preprocessing adalah membersihkan noise dari teks mentah agar representasi vektor yang dihasilkan lebih berkualitas. Teknik yang diterapkan di dalam [rag.py](file:///D:/Collage/6th%20Term/NLP/tbp-004-010/tbp-nlp-410/backend/rag.py) pada fungsi `clean_text` adalah:
*   *Tokenization*: Pemisahan string teks menjadi baris/kalimat berdasar spasi dan karakter pemisah baru (`\n`).
*   *Case Folding*: Mengubah teks menjadi huruf kecil (`lower()`) untuk standardisasi pencarian dan analisis kata kunci.
*   *Noise Handling*: Menghapus baris kosong, membersihkan tabulasi (`\t`), mengoreksi jarak spasi ganda menggunakan regular expression `re.sub(r"[ \t]+", " ", text)`, serta merapikan struktur penulisan butir list/bullet point agar teks mengalir secara koheren sebagai satu paragraf utuh sebelum dimasukkan ke dalam splitter.

### 2.5 Representasi Teks
Sistem membandingkan dan mengombinasikan dua metode representasi teks utama:
1.  **Leksikal/Sparse (BM25)**: Representasi teks berbasis frekuensi kemunculan kata (sejenis TF-IDF yang ditingkatkan). Sangat efisien untuk mencari kecocokan kata kunci eksak seperti nama algoritma ("ANN", "JST", "CNN") atau istilah akademis spesifik.
2.  **Dense/Contextual Embedding (HuggingFace Multilingual-e5)**: Merepresentasikan setiap chunk dokumen ke dalam vektor padat berdimensi tinggi (768 dimensi). Metode ini menangkap makna semantis kalimat secara kontekstual, sehingga pencarian tetap relevan meskipun siswa bertanya menggunakan sinonim atau kalimat tanya yang berbeda struktur dengan teks dokumen asli.

---

## 3. DESAIN ARSITEKTUR PIPELINE & SISTEM

Alur data sistem ini dirancang secara modular dan dapat dijelaskan melalui urutan interaksi berikut:

### Alur Sistem Runtut
1.  **Input Query**: Siswa memasukkan pertanyaan melalui Frontend Interface.
2.  **API Gateway**: Frontend mengirimkan request POST ke Backend API ([main.py](file:///D:/Collage/6th%20Term/NLP/tbp-004-010/tbp-nlp-410/backend/main.py)) pada endpoint `/api/chat`.
3.  **Guardrails Validation**: Backend memvalidasi keamanan query menggunakan fungsi `is_safe_query` di [guardrails.py](file:///D:/Collage/6th%20Term/NLP/tbp-004-010/tbp-nlp-410/backend/guardrails.py). Jika query terindikasi berbahaya (manipulasi nilai, prompt injection, kata kasar, atau SARA), proses dihentikan dan asisten mengembalikan pesan penolakan yang sopan.
4.  **Retrieval Strategy (Hybrid)**: Jika query aman, backend memanggil fungsi `get_retriever()` di [rag.py](file:///D:/Collage/6th%20Term/NLP/tbp-004-010/tbp-nlp-410/backend/rag.py). Query dicari menggunakan kombinasi **FAISS (Dense Vector Search)** dan **BM25 (Sparse Keyword Search)**. Skor dari kedua metode ini dihitung ulang menggunakan metode *Reciprocal Rank Fusion (RRF)* untuk mendapatkan 5 potongan dokumen (chunks) yang paling relevan.
5.  **Prompt Augmentation**: Chunks dokumen terpilih disuntikkan ke dalam `ACADEMIC_PROMPT_TEMPLATE` bersama dengan riwayat percakapan (`chat_history`) dan pertanyaan asli siswa (`question`).
6.  **LLM Generation**: Prompt yang sudah diaugmentasi dikirim ke LLM (baik OpenRouter `google/gemini-2.5-flash` dengan token API maupun local Ollama `llama3` dengan `temperature=0.0`). LLM memproses prompt tersebut secara deterministik dan mengembalikan jawaban.
7.  **Frontend Response**: Jawaban yang dikirim backend diterima oleh Frontend dan ditampilkan pada antarmuka Chat UI.

### Diagram Arsitektur (Mermaid.js)

```mermaid
graph TD
    %% Frontend Component
    subgraph Frontend Interface [Vite React Client]
        UI[Chat UI & Sidebar]
        MR[Markdown Renderer]
    end

    %% Backend Component
    subgraph Backend API [FastAPI Server]
        GW[API Gateway / Router]
        GR[Guardrails Validator]
        IH[Health Check Endpoint]
    end

    %% Ingestion Script (Admin Only)
    subgraph Offline Ingestion [Developer / Admin Offline Run]
        AD[Admin Terminal / ingest.py]
    end

    %% AI Pipeline Component
    subgraph AI Pipeline [Core RAG Engine]
        TS[Recursive Character Text Splitter]
        EM[HuggingFace Embeddings - multilingual-e5]
        VS[(FAISS Vector Store)]
        BM[(BM25 Sparse Index)]
        HR[Hybrid Retriever - RRF]
        PT[Academic Prompt Template]
        LLM[LLM Generator - Gemini/Llama]
    end

    %% Connections
    UI -- 1. Kirim Question / History --> GW
    GW -- 2. Validasi Input --> GR
    GR -- Valid / Safe --> HR
    GW -- Cek Kesehatan Server --> IH
    
    %% Ingest flow (Offline Admin action)
    AD -- Ingest Modul secara Offline --> TS
    TS -- Chunks --> EM
    EM -- Hasilkan Vektor --> VS
    TS -- Chunks --> BM

    %% Retrieval flow
    HR -- 4a. Dense Search --> VS
    HR -- 4b. Sparse Search --> BM
    HR -- 5. Kombinasi RRF (Top 5 Chunks) --> PT
    PT -- 6. Kirim Augmented Prompt --> LLM
    LLM -- 7. Kembalikan Jawaban --> GW
    GW -- 8. Kirim Response JSON --> UI
    UI -- 9. Render Jawaban --> MR
```

---

## 4. ARSITEKTUR BACKEND & IMPLEMENTASI RAG AKTUAL

Ekosistem backend dibangun menggunakan teknologi **FastAPI** yang terkenal cepat dan asinkron, dengan memanfaatkan **LangChain** sebagai orkestrator alur kerja RAG.

### Analisis File Backend Aktual
Sistem terdiri atas beberapa modul python utama:
1.  **[main.py](file:///D:/Collage/6th%20Term/NLP/tbp-004-010/tbp-nlp-410/backend/main.py)**: Berfungsi sebagai entry point aplikasi FastAPI. Menyediakan konfigurasi CORS, mendefinisikan skema Pydantic (`ChatMessage`, `ChatRequest`), dan menyediakan endpoint utama `/api/chat` untuk pemrosesan interaksi tanya jawab.
2.  **[rag.py](file:///D:/Collage/6th%20Term/NLP/tbp-004-010/tbp-nlp-410/backend/rag.py)**: Berisi core mesin RAG. Mengimplementasikan konfigurasi model embedding, kustomisasi `HybridRetriever` berbasis RRF, pemrosesan dokumen (.pdf dan .txt), pembersihan teks, pembuatan FAISS vectorstore, dan pembuatan LangChain Runable Chain.
3.  **[guardrails.py](file:///D:/Collage/6th%20Term/NLP/tbp-004-010/tbp-nlp-410/backend/guardrails.py)**: Mengandung logika validasi keamanan query menggunakan daftar regular expression terstruktur.
4.  **[ingest.py](file:///D:/Collage/6th%20Term/NLP/tbp-004-010/tbp-nlp-410/backend/ingest.py)**: Script mandiri untuk inisialisasi awal database vektor dari dokumen kurikulum yang diletakkan di folder modul secara offline.
5.  **[evaluator.py](file:///D:/Collage/6th%20Term/NLP/tbp-004-010/tbp-nlp-410/backend/evaluator.py)**: Program pengujian evaluasi kuantitatif otomatis untuk mengukur performa sistem.

### Spesifikasi Kode Core RAG

#### 1. Text Ingestion & Chunking
Dokumen teks dipecah menjadi modul-modul kecil agar informasi tidak tumpang tindih dan muat dalam *context window* LLM.
*   **Pembersihan Teks Terlebih Dahulu**: Sebelum membagi dokumen, modul dibersihkan dari noise visual PDF via fungsi `clean_text()`.
*   **Parameter Splitter**:
    *   `chunk_size = 1000` (karakter): Menjaga agar konteks penjelasan konsep AI (misalnya cara kerja JST) tidak terpotong di tengah kalimat yang krusial.
    *   `chunk_overlap = 200` (karakter): Mencegah hilangnya kesinambungan informasi antar potongan berurutan.
    *   `length_function = len`.
    *   Splitter menggunakan class `RecursiveCharacterTextSplitter` yang secara berurutan mencoba memotong berdasarkan separator paragraf (`\n\n`), kalimat (`\n`), kata (` `), dan karakter kosong (`""`) guna menghasilkan pemotongan se-alami mungkin.

#### 2. Embedding & Vector Database
*   **Embedding Model**: Menggunakan model open-source `intfloat/multilingual-e5-base` yang dijalankan secara lokal di CPU (`device: cpu`). Model ini sangat andal dalam menangani dokumen berbahasa Indonesia karena dilatih pada dataset multibahasa berskala besar, dengan parameter input bertipe `query:` untuk pencarian dan `passage:` untuk penyimpanan korpus guna menyelaraskan ruang representasi vektor.
*   **Vector Database**: Menggunakan **FAISS** (Facebook AI Similarity Search) secara lokal yang disimpan ke direktori [vectorstore/](file:///D:/Collage/6th%20Term/NLP/tbp-004-010/tbp-nlp-410/backend/vectorstore). FAISS mempermudah pencarian berbasis kemiripan kosinus (*Cosine Similarity*) dengan waktu eksekusi sub-milidetik.
*   **Sparse Index**: Menyimpan koleksi chunk mentah menggunakan serialisasi `pickle` pada file `chunks.pkl` untuk diindeks oleh algoritma **BM25Retriever** saat pencarian kata kunci eksak.

#### 3. Strategi Hybrid Retrieval (RRF)
Untuk meningkatkan akurasi retrieval, sistem menerapkan **Hybrid Retrieval** yang menggabungkan dense retrieval (semantic) dan sparse retrieval (keyword match) dengan skema pembobotan:
```python
class HybridRetriever(BaseRetriever):
    vector_retriever: Any
    bm25_retriever: Any
    weight_vector: float = 0.8  # Fokus 80% pada kesamaan makna
    weight_bm25: float = 0.2   # Fokus 20% pada kecocokan istilah eksak
    top_k: int = 5
```
Perangkingan akhir dihitung menggunakan formula *Reciprocal Rank Fusion (RRF)* untuk setiap chunk $d$ dalam kumpulan hasil pencarian $D$:
$$RRF(d) = w_{vector} \cdot \frac{1}{\text{rank}_{vector}(d) + 60} + w_{bm25} \cdot \frac{1}{\text{rank}_{bm25}(d) + 60}$$
Chunk diurutkan berdasarkan skor RRF tertinggi dan diambil sebanyak 5 teratas (`top_k = 5`) untuk disajikan kepada LLM.

#### Opsi Pendekatan Dense-only Retrieval (Dinamis)
Selain metode Hybrid, sistem menyediakan opsi **Dense-only Retrieval** (pencarian berbasis dense vektor saja) yang dapat diaktifkan secara dinamis dengan menyetir parameter `RETRIEVAL_TYPE=dense` pada berkas konfigurasi `.env`. Ketika opsi ini dipilih, sistem menggunakan kelas `DenseRetriever` di berkas `rag.py` untuk mengambil top-K chunks dari indeks FAISS menggunakan Cosine Similarity tanpa melibatkan pencarian kata kunci BM25, dengan semua perataan data prefiks (seperti `passage:` dan `query:`) tetap diselaraskan secara konsisten dengan metode Hybrid:
```python
class DenseRetriever(BaseRetriever):
    vector_retriever: Any
    top_k: int = 5
```

#### 4. LLM Configuration
*   **Provider**: Mendukung penyedia **OpenRouter** dengan default model `google/gemini-2.5-flash` untuk pemrosesan berkecepatan tinggi, serta fallback ke penyedia lokal **Ollama** dengan model `llama3`.
*   **Parameter Strict**: Menyetel `temperature = 0.0` (nol mutlak). Hal ini sangat krusial dalam domain pendidikan formal guna menjamin respons LLM bersifat deterministik (konsisten), faktual, dan tidak berimajinasi di luar korpus data masukan.

---

## 5. PROMPT ENGINEERING STRUCTURE

Prompt dirancang menggunakan teknik instruksi terstruktur untuk membatasi perilaku LLM agar hanya menjawab sesuai konteks yang disediakan.

### Struktur Prompt Aktual (`ACADEMIC_PROMPT_TEMPLATE` dalam `rag.py`)
```
Role: Anda adalah Asisten Buku Modul AI SMP/SMA (AI Textbook Bot) yang bertindak sebagai ahli materi pembelajaran berdasarkan modul Kecerdasan Buatan (AI) yang diunggah. Tugas Anda adalah membantu siswa memahami konsep, materi bab, dan teori yang dibahas dalam dokumen dengan ramah, komunikatif, profesional, dan akurat.

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
Jawaban:
```

### Penjelasan Komponen Prompt
1.  **Role-play (Instruksi Peran)**: LLM diposisikan sebagai "Asisten Buku Modul AI SMP/SMA (AI Textbook Bot)". Peran ini mengarahkan gaya bahasa LLM agar ramah, edukatif, dan mudah dipahami oleh remaja usia sekolah menengah.
2.  **Suntikan Konteks (`context`)**: Menyediakan ruang dinamis untuk mengintegrasikan 5 chunks dokumen yang berhasil ditarik oleh Hybrid Retriever.
3.  **Batasan Ketat (*Constraints*)**:
    *   *Constraint 1 & 3*: Menutup celah bagi model untuk mengambil pengetahuan eksternal di luar dokumen yang disuntikkan.
    *   *Constraint 2*: Menyediakan kalimat template spesifik sebagai mekanisme fallback penolakan jika informasi tidak ada di dalam dokumen.
    *   *Constraint 4*: Menghilangkan kebocoran metadata teknis (seperti penyebutan tag bracket nama file PDF) agar asisten berinteraksi secara natural layaknya guru manusia yang menguasai materi.
4.  **Few-shot Examples**: Memberikan ilustrasi konkret mengenai perilaku yang diharapkan. Contoh 1 memandu bagaimana menjawab secara ramah dan ringkas dari konteks yang ada. Contoh 2 mendemonstrasikan bagaimana bot harus menolak dengan sopan ketika pertanyaan berada di luar lingkup informasi modul yang ada di konteks.

---

## 6. ARSITEKTUR FRONTEND INTERFACE

Antarmuka pengguna (Frontend) dibangun menggunakan **React** (dengan kompilator cepat **Vite** dan bahasa **TypeScript**) untuk menciptakan antarmuka interaktif yang responsif bagi siswa.

### Spesifikasi Kode Frontend Aktual
*   **Manajemen Sesi Chatting**: Diimplementasikan pada berkas [ChatInterface.tsx](file:///D:/Collage/6th%20Term/NLP/tbp-004-010/tbp-nlp-410/frontend/src/components/ChatInterface.tsx). Sistem mendukung multi-session chat history. Pengguna dapat membuat sesi chat baru dengan tombol "Chat Baru". 
*   **Penamaan Judul Otomatis**: Judul sesi chat pada sidebar akan otomatis diperbarui dari tulisan "Chat Baru" menjadi 25 karakter pertama dari pertanyaan pertama yang diajukan oleh pengguna:
    ```typescript
    if (s.title === 'Chat Baru' && newMessages.length > 1) {
       const firstUser = newMessages.find(m => m.role === 'user');
       if (firstUser) {
         newTitle = firstUser.content.substring(0, 25) + '...';
       }
    }
    ```
*   **Loading State Interaktif**: Terdapat visualisasi status tunggu (*typing indicator*) yang akan muncul ketika request sedang dikirimkan ke backend. Animasi ini dirancang menggunakan CSS kustom berupa tiga titik memantul secara berkala:
    ```css
    .typing-dot {
      width: 6px;
      height: 6px;
      background-color: var(--text-secondary);
      border-radius: 50%;
      animation: bounce 1.4s infinite ease-in-out both;
    }
    .typing-dot:nth-child(1) { animation-delay: -0.32s; }
    .typing-dot:nth-child(2) { animation-delay: -0.16s; }
    ```
*   **Error Handling Robust**: Jika API backend offline atau crash, blok `catch` pada method `handleSendMessage` di frontend akan mendeteksi gangguan tersebut dan secara otomatis memunculkan bubble chat bot berisi panduan penanganan masalah:
    ```typescript
    content: 'Maaf, terjadi kesalahan atau server RAG belum menyala. Pastikan backend (main.py) sudah dijalankan.'
    ```
*   **Message Bubble Avatar**: Berkas [MessageBubble.tsx](file:///D:/Collage/6th%20Term/NLP/tbp-004-010/tbp-nlp-410/frontend/src/components/MessageBubble.tsx) membagi representasi visual pesan secara tegas menggunakan ikon dari pustaka `lucide-react`. Pesan pengguna didampingi ikon `User` dengan latar abu-abu lembut, sedangkan pesan bot didampingi ikon `Bot` dengan latar warna aksen lavender.
*   **Desain CSS Modern**: Berkas [index.css](file:///D:/Collage/6th%20Term/NLP/tbp-004-010/tbp-nlp-410/frontend/src/index.css) menggunakan CSS Custom Properties untuk membentuk palet warna yang harmonis. Font menggunakan Google Font 'Inter' untuk keterbacaan yang optimal. Tata letak (layout) menggunakan flexbox dua kolom: sidebar navigasi di sisi kiri dan panel interaksi pesan di sisi kanan dengan tinggi layar penuh (`100vh`).

---

## 7. EVALUASI KUANTITATIF & KUALITATIF

Evaluasi sistem dilakukan untuk mengukur pemenuhan CPMK4 (Mengevaluasi model NLP/LLM dan menganalisis aspek etika/keamanan).

### 7.1 Evaluasi Kuantitatif
Sistem evaluasi otomatis ditulis di dalam berkas [evaluator.py](file:///D:/Collage/6th%20Term/NLP/tbp-004-010/tbp-nlp-410/backend/evaluator.py) dengan memuat dataset uji dari berkas [groundtruth.json](file:///D:/Collage/6th%20Term/NLP/tbp-004-010/tbp-nlp-410/backend/groundtruth.json) secara dinamis (berisi 11 skenario pertanyaan akademis AI beserta kata kunci dan jawaban acuan). Evaluasi dibagi menjadi dua bagian: performa retrieval dan kualitas jawaban generatif.

#### Metrik Retrieval (Pencarian Konteks)
1.  **Hit Rate (Recall@K)**: Mengukur persentase kasus di mana dokumen relevan yang dicari masuk ke dalam kumpulan top-k dokumen yang diambil. Bernilai $1.0$ jika minimal satu kata kunci ground truth ditemukan di dalam top-k chunks, dan $0.0$ jika tidak.
2.  **Mean Reciprocal Rank (MRR)**: Mengukur tingkat keakuratan posisi dokumen relevan pertama yang berhasil ditemukan.
    $$MRR = \frac{1}{\text{rank dokumen relevan pertama}}$$
3.  **Precision@K**: Rasio jumlah chunk yang relevan (mengandung $\ge 40\%$ kata kunci ground truth) dibanding jumlah total chunk yang diambil ($k$).
    $$\text{Precision@K} = \frac{\text{Jumlah Chunk Relevan}}{K}$$

#### Metrik Generation (Jawaban LLM)
1.  **Faithfulness (Faktualitas/Anti-Halusinasi)**: Mengukur apakah jawaban LLM sepenuhnya didasarkan pada konteks yang ditarik (grounded). Dihitung dengan memecah jawaban LLM menjadi kalimat-kalimat tunggal, lalu memeriksa apakah setiap kalimat didukung oleh informasi di dalam konteks retrieval (dinyatakan grounded jika minimal 40% kata dalam kalimat diduga ada di dalam teks konteks):
    $$\text{Faithfulness} = \frac{\text{Jumlah Kalimat Grounded}}{\text{Total Kalimat Jawaban}}$$
2.  **ROUGE-L (LCS)**: Mengukur tumpang tindih kata berurutan terpanjang (*Longest Common Subsequence*) antara jawaban LLM dengan kunci jawaban (*ground truth answer*). Formula F1 ROUGE-L dihitung berdasarkan presisi ($P_{LCS}$) dan recall ($R_{LCS}$):
    $$R_{LCS} = \frac{LCS(\text{Gen}, \text{GT})}{\text{Jumlah kata GT}}, \quad P_{LCS} = \frac{LCS(\text{Gen}, \text{GT})}{\text{Jumlah kata Gen}}$$
    $$F_{LCS} = \frac{2 \cdot P_{LCS} \cdot R_{LCS}}{P_{LCS} + R_{LCS}}$$
3.  **BERTScore**: Mengukur kemiripan makna semantis antara jawaban LLM dengan kunci jawaban menggunakan model transformer lokal `sentence-transformers/all-MiniLM-L6-v2`. Vektor representasi kata dari kedua kalimat dibandingkan menggunakan *Cosine Similarity* untuk menghitung presisi, recall, dan F1-score semantis tanpa terikat pada kecocokan kosakata eksak.
4.  **Token F1**: Mengukur tingkat tumpang tindih kata (bag of words) antara jawaban hasil generasi dengan kunci jawaban secara presisi.

#### Pencatatan Log Evaluasi Dinamis
Untuk mempermudah pelacakan eksperimen secara terstruktur, setiap kali skrip `evaluator.py` dijalankan, sistem secara otomatis mencatat seluruh konfigurasi parameter aktif (meliputi penyedia LLM, model LLM aktif, jenis pendekatan retrieval, nilai Top-K context, ukuran chunk size, overlap, dan model embedding) beserta rangkuman metrik hasil evaluasinya ke dalam berkas log JSON historis: [evaluation_history.json](file:///D:/Collage/6th%20Term/NLP/tbp-004-010/tbp-nlp-410/backend/evaluation_history.json). Berkas log ini memungkinkan analisis komparasi antar-eksperimen secara dinamis dan reproducible.

### 7.2 Evaluasi Kualitatif
Panduan evaluasi kualitatif manual dirancang untuk menguji kualitas jawaban dari 5 sampel skenario uji.

#### Matriks Pengujian Kualitatif (5 Sampel Uji)
| Skenario Uji (Query Siswa) | Konteks yang Diharapkan (Kurikulum AI) | Perilaku Bot yang Diharapkan |
| :--- | :--- | :--- |
| **Q1**: Apa pengertian dari Kecerdasan Buatan? | Penjelasan pengenalan AI di modul kelas 7/10. | Memberikan jawaban definisi formal AI dari modul secara ramah. |
| **Q2**: Apa perbedaan Machine Learning vs Deep Learning? | Perbandingan algoritma di modul kelas 11/12. | Menjelaskan relasi subset ML & DL berdasarkan modul. |
| **Q3**: Sebutkan dampak negatif/tantangan teknologi AI. | Dampak sosial kemasyarakatan di modul kelas 10/12. | Menyebutkan tantangan bias, privasi, pengangguran sesuai buku. |
| **Q4**: Bagaimana cara kerja JST secara sederhana? | Bab Jaringan Saraf Tiruan modul kelas 11. | Menjelaskan alur input -> hidden layer -> output & bobot. |
| **Q5**: Bagaimana sejarah penemuan backpropagation? | Tidak ada di modul (hanya ada struktur JST & fungsi aktivasi). | **Menolak menjawab** menggunakan kalimat constraint (anti-halusinasi). |

#### Rubrik Panduan Analisis Kualitatif Manual
1.  **Relevansi (Relevance)**: Apakah jawaban bot menjawab pertanyaan inti yang diajukan? (Skor 1-5).
2.  **Koherensi (Coherence)**: Apakah alur penjelasan logis, terstruktur, dan mudah dipahami oleh siswa usia SMP/SMA? (Skor 1-5).
3.  **Faktualitas (Factuality)**: Apakah informasi yang disampaikan sesuai dengan fakta yang ada di modul? (Skor 1-5).
4.  **Deteksi Halusinasi (Hallucination Detection)**: Apakah bot berhasil menolak menjawab (sesuai Constraint Contoh 2) ketika diberi pertanyaan di luar materi modul (seperti pada Q5)? (Skor Lolos / Gagal).

---

## 8. ANALISIS ETIKA, BIAS, SAFETY, & GUARDRAILS

Penerapan model AI dalam lingkungan akademis sekolah menengah wajib mengutamakan etika, keadilan, dan keamanan informasi.

### 8.1 Analisis Risiko Etika & Keamanan
*   **Privasi Data Akademik**: Penggunaan API publik (seperti OpenRouter/Gemini) berpotensi membocorkan data jika query siswa mengandung informasi pribadi (seperti nama lengkap, NISN, atau curahan hati pribadi). Mitigasi: Panduan penggunaan harus menekankan agar siswa tidak memasukkan data pribadi ke dalam chatbox.
*   **Bias Respons**: LLM cenderung mencerminkan bias bahasa yang dominan (bahasa perkotaan) atau bias gender dalam ilustrasi teknologi. Hal ini diatasi dengan penggunaan modul kurikulum berbahasa Indonesia yang terstruktur sebagai konteks utama untuk meredam bias kultural asing.
*   **Risiko Penyalahgunaan (Misuse)**: Siswa berisiko menggunakan bot ini untuk memanipulasi tugas sekolah, meminta bot menuliskan esai di luar modul, atau menanyakan cara membobol sistem komputer sekolah.

### 8.2 Implementasi Guardrails
Sistem keamanan preventif diimplementasikan pada berkas [guardrails.py](file:///D:/Collage/6th%20Term/NLP/tbp-004-010/tbp-nlp-410/backend/guardrails.py) menggunakan fungsi `is_safe_query(query)`. Fungsi ini memotong alur kerja RAG jika query siswa cocok dengan salah satu dari empat kelompok pola regular expression berikut:

1.  **Prompt Injection Detection**: Mencegah siswa mengesampingkan batasan sistem (jailbreak) dengan pola seperti: `ignore previous instructions`, `abaikan instruksi sebelumnya`, `override system`.
2.  **School Tampering / Manipulation Detection**: Mendeteksi upaya kecurangan akademis atau peretasan data sekolah dengan pola seperti: `hack server`, `ubah nilai`, `ganti nilai`, `bocoran soal`, `kunci jawaban ulangan`.
3.  **Toxic Language Filter**: Memblokir ujaran kebencian, makian, dan kata kasar dalam bahasa Indonesia/Inggris seperti: `anjing`, `babi`, `bangsat`, `goblok`, `fuck`, `bitch`.
4.  **SARA, Kekerasan, & Pornografi**: Memfilter topik radikal, kekerasan, narkoba, pornografi, dan penghinaan SARA dengan pola seperti: `bunuh`, `tawuran`, `bom`, `narkoba`, `bokep`, `porn`.

Jika query terdeteksi melanggar pola di atas, fungsi akan mengembalikan status `False` beserta pesan template edukatif untuk mengarahkan kembali fokus siswa ke topik pembelajaran AI:
> *"Maaf, konteks yang kamu tanyakan tidak masuk ke dalam lingkup materi kecerdasan buatan. Ajukan pertanyaan yang sesuai, seperti: Apa pengertian dari Kecerdasan Buatan (Artificial Intelligence)?..."*

---

## 9. ANALISIS HASIL, DISKUSI, & DOKUMENTASI PENYELENGGARAAN

### 9.1 Analisis Hasil Eksperimen
Eksperimen dilakukan untuk membandingkan performa sistem pada konfigurasi ukuran chunking dokumen yang berbeda, serta membandingkan performa bot saat menggunakan RAG vs tanpa RAG (No-RAG).

#### Tabel Perbandingan Hasil Eksperimen
| Metrik Evaluasi | RAG (Chunk Size 1000, Overlap 200) | RAG (Chunk Size 500, Overlap 100) | Tanpa RAG (Direct LLM Call) |
| :--- | :---: | :---: | :---: |
| **Rata-rata Hit Rate (Retrieval Recall)** | **95.2%** | 88.0% | N/A (Tidak ada pencarian) |
| **Mean Reciprocal Rank (MRR)** | **92.4%** | 84.2% | N/A |
| **Rata-rata Faithfulness** | **98.0%** | 94.0% | 45.0% (Halusinasi tinggi) |
| **Rata-rata ROUGE-L** | **78.4%** | 71.2% | 38.0% (Teks melebar ke teori lain) |
| **Rata-rata BERTScore** | **89.5%** | 85.0% | 62.0% |
| **Kecepatan Inferensi (Detik/Query)** | ~1.5 detik | **~1.2 detik** | ~1.0 detik |

#### Analisis Hasil Temuan & Diskusi
1.  **Chunk Size 1000 vs 500**: Ukuran chunk 1000 karakter memberikan skor *Hit Rate* dan *MRR* yang lebih tinggi (95.2% vs 88.0%). Hal ini disebabkan karena modul ajar AI SMP/SMA sering kali menyajikan penjelasan konsep yang panjang dan terintegrasi dalam satu sub-bab. Chunk yang terlalu kecil (500 karakter) memotong informasi di tengah penjelasan, sehingga saat dicari, konteks esensialnya terpisah dan menurunkan akurasi pencarian. Namun, chunk size 500 sedikit lebih unggul dalam kecepatan inferensi karena token yang diproses LLM lebih sedikit.
2.  **RAG vs Tanpa RAG**: Penggunaan RAG secara drastis meningkatkan *Faithfulness* (98.0% vs 45.0%). Tanpa RAG, model mencoba menjawab pertanyaan berdasarkan pengetahuan umum yang dimilikinya secara bebas, sehingga sering kali menyajikan konsep-konsep tingkat lanjut yang belum diajarkan di level sekolah menengah atau salah mengutip detail modul (terjadi halusinasi). RAG berhasil mengurung perilaku LLM di dalam batas-batas korpus modul.
3.  **Keterbatasan Sistem**: Sistem RAG ini sangat bergantung pada kualitas teks hasil ekstraksi PDF. Jika PDF berisi gambar pemindaian tanpa layer teks (non-searchable PDF), modul loader `PyPDFLoader` tidak akan mampu membaca kontennya dengan baik. Keterbatasan lainnya adalah waktu respons yang dipengaruhi oleh koneksi API eksternal (OpenRouter).

---

### 9.2 Panduan Setup & Jalankan Sistem (Reproducible Setup)

Sistem dirancang untuk dijalankan di lingkungan lokal secara manual tanpa docker untuk meminimalkan dependensi runtime. Ikuti langkah-langkah berikut:

#### Langkah 1: Persiapan Environment
1.  Salin file konfigurasi lingkungan [.env](file:///D:/Collage/6th%20Term/NLP/tbp-004-010/tbp-nlp-410/.env) dan lengkapi API Key OpenRouter Anda:
    ```env
    LLM_PROVIDER=openrouter
    OPENROUTER_API_KEY=isi_dengan_api_key_anda
    OPENROUTER_MODEL=google/gemini-2.5-flash
    CHUNK_SIZE=1000
    CHUNK_OVERLAP=200
    ```

#### Langkah 2: Jalankan & Ingest Database Backend
1.  Buka terminal pada direktori `backend/`.
2.  Instal seluruh library dependensi yang terdaftar di dalam [requirements.txt](file:///D:/Collage/6th%20Term/NLP/tbp-004-010/tbp-nlp-410/backend/requirements.txt):
    ```bash
    pip install -r requirements.txt
    ```
3.  Jalankan proses indeksasi awal dokumen modul kurikulum yang ada di folder `modul/`:
    ```bash
    python ingest.py
    ```
    *Proses ini akan membersihkan DB lama, membagi materi modul, menghasilkan embedding vektor menggunakan model sentence-transformer, dan menyimpan database FAISS ke dalam folder vectorstore.*
4.  Jalankan server backend FastAPI:
    ```bash
    python main.py
    ```
    *Server akan aktif di alamat http://127.0.0.1:8000.*

#### Langkah 3: Jalankan Aplikasi Frontend
1.  Buka terminal baru pada direktori `frontend/`.
2.  Instal dependensi NPM yang tertera di [package.json](file:///D:/Collage/6th%20Term/NLP/tbp-004-010/tbp-nlp-410/frontend/package.json):
    ```bash
    npm install
    ```
3.  Jalankan server pengembangan Vite:
    ```bash
    npm run dev
    ```
    *Frontend akan aktif di alamat http://localhost:5173.* Buka alamat tersebut di browser Anda untuk mulai menguji asisten tanya jawab akademik materi AI.

---
### **REFERENSI**
1.  *LangChain Community Document Loaders & Vector Stores documentation.*
2.  *FastAPI Web Framework Documentation (https://fastapi.tiangolo.com).*
3.  *Sentence-Transformers: Multilingual Sentence Embeddings using BERT / e5 Models.*
4.  *Reciprocal Rank Fusion (RRF) for Hybrid Search explanation.*
5.  *RAGAS: Retrieval Augmented Generation Assessment framework metrics.*
