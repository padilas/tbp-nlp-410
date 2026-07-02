import os
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag import (
    get_retriever, get_embeddings, clean_text, get_llm, ACADEMIC_PROMPT_TEMPLATE,
    LLM_PROVIDER, OPENROUTER_MODEL, OLLAMA_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME
)
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

BERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
try:
    print(f"Menginisialisasi model token-embedding untuk BERTScore ({BERT_MODEL_NAME})...")
    try:
        bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME, local_files_only=True)
        bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME, local_files_only=True)
    except Exception:
        bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME)
except Exception as e:
    print(f"[WARNING] Gagal memuat model BERTScore: {e}")
    bert_tokenizer = None
    bert_model = None

import json

# dari groundtruth.json
GROUNDTRUTH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "groundtruth.json")
try:
    print(f"Membaca dataset evaluasi dari: {GROUNDTRUTH_PATH}...")
    with open(GROUNDTRUTH_PATH, "r", encoding="utf-8") as f:
        TEST_DATASET = json.load(f)
    print(f"Sukses memuat {len(TEST_DATASET)} sampel uji dari groundtruth.json")
except Exception as e:
    print(f"[WARNING] Gagal memuat groundtruth.json secara otomatis: {e}. Menggunakan fallback.")
    TEST_DATASET = [
        {
            "question": "Apa pengertian dari Kecerdasan Buatan (Artificial Intelligence)?",
            "ground_truth_keywords": ["kecerdasan buatan", "artificial intelligence", "simulasi", "komputer", "berpikir"],
            "ground_truth_answer": "Kecerdasan Buatan atau Artificial Intelligence (AI) adalah teknologi yang mensimulasikan kecerdasan manusia ke dalam mesin atau sistem komputer agar mampu berpikir, belajar, menalar, dan memecahkan masalah."
        },
        {
            "question": "Apa perbedaan antara Machine Learning dan Deep Learning?",
            "ground_truth_keywords": ["machine learning", "deep learning", "algoritma", "jaringan saraf tiruan", "neural networks", "arsitektur"],
            "ground_truth_answer": "Machine Learning adalah cabang AI yang memungkinkan komputer belajar dari data tanpa diprogram secara eksplisit, sedangkan Deep Learning adalah subbidang Machine Learning yang menggunakan arsitektur Jaringan Saraf Tiruan (Neural Networks) berlapis banyak untuk memproses pola data yang lebih kompleks."
        },
        {
            "question": "Sebutkan dampak negatif atau tantangan dari penggunaan teknologi AI.",
            "ground_truth_keywords": ["dampak negatif", "tantangan", "bias", "privasi", "pekerjaan", "pengangguran"],
            "ground_truth_answer": "Dampak negatif atau tantangan AI meliputi ancaman terhadap privasi data, potensi bias dalam keputusan model, penyebaran disinformasi/hoaks, serta otomatisasi pekerjaan yang dapat memicu pengangguran."
        },
        {
            "question": "Bagaimana cara kerja jaringan saraf tiruan secara sederhana?",
            "ground_truth_keywords": ["jaringan saraf tiruan", "neuron", "input", "hidden layer", "output", "bobot", "weight"],
            "ground_truth_answer": "Jaringan saraf tiruan bekerja dengan meniru neuron biologis manusia, di mana data input dimasukkan, diproses melalui lapisan tersembunyi (hidden layers) dengan perkalian bobot (weights) dan bias, serta menghasilkan output setelah melewati fungsi aktivasi."
        },
        {
            "question": "Mengapa data berkualitas sangat penting dalam melatih model AI?",
            "ground_truth_keywords": ["data berkualitas", "penting", "melatih model", "akurasi", "bias", "pembelajaran"],
            "ground_truth_answer": "Data berkualitas tinggi sangat penting karena model AI belajar sepenuhnya dari pola data. Jika data yang digunakan melatih model mengandung banyak noise atau bias, maka hasil prediksi model juga akan tidak akurat dan bias (garbage in, garbage out)."
        }
    ]

def calculate_lcs(x: list, y: list) -> int:
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

def calculate_rouge_l(candidate: str, reference: str) -> float:
    c_tokens = candidate.lower().split()
    r_tokens = reference.lower().split()
    
    if not c_tokens or not r_tokens:
        return 0.0   
    lcs_len = calculate_lcs(c_tokens, r_tokens)
    recall = lcs_len / len(r_tokens)
    precision = lcs_len / len(c_tokens)
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    return f1

def calculate_bertscore(candidate: str, reference: str) -> float:
    if not bert_tokenizer or not bert_model or not candidate.strip() or not reference.strip():
        return 0.0
        
    try:
        c_inputs = bert_tokenizer(candidate, return_tensors="pt", truncation=True, max_length=512)
        r_inputs = bert_tokenizer(reference, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            c_outputs = bert_model(**c_inputs)
            r_outputs = bert_model(**r_inputs)
            
        c_emb = c_outputs.last_hidden_state[0]
        r_emb = r_outputs.last_hidden_state[0]
        c_emb = c_emb / c_emb.norm(dim=-1, keepdim=True)
        r_emb = r_emb / r_emb.norm(dim=-1, keepdim=True)

        sim_matrix = torch.matmul(c_emb, r_emb.T)
        recall = sim_matrix.max(dim=0).values.mean().item()
        precision = sim_matrix.max(dim=1).values.mean().item()
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        return f1
    except Exception as e:
        print(f"[Error calculating BERTScore: {e}]")
        return 0.0

def calculate_token_f1(generated: str, ground_truth: str) -> float:
    gen_tokens = generated.lower().split()
    gt_tokens = ground_truth.lower().split()
    
    if not gen_tokens or not gt_tokens:
        return 0.0    
    gen_set = set(gen_tokens)
    gt_set = set(gt_tokens)
    common = gen_set.intersection(gt_set)
    if not common:
        return 0.0   
    precision = len(common) / len(gen_set)
    recall = len(common) / len(gt_set) 
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def calculate_faithfulness(generated: str, retrieved_text: str) -> float:
    sentences = [s.strip() for s in generated.replace("?", ".").replace("!", ".").split(".") if s.strip()]
    if not sentences:
        return 1.0
        
    grounded_count = 0
    for sentence in sentences:
        words = set([w for w in sentence.lower().split() if len(w) > 3])
        if not words:
            grounded_count += 1
            continue
        matches = sum(1 for w in words if w in retrieved_text.lower())
        match_ratio = matches / len(words) if len(words) > 0 else 1.0
        if match_ratio >= 0.4:
            grounded_count += 1     
    return grounded_count / len(sentences)

def evaluate_retrieval_metrics(retrieved_docs: list, keywords: list) -> tuple[float, float, float]:
    if not retrieved_docs:
        return 0.0, 0.0, 0.0
        
    k = len(retrieved_docs)
    relevant_ranks = []
    relevant_count = 0
    covered_keywords = set()
    for rank, doc in enumerate(retrieved_docs, 1):
        content = doc.page_content.lower()
        matched_keywords = {kw.lower() for kw in keywords if kw.lower() in content}
        coverage_ratio = len(matched_keywords) / len(keywords) if keywords else 0.0
        covered_keywords.update(matched_keywords)
        
        if coverage_ratio >= 0.4:
            relevant_ranks.append(rank)
            relevant_count += 1

    hit_rate = 1.0 if relevant_count > 0 else 0.0
    rr = 1.0 / relevant_ranks[0] if relevant_ranks else 0.0
    recall_at_k = len(covered_keywords) / len(keywords) if keywords else 0.0
    precision_at_k = relevant_count / k
    return hit_rate, recall_at_k, rr, precision_at_k

def run_evaluation():
    print("=====")
    print("MEMULAI EVALUASI KUANTITATIF PIPELINE HYBRID RAG (BUKU AI)")
    print("Metrik: Hit Rate, MRR, Precision@K, Faithfulness, ROUGE-L, BERTScore")
    print("=====")
    
    retriever = get_retriever()
    if not retriever:
        print("[WARNING]  Retriever belum diinisialisasi dan database blm diinsisasi.")
        print("Menggunakan mock retriever (context disuntikkan secara statik untuk tujuan simulasi).")
        mock_contexts = {
            "Apa pengertian dari Kecerdasan Buatan (Artificial Intelligence)?": "Artificial Intelligence atau Kecerdasan Buatan adalah sebuah bidang ilmu komputer yang berfokus pada penciptaan mesin atau sistem komputer yang mampu meniru kemampuan intelektual dan cara bertindak manusia.",
            "Apa tujuan pengembangan AI?": "Tujuan pengembangan AI adalah untuk menciptakan sistem yang mampu meniru kemampuan intelektual manusia, meningkatkan efisiensi dan produktivitas, serta memecahkan masalah kompleks yang sulit diselesaikan oleh manusia secara manual.",
            "Darimana asal jawaban yang diberikan AI?": "Asal jawaban yang diberikan AI berasal dari data dan informasi yang telah digunakan untuk melatih model bahasa, serta dari basis data hingga big data yang tersedia. Bahan bakar utama kecerdasan buatan (AI) adalah data.",
            "Apa saja jenis-jenis Machine Learning?": "Jenis-jenis Machine Learning meliputi Supervised Learning, Unsupervised Learning, dan Reinforcement Learning. Supervised learning belajar dari data yang dilabeli, unsupervised dari data tanpa label.",
            "Bagaimana etika dan hukum tentang penggunaan Kecerdasan Buatan (AI)?": "Etika dan hukum tentang penggunaan Kecerdasan Buatan (AI) mencakup aspek-aspek seperti responsibilitas pengembang dan pengguna AI, privasi data, keamanan informasi, serta dampak sosial dari penerapan AI.",
            "Apakah Kecerdasan Buatan (AI) berbahaya?": "Kecerdasan Buatan (AI) bisa berbahaya jika tidak digunakan secara bertanggung jawab. Risiko potensial mencakup masalah privasi, bias, dan dampak sosial dari penerapan AI.",
            "Apa itu privasi data?": "Privasi data berarti melindungi informasi pribadi kita, seperti nama, alamat, lokasi, hobi, dan bahkan kebiasaan browsing. Ini penting karena AI sering mengumpulkan data dari pengguna.",
            "Apa itu algoritma?": "Algoritma adalah serangkaian langkah-langkah yang digunakan untuk menyelesaikan suatu masalah atau melakukan tugas tertentu. Dalam konteks AI, algoritma digunakan untuk memproses data.",
            "Apa hubungan kecerdasan buatan (AI) dengan machine learning (ML)?": "Kecerdasan Buatan (AI) adalah bidang yang lebih luas. Machine Learning (ML) adalah salah satu pendekatan utama dalam AI yang fokus pada pengembangan algoritma agar mesin dapat belajar dari data.",
            "Bagaimana cara kecerdasan buatan (AI) belajar dari data?": "AI belajar dari pola. Dengan menggunakan algoritma pembelajaran mesin, AI menganalisis data yang diberikan, menemukan pola atau hubungan di dalamnya, dan membuat prediksi.",
            "Bagaimana contoh cara AI belajar": "Ketika kita memasukkan gambar ke dalam sistem AI seperti Teachable Machine, AI melihat kumpulan angka—dalam bentuk piksel, warna, kontras, dan sebagainya."
        }
    else:
        mock_contexts = None    
    llm = get_llm()
    prompt = PromptTemplate(template=ACADEMIC_PROMPT_TEMPLATE, input_variables=["context", "question"])
    results = []
    
    for idx, item in enumerate(TEST_DATASET, 1):
        q = item["question"]
        keywords = item["ground_truth_keywords"]
        gt_answer = item["ground_truth_answer"]
        print(f"\n[Test Case {idx}] Query: '{q}'")
        
        # retrieve context
        if mock_contexts:
            context = mock_contexts.get(q, "Konteks simulasi default untuk: " + q)
            retrieved_docs = [Document(page_content=context)]
        else:
            retrieved_docs = retriever.invoke(q)
            context = "\n\n".join([d.page_content for d in retrieved_docs])  
        print(f"-> Context Length: {len(context)} chars")
        print("---")
        print(f"RETRIVED CONTEXT (TOP {len(retrieved_docs)}) UNTUK QUERY: '{q}'")
        print("\n")
        for d_idx, d in enumerate(retrieved_docs, 1):
            src = d.metadata.get("source", "Modul AI")
            pg = d.metadata.get("page", 0) + 1
            print(f"[Chunk {d_idx}] File: {os.path.basename(src)} (Halaman {pg})")
            print(f"Teks:\n{clean_text(d.page_content)}")
            print("-" * 40)
        print("\n")
        
        # eval retriever + nunjukin hasil retrieve untuk contoh yg dikasih
        hit_rate, recall_k, mrr, prec_k = evaluate_retrieval_metrics(retrieved_docs, keywords)
        formatted_prompt = prompt.format(context=context, chat_history="", question=q)
        try:
            response = llm.invoke(formatted_prompt).content
        except Exception as e:
            print(f"Error predicting: {e}")
            response = "[Error LLM]"  
        print(f"-> Generated Answer: {response[:90]}...")

        # eval generator
        faithfulness = calculate_faithfulness(response, context)
        rouge_l = calculate_rouge_l(response, gt_answer)
        bert_score = calculate_bertscore(response, gt_answer)
        token_f1 = calculate_token_f1(response, gt_answer)
        
        results.append({
            "question": q,
            "hit_rate": hit_rate,
            "recall_k": recall_k,
            "mrr": mrr,
            "precision_k": prec_k,
            "faithfulness": faithfulness,
            "rouge_l": rouge_l,
            "bert_score": bert_score,
            "token_f1": token_f1
        }) 
        print(f"   [Retrieval] Hit Rate: {hit_rate:.2f} | Recall@K: {recall_k:.2f} | MRR: {mrr:.2f} | Precision@K: {prec_k:.2f}")
        print(f"   [Generation] Faithfulness: {faithfulness:.2f} | ROUGE-L (LCS): {rouge_l:.2f} | BERTScore: {bert_score:.2f}")

    avg_hr = sum(r["hit_rate"] for r in results) / len(results)
    avg_recall = sum(r["recall_k"] for r in results) / len(results)
    avg_rr = sum(r["mrr"] for r in results) / len(results)
    avg_prec = sum(r["precision_k"] for r in results) / len(results)
    avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
    avg_rouge = sum(r["rouge_l"] for r in results) / len(results)
    avg_bert = sum(r["bert_score"] for r in results) / len(results)
    avg_f1 = sum(r["token_f1"] for r in results) / len(results)
    
    print("\n" + "=====")
    print("RINGKASAN EVALUASI KUANTITATIF RAG")
    print("=====")
    print("RETRIEVAL EVALUATION:")
    print(f"1. Rata-rata Hit Rate            : {avg_hr:.2%}")
    print(f"2. Rata-rata Recall@K            : {avg_recall:.2%}")
    print(f"3. Rata-rata MRR                 : {avg_rr:.2%}")
    print(f"4. Rata-rata Precision@K         : {avg_prec:.2%}")
    print("\nGENERATION EVALUATION:")
    print(f"4. Rata-rata Faithfulness        : {avg_faithfulness:.2%}")
    print(f"5. Rata-rata ROUGE-L (LCS)       : {avg_rouge:.2%}")
    print(f"6. Rata-rata BERTScore (F1)      : {avg_bert:.2%}")
    print(f"7. Rata-rata Token F1 (vs GT)    : {avg_f1:.2%}")
    print("\n")
    
    import datetime
    if LLM_PROVIDER == "openrouter":
        llm_model = OPENROUTER_MODEL
    else:
        llm_model = OLLAMA_MODEL
        
    if retriever:
        if retriever.__class__.__name__ == "DenseRetriever":
            retrieval_approach = "Dense"
        else:
            retrieval_approach = "Hybrid Retrieval"
        top_k = getattr(retriever, "top_k", 5)
    else:
        retrieval_approach = "Mock Retriever (Context Static / DB Kosong)"
        top_k = 1
        
    log_entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "configuration": {
            "llm_provider": LLM_PROVIDER,
            "llm_model": llm_model,
            "retrieval_approach": retrieval_approach,
            "top_k": top_k,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "embedding_model": EMBEDDING_MODEL_NAME
        },
        "metrics": {
            "avg_hit_rate": round(avg_hr, 4),
            "avg_recall_k": round(avg_recall, 4),
            "avg_mrr": round(avg_rr, 4),
            "avg_precision_k": round(avg_prec, 4),
            "avg_faithfulness": round(avg_faithfulness, 4),
            "avg_rouge_l": round(avg_rouge, 4),
            "avg_bertscore": round(avg_bert, 4),
            "avg_token_f1": round(avg_f1, 4)
        }
    }
    
    log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_history.json")
    history = []
    if os.path.exists(log_file_path):
        try:
            with open(log_file_path, "r", encoding="utf-8") as f:
                history = json.load(f)
                if not isinstance(history, list):
                    history = []
        except Exception:
            history = []
            
    history.append(log_entry)
    
    try:
        with open(log_file_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4, ensure_ascii=False)
        print(f"[LOG] Hasil evaluasi dan konfigurasi berhasil disimpan ke: {log_file_path}")
    except Exception as e:
        print(f"[ERROR] Gagal menyimpan log sejarah evaluasi: {e}")
        
if __name__ == "__main__":
    run_evaluation()