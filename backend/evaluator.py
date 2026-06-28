import os
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag import get_retriever, get_embeddings, clean_text
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from rag import get_llm, ACADEMIC_PROMPT_TEMPLATE

BERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
try:
    print(f"Menginisialisasi model token-embedding untuk BERTScore ({BERT_MODEL_NAME})...")
    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME)
except Exception as e:
    print(f"[WARNING] Gagal memuat model BERTScore secara otomatis: {e}")
    bert_tokenizer = None
    bert_model = None

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
    for rank, doc in enumerate(retrieved_docs, 1):
        is_relevant = False
        content = doc.page_content.lower()
        matches = sum(1 for kw in keywords if kw.lower() in content)
        match_ratio = matches / len(keywords) if keywords else 1.0
        
        if match_ratio >= 0.4:
            is_relevant = True
            relevant_ranks.append(rank)
            relevant_count += 1

    hit_rate = 1.0 if relevant_count > 0 else 0.0
    mrr = 1.0 / relevant_ranks[0] if relevant_ranks else 0.0
    precision_at_k = relevant_count / k
    return hit_rate, mrr, precision_at_k

def run_evaluation():
    print("=====")
    print("MEMULAI EVALUASI KUANTITATIF PIPELINE HYBRID RAG (BUKU AI)")
    print("Metrik: Hit Rate, MRR, Precision@K, Faithfulness, ROUGE-L, BERTScore")
    print("=====")
    
    retriever = get_retriever()
    if not retriever:
        print("[WARNING] Hybrid retriever belum diinisialisasi. (Database kosong)")
        print("Menggunakan mock retriever (context disuntikkan secara statik untuk tujuan simulasi).")
        mock_contexts = {
            "Apa pengertian dari Kecerdasan Buatan (Artificial Intelligence)?": "Kecerdasan buatan atau Artificial Intelligence (AI) didefinisikan sebagai sistem komputer atau mesin yang mensimulasikan kecerdasan manusia agar mampu melakukan proses berpikir, belajar dari pengalaman, penalaran logis, dan penyesuaian diri.",
            "Apa perbedaan antara Machine Learning dan Deep Learning?": "Pembelajaran Mesin atau Machine Learning adalah ilmu komputer yang fokus pada pengembangan algoritma agar dapat belajar mandiri dari kumpulan data. Sedangkan Deep Learning adalah bagian dari machine learning yang terinspirasi oleh struktur otak manusia dengan menggunakan jaringan saraf tiruan (artificial neural networks) multi-layer untuk menangani data besar.",
            "Sebutkan dampak negatif atau tantangan dari penggunaan teknologi AI.": "Tantangan dan dampak negatif dari teknologi AI meliputi masalah kebocoran privasi data pribadi, munculnya keputusan bias karena bias data latih, penyebaran berita bohong atau hoaks yang mudah dibuat oleh generatif AI, serta meningkatnya otomatisasi yang memicu ancaman pengangguran.",
            "Bagaimana cara kerja jaringan saraf tiruan secara sederhana?": "Jaringan Saraf Tiruan (JST) dirancang meniru kerja neuron otak manusia. Cara kerjanya dimulai dari data input yang diterima neuron input, diteruskan ke hidden layer, di mana data dikalikan dengan bobot (weight) ditambah nilai bias, lalu diaktifkan oleh fungsi aktivasi untuk menghasilkan output prediksi akhir.",
            "Mengapa data berkualitas sangat penting dalam melatih model AI?": "Keberhasilan model AI sangat bertumpu pada kualitas data latihnya. Menggunakan data berkualitas rendah atau bias akan menyebabkan model menghasilkan keputusan salah atau bias. Istilah terkenalnya adalah 'garbage in, garbage out' yang artinya jika data latih buruk maka prediksi AI juga buruk."
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
            context = mock_contexts[q]
            retrieved_docs = [Document(page_content=context)]
        else:
            retrieved_docs = retriever.invoke(q)
            context = "\n\n".join([d.page_content for d in retrieved_docs])  
        print(f"-> Context Length: {len(context)} chars")
        print("=====")
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
        hit_rate, mrr, prec_k = evaluate_retrieval_metrics(retrieved_docs, keywords)
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
            "mrr": mrr,
            "precision_k": prec_k,
            "faithfulness": faithfulness,
            "rouge_l": rouge_l,
            "bert_score": bert_score,
            "token_f1": token_f1
        }) 
        print(f"   [Retrieval] Hit Rate: {hit_rate:.2f} | MRR: {mrr:.2f} | Precision@K: {prec_k:.2f}")
        print(f"   [Generation] Faithfulness: {faithfulness:.2f} | ROUGE-L (LCS): {rouge_l:.2f} | BERTScore: {bert_score:.2f}")

    avg_hr = sum(r["hit_rate"] for r in results) / len(results)
    avg_mrr = sum(r["mrr"] for r in results) / len(results)
    avg_prec = sum(r["precision_k"] for r in results) / len(results)
    avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
    avg_rouge = sum(r["rouge_l"] for r in results) / len(results)
    avg_bert = sum(r["bert_score"] for r in results) / len(results)
    avg_f1 = sum(r["token_f1"] for r in results) / len(results)
    
    print("\n" + "=====")
    print("RINGKASAN EVALUASI KUANTITATIF HYBRID RAG")
    print("=====")
    print("RETRIEVAL EVALUATION:")
    print(f"1. Rata-rata Hit Rate (Recall@K) : {avg_hr:.2%}")
    print(f"2. Rata-rata MRR                 : {avg_mrr:.2%}")
    print(f"3. Rata-rata Precision@K         : {avg_prec:.2%}")
    print("\nGENERATION EVALUATION:")
    print(f"4. Rata-rata Faithfulness        : {avg_faithfulness:.2%}")
    print(f"5. Rata-rata ROUGE-L (LCS)       : {avg_rouge:.2%}")
    print(f"6. Rata-rata BERTScore (F1)      : {avg_bert:.2%}")
    print(f"7. Rata-rata Token F1 (vs GT)    : {avg_f1:.2%}")
    print("\n")
    
if __name__ == "__main__":
    run_evaluation()