import os
import sys
import json
import datetime
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import types
m = types.ModuleType("langchain_community.chat_models.vertexai")
m.ChatVertexAI = object
sys.modules["langchain_community.chat_models.vertexai"] = m
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
except ModuleNotFoundError as e:
    print(f"[Error] Dependensi belum lengkap: {e}")
    print("install dl, pip install ragas datasets")
    sys.exit(1)

from rag import (
    get_retriever, get_embeddings, get_llm, ACADEMIC_PROMPT_TEMPLATE,
    LLM_PROVIDER, OPENROUTER_MODEL, OLLAMA_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME,
    OPENROUTER_API_KEY
)

RAGAS_MAX_TOKENS = int(os.getenv("RAGAS_MAX_TOKENS", "4096"))

def run_ragas_evaluation():
    print("=====")
    print("MEMULAI EVALUASI PIPELINE RAG MENGGUNAKAN RAGAS FRAMEWORK")
    print("\n")

    groundtruth_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "groundtruth.json")
    if not os.path.exists(groundtruth_path):
        print(f"[ERROR] Ground truth tidak ditemukan di: {groundtruth_path}")
        return
        
    try:
        with open(groundtruth_path, "r", encoding="utf-8") as f:
            test_dataset = json.load(f)
        print(f"Sukses memuat {len(test_dataset)} sampel uji dari groundtruth.json")
    except Exception as e:
        print(f"[ERROR] Gagal membaca groundtruth.json: {e}")
        return
    print("\n[1/3] Menginisialisasi Retriever dan LLM...")
    retriever = get_retriever()
    if not retriever:
        print("[ERROR] Retriever gagal dimuat. Jalankan ingest terlebih dahulu.")
        return
        
    llm = get_llm()
    print("\n[2/3] Generate Jawaban dari Pipeline RAG...")
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    from langchain_core.prompts import PromptTemplate
    prompt = PromptTemplate(template=ACADEMIC_PROMPT_TEMPLATE, input_variables=["context", "chat_history", "question"])
    
    for idx, item in enumerate(test_dataset, 1):
        q = item["question"]
        gt = item["ground_truth_answer"]
        print(f"[{idx}/{len(test_dataset)}] Memproses Query: '{q}'")
        retrieved_docs = retriever.invoke(q)
        ctx_list = [doc.page_content for doc in retrieved_docs]
        joined_context = "\n\n".join(ctx_list)
        formatted_prompt = prompt.format(context=joined_context, chat_history="", question=q)
        try:
            ans = llm.invoke(formatted_prompt).content
        except Exception as e:
            print(f"   [Error LLM] Gagal generate jawaban: {e}")
            ans = "[Error LLM]"
            
        questions.append(q)
        answers.append(ans)
        contexts.append(ctx_list)
        ground_truths.append(gt)

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)
    
    print("\n[3/3] Menghubungkan dengan LLM & Embedding aktif...")
    if LLM_PROVIDER == "openrouter" and OPENROUTER_API_KEY:
        evaluator_base_llm = get_llm(max_tokens=RAGAS_MAX_TOKENS, streaming=False)
    else:
        evaluator_base_llm = get_llm(streaming=False)

    evaluator_llm = LangchainLLMWrapper(evaluator_base_llm, is_finished_parser=lambda x: True)
    embeddings = get_embeddings()
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    print("\nEvaluating RAGAS...")
    try:
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings
        )
        
        print("\n======")
        print("HASIL METRIK EVALUASI RAGAS:")
        print("\n")
        for key, value in result._repr_dict.items():
            print(f"- {key:25} : {value:.4f}")

        if LLM_PROVIDER == "openrouter":
            active_model = OPENROUTER_MODEL
        else:
            active_model = OLLAMA_MODEL
        if retriever.__class__.__name__ == "DenseRetriever":
            retrieval_approach = "Dense"
        else:
            retrieval_approach = "Hybrid"
            
        log_entry = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "configuration": {
                "llm_provider": LLM_PROVIDER,
                "llm_model": active_model,
                "retrieval_approach": retrieval_approach,
                "top_k": getattr(retriever, "top_k", 5),
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "embedding_model": EMBEDDING_MODEL_NAME
            },
            "ragas_metrics": {key: round(value, 4) for key, value in result._repr_dict.items()}
        }
        
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ragas_evaluation_history.json")
        history = []
        if os.path.exists(log_path):
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
                    if not isinstance(history, list):
                        history = []
            except Exception:
                history = []
                
        history.append(log_entry)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4, ensure_ascii=False)
        print(f"[LOG] Hasil evaluasi RAGAS berhasil disimpan ke: {log_path}\n")
        
    except Exception as e:
        print(f"[ERROR] Gagal menjalankan proses evaluasi RAGAS: {e}")

if __name__ == "__main__":
    run_ragas_evaluation()
