import os
import shutil
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag import process_and_store_document, DB_DIR, CHUNKS_PKL_PATH

MODUL_DIR = r"D:\Collage\6th Term\NLP\tbp-004-010\modul"

def main():
    print("-----")
    print("Ingestion Script with Hybrid Retrieval")
    print("-----")
    
    if not os.path.exists(MODUL_DIR):
        print(f"[ERROR] documents folder not found: {MODUL_DIR}")
        print("make sure the folder exists & the documents are placed there.")
        sys.exit(1)
    files = [f for f in os.listdir(MODUL_DIR) if f.endswith(".pdf")]
    
    if not files:
        print(f"[WARNING] no files found in: {MODUL_DIR}")
        sys.exit(0)
    print(f"{len(files)} need to be processed & indexed")
    for idx, f in enumerate(files, 1):
        print(f" {idx}. {f}")
        
    # menghapus db lama dan buat db baru, ini biar kalau ada perubahan (secara keseluruhan) db nya direbuid dari 0
    if os.path.exists(DB_DIR):
        print("\nInitiating database, clean up the old ones")
        try:
            shutil.rmtree(DB_DIR)
            os.makedirs(DB_DIR, exist_ok=True)
            print("Database lama berhasil dibersihkan.")
        except Exception as e:
            print(f"Gagal membersihkan database lama: {e}")
            
    print("\nProceed indexing [model: intfloat/multilingual-e5-base]")
    
    start_time = time.time()
    total_chunks = 0
    for idx, filename in enumerate(files, 1):
        file_path = os.path.join(MODUL_DIR, filename)
        print(f"[{idx}/{len(files)}] Memproses: {filename}...")
        
        file_start = time.time()
        try:
            msg = process_and_store_document(file_path)
            file_end = time.time()
            print(f"  -> Sukses! {msg} (Durasi: {file_end - file_start:.2f} detik)")
        except Exception as e:
            print(f"  -> [ERROR] Gagal memproses {filename}: {e}")
            
    end_time = time.time()
    print("\n")
    print("PROSES INDEXING SELESAI")
    print("-----")
    print(f"Total Dokumen Diproses : {len(files)}")
    print(f"Total Waktu Eksekusi   : {end_time - start_time:.2f} detik")
    print(f"Vector Database disimpan di : {DB_DIR}")
    print("-----")

if __name__ == "__main__":
    main()
