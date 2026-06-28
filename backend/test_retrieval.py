import os
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import torch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag import get_retriever, clean_text

def main():
    print("=====")
    print("Hybrid Retrieval")
    print("\n")
    
    retriever = get_retriever()
    if not retriever:
        print("[ERROR] Database isn't available. Make sure the database has beeen built")
        sys.exit(1)
        
    print("Database is successfully loaded")
    print("Put or type your query here to test the retrieval and see the retrieved contexts")
    print("Type 'exit' to quit the program")
    
    while True:
        try:
            query = input("\nQuery input: ")
            if not query.strip():
                continue
            if query.strip().lower() == "exit":
                print("Exits.")
                break
                
            print(f"\nFind top-k contexts for: '{query}'...\n")

            retrieved_docs = retriever.invoke(query)
            print("=====")
            print(f"Retrieved Contexts ({len(retrieved_docs)} is found):")
            print("\n")
            
            for idx, doc in enumerate(retrieved_docs, 1):
                source_path = doc.metadata.get("source", "Modul AI")
                filename = os.path.basename(source_path)
                page = doc.metadata.get("page", 0) + 1
                
                print(f"[DOKUMEN {idx}]")
                print(f"File   : {filename}")
                print(f"Halaman: {page}")
                print("Retrieved Context: \n")
                print(clean_text(doc.page_content))
                print("\n")    
        except (KeyboardInterrupt, EOFError):
            print("\nKeluar dari program.")
            break

if __name__ == "__main__":
    main()