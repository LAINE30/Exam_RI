import json
import os
import random

def generate_qrels():
    corpus_path = "data/processed/corpus.json"
    qrels_dir = "data/evaluation"
    qrels_path = os.path.join(qrels_dir, "qrels.json")
    
    os.makedirs(qrels_dir, exist_ok=True)
    
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
        
    random.seed(42)
    sample_papers = random.sample(corpus, 5)
    
    queries = []
    for i, paper in enumerate(sample_papers):
        queries.append({
            "query_id": f"q{i+1}",
            "query_text": paper["title"],
            "relevant_doc_ids": [paper["id"]]
        })
        
    with open(qrels_path, "w", encoding="utf-8") as f:
        json.dump({"queries": queries}, f, indent=4)
        
    print(f"Generado {qrels_path} con {len(queries)} consultas de prueba.")

if __name__ == "__main__":
    generate_qrels()
