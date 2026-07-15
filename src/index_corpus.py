"""
Script para generar embeddings del corpus procesado e indexarlos en ChromaDB.
"""
import json
import os
import sys
from tqdm import tqdm

# Aseguramos que la raíz del proyecto esté en el path de Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embeddings import TextEmbedder
from src.vector_db import VectorDB
from dotenv import load_dotenv

CORPUS_PATH = "data/processed/corpus.json"

def main():
    load_dotenv()
    
    if not os.path.exists(CORPUS_PATH):
        print(f"No se encontró el corpus en {CORPUS_PATH}. Asegúrate de correr data_processing.py primero.")
        return

    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
        
    if not corpus:
        print("El corpus está vacío.")
        return
        
    print(f"Cargando modelo Gemini y VectorDB...")
    embedder = TextEmbedder()
    db = VectorDB()
    
    print(f"Indexando {len(corpus)} elementos...")
    
    batch_size = 100
    for i in tqdm(range(0, len(corpus), batch_size), desc="Generando e indexando batches"):
        batch = corpus[i:i + batch_size]
        
        ids = []
        documents = []
        metadatas = []
        texts_to_embed = []
        
        for item in batch:
            ids.append(item['id'])
            # El texto completo se usará tanto para embedear como documento a recuperar
            full_text = item['text']
            documents.append(full_text)
            texts_to_embed.append(full_text)
            metadatas.append({
                "title": item.get('title', ''),
                "topics": item.get('topics', ''),
                "url": item.get('url', '')
            })
            
        # Generamos embeddings de texto usando Gemini
        embeddings = embedder.embed_text(texts_to_embed)
        
        db.index_documents(ids, embeddings, documents, metadatas)
        
    print("¡Indexación completada exitosamente!")

if __name__ == "__main__":
    main()
