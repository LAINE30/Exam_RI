"""
Módulo para la preparación del corpus (Requerimiento A).
Descarga el dataset arXiv Paper Abstracts usando kagglehub y procesa los datos.
"""
import os
import json
import kagglehub
import pandas as pd
from tqdm import tqdm

# Configuración
OUTPUT_CORPUS = "data/processed/corpus.json"
LIMIT_PAPERS = 2000  # Límite para indexado rápido en evaluación

def setup_directories():
    """Asegura que los directorios necesarios existan."""
    os.makedirs(os.path.dirname(OUTPUT_CORPUS), exist_ok=True)

def process_arxiv_data(df: pd.DataFrame, limit: int = LIMIT_PAPERS):
    """Extrae la información relevante de los papers."""
    corpus = []
    
    # Algunas versiones de datasets tienen nombres variables
    col_title = 'title' if 'title' in df.columns else 'titles' if 'titles' in df.columns else df.columns[0]
    col_abstract = 'abstract' if 'abstract' in df.columns else 'summary' if 'summary' in df.columns else df.columns[1]
    col_terms = 'terms' if 'terms' in df.columns else 'categories' if 'categories' in df.columns else None
    col_url = 'url' if 'url' in df.columns else 'id' if 'id' in df.columns else None

    # Iterar sobre las filas
    for index, row in tqdm(df.head(limit).iterrows(), total=min(limit, len(df)), desc="Procesando papers"):
        title = str(row.get(col_title, '')).strip()
        abstract = str(row.get(col_abstract, '')).strip()
        
        if not title or not abstract or title == 'nan' or abstract == 'nan':
            continue
            
        full_text = f"Title: {title}\nAbstract: {abstract}"
        
        paper_id = f"arxiv_{index}"
        if col_url and str(row.get(col_url, '')) != 'nan':
            paper_id = str(row[col_url])
            
        topics = str(row.get(col_terms, 'unknown')) if col_terms else 'unknown'
        
        corpus.append({
            "id": paper_id,
            "title": title,
            "text": full_text,
            "topics": topics,
            "url": paper_id if str(paper_id).startswith('http') else f"https://arxiv.org/abs/{paper_id}"
        })
        
    return corpus

def main():
    setup_directories()
    
    print("Descargando dataset de arXiv usando kagglehub...")
    path = kagglehub.dataset_download("spsayakpaul/arxiv-paper-abstracts")
    print("Path to dataset files:", path)
    
    # Buscar el archivo CSV en la carpeta descargada
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("No se encontró ningún archivo .csv en el dataset descargado.")
        
    csv_path = os.path.join(path, csv_files[0])
    print(f"Leyendo datos desde: {csv_path}")
    
    # Cargar CSV
    df = pd.read_csv(csv_path)
    print(f"Total de papers en el dataset: {len(df)}")
    
    # Procesar
    corpus = process_arxiv_data(df, limit=LIMIT_PAPERS)
    
    # Guardar corpus procesado
    with open(OUTPUT_CORPUS, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=4)
        
    print(f"\\nProcesamiento completado. Corpus guardado en {OUTPUT_CORPUS} con {len(corpus)} papers.")

if __name__ == "__main__":
    main()
