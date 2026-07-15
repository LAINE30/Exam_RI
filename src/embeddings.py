"""
Módulo para la generación de embeddings de texto localmente.
Requerimiento B.
"""
from typing import List, Union
from sentence_transformers import SentenceTransformer

class TextEmbedder:
    """
    Clase para manejar la generación de embeddings de texto de forma 100% local.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Inicializa el modelo de embeddings local (se descargará la primera vez).
        """
        print(f"Inicializando modelo de embeddings local ({model_name})...")
        self.model = SentenceTransformer(model_name)
        
    def embed_text(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Genera embeddings para uno o múltiples textos.
        
        Args:
            texts: String único o lista de strings.
            
        Returns:
            Lista de listas de floats (los embeddings).
        """
        if isinstance(texts, str):
            texts = [texts]
            
        # Generar embeddings localmente
        embeddings = self.model.encode(texts).tolist()
        return embeddings

