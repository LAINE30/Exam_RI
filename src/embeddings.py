"""
Módulo para la generación de embeddings de texto usando Google Gemini.
Requerimiento B.
"""
import os
from typing import List, Union
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class TextEmbedder:
    """
    Clase para manejar la generación de embeddings de texto con Google Gemini.
    """
    def __init__(self, model_name: str = "models/text-embedding-004"):
        """
        Inicializa el modelo de embeddings de Gemini.
        """
        print(f"Inicializando modelo de embeddings de Gemini ({model_name})...")
        
        # Validar que exista la API key en el entorno
        if not os.environ.get("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY no encontrada en las variables de entorno.")
            
        self.embeddings_model = GoogleGenerativeAIEmbeddings(model=model_name)
        
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
            
        # Generar embeddings usando la API de Gemini
        embeddings = self.embeddings_model.embed_documents(texts)
        return embeddings

