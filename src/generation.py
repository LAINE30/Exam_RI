"""
Módulo para la generación de respuestas usando un LLM (Retrieval-Augmented Generation).
Requerimiento E.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from typing import List, Dict, Any

class RAGGenerator:
    """
    Clase que maneja la formulación del contexto y la generación de la respuesta
    conversacional usando un modelo de lenguaje.
    """
    def __init__(self, model_name: str = "gemini-2.5-flash", temperature: float = 0.0):
        """
        Inicializa el modelo de lenguaje de LangChain (Gemini).
        Nota: Requiere que la variable de entorno GOOGLE_API_KEY esté configurada.
        Temperature a 0.0 para evitar alucinaciones.
        """
        # Si no hay API key, inicializar fallará al intentar invocar
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        
        # Definir el template del RAG
        prompt_template = """
        Eres un asistente académico experto. Utiliza ÚNICAMENTE la siguiente información 
        recuperada de resúmenes de artículos científicos (contexto) para responder a la consulta del usuario.
        
        Reglas Estrictas:
        1. Responde basándote exclusivamente en la información provista en el contexto.
        2. Si la respuesta a la consulta del usuario NO se encuentra en el contexto, DEBES indicar 
           explícitamente que el corpus no contiene información suficiente para responder la consulta. 
           Bajo ninguna circunstancia inventes o deduzcas información externa.
        3. Si usas información de un artículo, cita su título sutilmente para respaldar tu respuesta.
        
        Contexto Recuperado:
        {context}
        
        Consulta del Usuario: {question}
        
        Respuesta:
        """
        
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template
        )
        
        self.chain = self.prompt | self.llm

    def format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Formatea los documentos recuperados en un solo texto para pasarlo al LLM.
        """
        if not retrieved_docs:
            return "No se recuperaron documentos relevantes del corpus."
            
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            title = doc.get('title', 'Documento sin título')
            text = doc.get('text', '')
            context_parts.append(f"Documento {i}:\n- Título: {title}\n- Contenido: {text}")
            
        return "\n\n".join(context_parts)

    def generate_response(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Genera la respuesta final usando el contexto recuperado.
        
        Args:
            query: La consulta original del usuario.
            retrieved_docs: Los resultados devueltos por TextRetriever.
            
        Returns:
            La respuesta generada por el LLM en formato de texto.
        """
        context_str = self.format_context(retrieved_docs)
        
        try:
            response = self.chain.invoke({
                "context": context_str, 
                "question": query
            })
            return response.content
        except Exception as e:
            return f"Error generando la respuesta: {e}"
