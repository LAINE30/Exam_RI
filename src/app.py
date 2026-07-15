"""
Interfaz web principal utilizando Streamlit para el RAG de arXiv.
Requerimientos F y G.
"""
import streamlit as st
import sys
import os
import time

# Añadir el directorio raíz al path para que Python encuentre el paquete 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Cargar variables de entorno (como GOOGLE_API_KEY)
load_dotenv()

# Configuración de página
st.set_page_config(page_title="arXiv RAG Assistant", page_icon="📚", layout="wide")

if "GOOGLE_API_KEY" not in os.environ or not os.environ["GOOGLE_API_KEY"]:
    st.error("⚠️ No se encontró la variable GOOGLE_API_KEY. Por favor, añádela a tu archivo .env o en los secrets de Streamlit.")
    st.stop()

@st.cache_resource
def load_pipeline():
    from src.retrieval import TextRetriever
    from src.generation import RAGGenerator
    retriever = TextRetriever()
    generator = RAGGenerator(model_name="gemini-2.5-flash", temperature=0.0)
    return retriever, generator

with st.spinner("Cargando modelo Gemini y base de datos vectorial..."):
    try:
        retriever, generator = load_pipeline()
    except Exception as e:
        st.error(f"Error cargando el pipeline: {e}")
        st.stop()

# ==========================================
# INTERFAZ PRINCIPAL Y LAYOUT
# ==========================================

st.title("📚 Asistente de Investigación arXiv (RAG)")
st.markdown("Consulta resúmenes de artículos científicos de arXiv utilizando lenguaje natural.")

# Inicializar historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# FUNCIONES AUXILIARES
# ==========================================

def render_evidences(evidences):
    """Renderiza las evidencias (Requerimiento F)."""
    if not evidences:
        st.info("No se encontró información relevante en el corpus.")
        return
    
    st.markdown("##### 📄 Evidencias Recuperadas")
    for idx, ev in enumerate(evidences):
        with st.expander(f"Paper {idx+1}: {ev.get('title', 'Sin título')}"):
            st.markdown(f"**Tópicos:** {ev.get('topics', 'N/A')}")
            st.markdown(f"**URL:** {ev.get('url', 'N/A')}")
            st.markdown("**Resumen:**")
            st.write(ev.get("text", ""))

# ==========================================
# RENDERIZADO DEL HISTORIAL DE CHAT
# ==========================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if msg["role"] == "assistant" and "evidences" in msg and msg["evidences"]:
            render_evidences(msg["evidences"])


# ==========================================
# FLUJO: BÚSQUEDA POR TEXTO
# ==========================================
if prompt := st.chat_input("Ej: What are the main applications of Graph Neural Networks?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en la base de datos de arXiv..."):
            evidences = retriever.retrieve(prompt, top_k=3)
        with st.spinner("Generando respuesta fundamentada..."):
            answer = generator.generate_response(prompt, evidences)
            st.markdown(answer)
            render_evidences(evidences)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer, 
                "evidences": evidences
            })
