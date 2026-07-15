# Sistema RAG de arXiv Paper Abstracts

Este proyecto es la implementación final para el examen de Recuperación de Información. Consiste en un sistema RAG (Retrieval-Augmented Generation) especializado en buscar e indexar resúmenes de artículos científicos (papers) alojados en arXiv, procesando el lenguaje natural para responder consultas concretas de investigación.

El sistema fue optimizado para ejecutarse en la nube utilizando Streamlit Community Cloud.

## Tecnologías Utilizadas
- **Python 3.10+**
- **LangChain & Google Gemini (Flash 2.5):** Generación de respuestas y control estricto de contexto (Evitar alucinaciones).
- **Sentence-Transformers (HuggingFace):** Generación de representaciones vectoriales (embeddings) de forma 100% local, sin consumir cuota de API.
- **ChromaDB:** Base de datos vectorial persistente.
- **KaggleHub:** Descarga dinámica del dataset.
- **Streamlit:** Interfaz gráfica tipo Chat.

---

## 🛠️ Instrucciones de Instalación Local

Para ejecutar el sistema RAG en tu máquina local, sigue estos pasos:

### 1. Clonar el repositorio
```bash
git clone <URL_DE_TU_REPOSITORIO_GITHUB>
cd Exam_RI
```

### 2. Crear y activar un entorno virtual
Se recomienda usar un entorno virtual para no tener conflictos de paquetes.
```bash
python -m venv venv
```
Activa el entorno:
- **En Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **En Linux/Mac:**
  ```bash
  source venv/bin/activate
  ```

### 3. Instalar las dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar Variables de Entorno
Crea un archivo llamado `.env` en la raíz de tu proyecto. Ábrelo y agrega tu token o clave (API KEY) de Google Studio:
```env
GOOGLE_API_KEY="AIzaSy... (tu-api-key-aqui)"
```

### 5. Preparar el Corpus (Descarga de Datos e Indexación)
Debes procesar y descargar el corpus antes de poder hacer consultas. (Asegúrate de que KaggleHub tenga acceso, no suele requerir API, descarga directamente open datasets).
```bash
# 1. Procesa y junta el dataset
python src/data_processing.py

# 2. Crea los embeddings y los guarda en ChromaDB localmente
# Nota: La primera vez que lo corras, descargará el modelo all-MiniLM-L6-v2 de HuggingFace.
python src/index_corpus.py
```

### 6. Evaluación Cuantitativa del Sistema
Para calcular las métricas estándar de Recuperación de Información (Precision@k, Recall@k, NDCG@k) en base al corpus actual:
```bash
# 1. Genera un set de pruebas (Ground Truth sintético aleatorio)
python src/generate_qrels.py

# 2. Ejecuta la evaluación matemática
python src/evaluate.py
```

### 6. Iniciar la Interfaz Web (Streamlit)
Una vez que el proceso de indexación termina, puedes levantar la página web.
```bash
streamlit run src/app.py
```

---

## 🚀 Despliegue en Streamlit Cloud

Para desplegar esta aplicación en Streamlit Cloud:
1. Sube este repositorio a **GitHub**.
2. Entra a [share.streamlit.io](https://share.streamlit.io/).
3. Conecta tu repositorio de GitHub y selecciona el archivo de entrada: `src/app.py`.
4. En **Advanced Settings** de Streamlit, debes registrar la variable de entorno `GOOGLE_API_KEY` con su respectivo valor (secrets).
5. Haz clic en **Deploy**.
