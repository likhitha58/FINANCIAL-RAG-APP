# AI Financial Report Analyzer 📊

A clean, user-friendly RAG (Retrieval-Augmented Generation) system for analyzing financial PDF documents. This project is an improved version of the baseline RAG pipeline, adding a dynamic UI and real-time processing capabilities.

## 🚀 Features
- **Dynamic PDF Upload**: Process any financial report (10-K, 10-Q, etc.) on the fly.
- **Local RAG Pipeline**: Uses LangChain, FAISS, and PyMuPDF.
- **Privacy Focused**: Runs entirely locally using HuggingFace models (no OpenAI/API keys needed).
- **Streamlit UI**: A professional-grade dashboard for document analysis.
- **Source Highlighting**: See exactly which parts of the document contributed to the answer.

## 🛠️ Tech Stack
- **UI**: Streamlit
- **PDF Extraction**: PyMuPDF (fitz)
- **Text Processing**: LangChain
- **Embeddings**: `all-MiniLM-L6-v2` (Sentence-Transformers)
- **LLM**: `flan-t5-base` (Google)
- **Vector DB**: FAISS

## 📥 Installation

1. **Clone/Download** this folder.
2. **Open Terminal** in the `FINANCIAL-RAG-APP` directory.
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃 How to Run

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. The browser will open automatically (usually at `http://localhost:8501`).
3. Upload a PDF from the sidebar.
4. Click **Process Document**.
5. Ask any question about your financial report!

## 📁 Project Structure
- `app.py`: Streamlit frontend and UI logic.
- `utils.py`: Core RAG logic (extraction, chunking, embeddings, LLM generation).
- `requirements.txt`: Python package dependencies.
- `temp_uploads/`: Temporary storage for uploaded PDF files.
