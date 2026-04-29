# 📊 Payout AI - Financial Intelligence System

A state-of-the-art **Retrieval-Augmented Generation (RAG)** application designed to analyze complex financial documents. This project transforms a basic Python notebook into a premium, interactive dashboard ("Payout AI") that provides expert-level financial insights.

---

## 🛠️ Project Workflow

The application follows a modular RAG pipeline to ensure accuracy and context-aware responses:

### High-Level Architecture
```mermaid
graph TD
    UI[Streamlit Frontend - Payout AI] -->|PDF Upload| Extractor[PyMuPDF Text Extractor]
    Extractor --> Chunker[LangChain Text Splitter]
    Chunker --> Embedder[Sentence-Transformers all-MiniLM-L6-v2]
    Embedder --> VectorDB[(FAISS Vector Store)]
    
    UI -->|User Query| QueryEmbedder[Query Embedding]
    QueryEmbedder -->|Similarity Search| VectorDB
    VectorDB -->|Top 2 Chunks| ContextBuilder[Context & Prompt Builder]
    ContextBuilder --> LLM[HuggingFace Pipeline: flan-t5-base]
    LLM -->|Text + Citation| UI
```

### Data Flow Diagram
```mermaid
sequenceDiagram
    participant User
    participant App as app.py (Payout AI)
    participant Utils as utils.py
    participant FAISS
    participant LLM as FLAN-T5

    User->>App: Uploads PDF
    App->>Utils: extract_text_from_pdf()
    Utils-->>App: Raw Text
    App->>Utils: chunk_text(text)
    Utils-->>App: Text Chunks
    App->>Utils: create_vector_store(chunks)
    Utils->>FAISS: Create Embeddings & Store
    FAISS-->>App: Vector DB Ready
    
    User->>App: Asks Question
    App->>FAISS: similarity_search(query, k=2)
    FAISS-->>App: Top 2 Relevant Chunks
    App->>Utils: get_answer(query, docs)
    Utils->>LLM: Strict Prompt + Context
    LLM-->>Utils: Exact Answer + Citation
    Utils-->>App: Response Text
    App->>User: Displays Answer & Sources
```

1.  **Ingestion**: Extracts raw text from uploaded financial reports.
2.  **Processing**: Splits text into 1,000-character chunks with overlap to maintain context.
3.  **Indexing**: Converts chunks into vector embeddings and stores them in a local FAISS database.
4.  **Retrieval**: Performs a semantic search to find the most relevant document sections for any user query.
5.  **Inference**: A professional Financial Analyst prompt guides the LLM to explain concepts, formulas, and results.

---

## ✨ Features
- **Dynamic PDF Support**: Upload and process any PDF report in real-time.
- **Strict Citation Engine**: The AI extracts exact numbers and explicitly cites document references for 100% accuracy and zero hallucination.
- **Premium Deep Space UI**: A beautiful, glassmorphism-inspired dark mode dashboard with dynamic hover effects, gradients, and a responsive layout.
- **Interactive Chat History**: Your past queries are saved as interactive buttons in the "Recent Activity" tab—clicking them seamlessly reloads past context and sources.
- **Privacy First**: Runs 100% locally on your machine—no data leaves your computer.

---

## 🚀 Installation & Setup

### 1. Requirements
Ensure you have **Python 3.9+** installed.

### 2. Clone the Repository
```bash
git clone https://github.com/likhitha58/FINANCIAL-RAG-APP.git
cd FINANCIAL-RAG-APP
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏃 How to Run

1.  **Launch the Dashboard**:
    ```bash
    streamlit run app.py
    ```
2.  **Access the App**: Open your browser to `http://localhost:8501`.
3.  **Analyze**:
    - Upload a financial report via the sidebar under **Payout AI**.
    - Click **🚀 Process Analytics**.
    - Ask questions like: *"What is the debt to equity ratio?"* or *"Analyze the revenue growth."*

---

## 📁 Project Structure
- `app.py`: High-performance dashboard built with Streamlit, containing the Payout AI UI.
- `utils.py`: The core RAG intelligence engine managing embeddings, chunking, and LLM inference.
- `requirements.txt`: Project dependencies.
- `temp_uploads/`: Secure local storage for your analysis session.
