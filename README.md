# 📊 AI Financial Report Analyzer (RAG System)

A state-of-the-art **Retrieval-Augmented Generation (RAG)** application designed to analyze complex financial documents. This project transforms a basic Python notebook into a premium, interactive dashboard that provides expert-level financial insights.

---

## 🛠️ Project Workflow

The application follows a modular RAG pipeline to ensure accuracy and context-aware responses:

```mermaid
graph TD
    A[Upload PDF] --> B[Text Extraction - PyMuPDF]
    B --> C[Text Chunking - LangChain]
    C --> D[Embeddings Creation - all-MiniLM-L6-v2]
    D --> E[Vector Storage - FAISS]
    E --> F[User Question]
    F --> G[Semantic Retrieval - Context Search]
    G --> H[LLM Generation - FLAN-T5]
    H --> I[Detailed Financial Answer]
```

1.  **Ingestion**: Extracts raw text from uploaded financial reports.
2.  **Processing**: Splits text into 1,000-character chunks with overlap to maintain context.
3.  **Indexing**: Converts chunks into vector embeddings and stores them in a local FAISS database.
4.  **Retrieval**: Performs a semantic search to find the most relevant document sections for any user query.
5.  **Inference**: A professional Financial Analyst prompt guides the LLM to explain concepts, formulas, and results.

---

## ✨ Features
- **Dynamic PDF Support**: Upload and process any PDF report in real-time.
- **Expert Analyst Mode**: AI provides definitions, formulas, and localized results.
- **Premium Dashboard**: Professional UI inspired by the "Payout" design system.
- **Privacy First**: Runs 100% locally on your machine—no data leaves your computer.
- **Explainable AI**: View the exact document references used for every answer.

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
    - Upload a financial report via the sidebar.
    - Click **Process Analytics**.
    - Ask questions like: *"What is the debt to equity ratio?"* or *"Analyze the revenue growth."*

---

## 📁 Project Structure
- `app.py`: High-performance dashboard built with Streamlit.
- `utils.py`: The core RAG intelligence engine.
- `requirements.txt`: Project dependencies.
- `temp_uploads/`: Secure local storage for your analysis session.
