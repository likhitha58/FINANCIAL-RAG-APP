import streamlit as st
import os
from utils import extract_text_from_pdf, chunk_text, create_vector_store, load_llm, get_answer
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Financial Report Analyzer",
    page_icon="📊",
    layout="wide"
)

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    /* Main Background and Text Color */
    .stApp {
        background-color: #f8f9fa;
        color: #1a202c; /* Clear dark text */
    }
    
    /* Global Text Contrast */
    p, span, label, .stMarkdown {
        color: #1a202c !important;
    }

    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        background-color: #1e3a8a;
        color: white !important; /* Ensure button text is white */
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #d1d5db;
        color: #1a202c !important;
    }

    .header-style {
        font-size: 42px;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 10px;
        text-align: center;
        letter-spacing: -0.5px;
    }
    .subheader-style {
        font-size: 18px;
        color: #4b5563;
        margin-bottom: 40px;
        text-align: center;
        font-weight: 400;
    }
    .source-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #1e3a8a;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        color: #374151 !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] p {
        color: #1e3a8a !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- App Title and Header ---
st.markdown("<div class='header-style'>📊 AI Financial Report Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader-style'>Upload your financial PDF and ask questions to get instant AI-powered insights</div>", unsafe_allow_html=True)

# --- Initialize Session State ---
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'llm' not in st.session_state:
    with st.spinner("Loading AI Brain... Please wait."):
        st.session_state.llm = load_llm()
if 'history' not in st.session_state:
    st.session_state.history = []

# --- Sidebar: File Upload ---
with st.sidebar:
    st.title("📂 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
    
    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Processing PDF..."):
                # Save temp file
                temp_dir = "temp_uploads"
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)
                
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # RAG Pipeline
                text = extract_text_from_pdf(file_path)
                chunks = chunk_text(text)
                st.session_state.vector_store = create_vector_store(chunks)
                st.success("✅ Document processed successfully!")
                st.info(f"Splits: {len(chunks)} chunks created.")

# --- Main Interface ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Ask Your Question")
    user_query = st.text_input("Example: What is the company's total revenue for 2024?", key="user_input")
    
    if st.button("Get Answer"):
        if st.session_state.vector_store is not None:
            if user_query:
                with st.spinner("Analyzing document..."):
                    start_time = time.time()
                    answer, sources = get_answer(user_query, st.session_state.vector_store, st.session_state.llm)
                    end_time = time.time()
                    
                    st.markdown("### 🤖 Answer")
                    st.write(answer)
                    st.caption(f"Generated in {round(end_time - start_time, 2)} seconds")
                    
                    st.session_state.history.append({"q": user_query, "a": answer})
                    
                    # Show Sources
                    with st.expander("🔍 View Source Relevant Context"):
                        for i, doc in enumerate(sources):
                            st.markdown(f"<div class='source-box'><strong>Source {i+1}:</strong><br>{doc.page_content}</div>", unsafe_allow_html=True)
            else:
                st.warning("Please enter a question.")
        else:
            st.error("Please upload and process a PDF document first!")

with col2:
    st.subheader("📜 Question History")
    for item in reversed(st.session_state.history):
        st.write(f"**Q:** {item['q']}")
        st.write(f"**A:** {item['a']}")
        st.markdown("---")

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Built with Streamlit & LangChain (No API keys required)</p>", unsafe_allow_html=True)
