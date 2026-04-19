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

# --- Custom CSS for Payout-inspired Styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: #f1f4f9; 
    }

    /* Global Text Color - Force high contrast */
    [data-testid="stAppViewContainer"] {
        color: #1a202c !important;
    }

    /* Heading Colors */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, h1, h2, h3 {
        color: #1a365d !important;
    }
    
    .stSubheader {
        color: #1a365d !important;
    }
    header {visibility: hidden;}

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    /* Sidebar Styling - Darker text and clear backgrounds */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    
    [data-testid="stSidebar"] .stMarkdown h1, 
    [data-testid="stSidebar"] .stMarkdown h2, 
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: #1a202c !important;
        font-weight: 600 !important;
    }

    /* Card Styling */
    .dashboard-card {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
        margin-bottom: 24px;
        color: #1a202c !important;
    }

    .dashboard-card p, .dashboard-card h3 {
        color: #1a202c !important;
    }

    /* Title Styling */
    .header-text {
        font-size: 36px;
        font-weight: 800;
        color: #1a365d;
        margin-bottom: 8px;
    }
    .subheader-text {
        font-size: 16px;
        color: #2d3748; /* Darker than before */
        margin-bottom: 32px;
        font-weight: 500;
    }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.5em;
        background-color: #1a365d; /* Deeper navy */
        color: white !important;
        font-weight: 600;
        border: none;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #2a4365;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* Input Styling - Fix visibility while typing */
    .stTextInput label {
        color: #1a202c !important;
        font-weight: 600 !important;
    }
    
    /* Target the input element directly and its focus state */
    .stTextInput div[data-baseweb="input"] input {
        color: #1a202c !important;
        -webkit-text-fill-color: #1a202c !important; /* Force for some browsers */
        caret-color: #1a202c !important;
        background-color: #ffffff !important;
    }

    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 1px solid #cbd5e0;
        padding: 12px;
        background-color: #ffffff !important;
        color: #1a202c !important;
    }

    /* Source Box Styling */
    .source-container {
        font-size: 14px;
        color: #1a202c;
        background-color: #f7fafc;
        padding: 18px;
        border-radius: 12px;
        border-left: 5px solid #2b6cb0;
        margin-top: 12px;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.05);
    }

    /* Status & Spinner text */
    .stSpinner>div>div {
        color: #1a365d !important;
    }
    .stAlert p {
        color: #1a202c !important;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Top Header Section ---
st.markdown("<div class='header-text'>Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader-text'>AI Financial Intelligence System</div>", unsafe_allow_html=True)

# --- Initialize Session State ---
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'llm' not in st.session_state:
    with st.spinner("Initializing AI Core..."):
        st.session_state.llm = load_llm()
if 'history' not in st.session_state:
    st.session_state.history = []

# --- Sidebar: File Manager ---
with st.sidebar:
    st.markdown("<h1>📊 Payout AI</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.write("### 📂 Document Center")
    uploaded_file = st.file_uploader("Upload PDF Report", type=['pdf'], label_visibility="collapsed")
    
    if uploaded_file:
        st.info(f"Selected: {uploaded_file.name}")
        if st.button("🚀 Process Analytics"):
            with st.spinner("Analyzing Document..."):
                temp_dir = "temp_uploads"
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)
                
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                text = extract_text_from_pdf(file_path)
                chunks = chunk_text(text)
                st.session_state.vector_store = create_vector_store(chunks)
                st.success("Analysis Ready!")

# --- Main App Content ---
if st.session_state.vector_store is None:
    # Welcome Layout
    st.markdown("""
        <div class='dashboard-card'>
            <h3 style='color: #2d3748;'>Welcome to the Financial Analyzer</h3>
            <p style='color: #718096;'>Please upload a financial report in the sidebar to begin your intelligence session. 
            Once processed, you can ask complex questions about revenue, liabilities, and growth metrics.</p>
        </div>
    """, unsafe_allow_html=True)
else:
    # Analysis Mode
    col_main, col_hist = st.columns([2, 1])

    with col_main:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.subheader("💬 Query Intelligence")
        user_query = st.text_input("Enter your question (e.g., 'What is the debt-to-equity ratio?')", key="query_in")
        
        if st.button("Run AI Inference"):
            if user_query:
                with st.spinner("Generating Insights..."):
                    answer, sources = get_answer(user_query, st.session_state.vector_store, st.session_state.llm)
                    
                    st.markdown("### 🔍 Result")
                    st.write(answer)
                    
                    st.session_state.history.append({"q": user_query, "a": answer})
                    
                    with st.expander("📚 View Supporting Evidence"):
                        for i, doc in enumerate(sources):
                            st.markdown(f"<div class='source-container'><strong>Reference {i+1}:</strong><br>{doc.page_content}</div>", unsafe_allow_html=True)
            else:
                st.warning("Please enter a query.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_hist:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.subheader("📜 Recent Activity")
        if not st.session_state.history:
            st.write("No queries yet.")
        for item in reversed(st.session_state.history[-5:]): # Show last 5
            st.markdown(f"""
                <div style='margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid #edf2f7;'>
                    <div style='font-weight: 600; color: #4a5568;'>Q: {item['q']}</div>
                    <div style='font-size: 14px; color: #718096;'>A: {item['a'][:100]}...</div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# --- Footer ---
