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

# --- Custom CSS for Premium Dark Styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        font-family: 'Outfit', sans-serif;
        background-color: #0f172a; 
        color: #e2e8f0 !important;
    }

    /* Global Text Color - Force high contrast */
    [data-testid="stAppViewContainer"] {
        color: #e2e8f0 !important;
    }

    /* Heading Colors */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, h1, h2, h3 {
        color: #f8fafc !important;
        font-weight: 600;
    }
    
    .stSubheader {
        color: #f8fafc !important;
    }
    header {visibility: hidden;}

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid #334155;
    }
    
    [data-testid="stSidebar"] .stMarkdown h1, 
    [data-testid="stSidebar"] .stMarkdown h2, 
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
    }

    /* Card Styling */
    .dashboard-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        border: 1px solid #334155;
        margin-bottom: 24px;
        color: #e2e8f0 !important;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }

    .dashboard-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 35px -10px rgba(0, 0, 0, 0.6);
        border-color: #8b5cf6;
    }

    .dashboard-card p, .dashboard-card h3 {
        color: #e2e8f0 !important;
    }

    /* Title Styling */
    .header-text {
        font-size: 42px;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .subheader-text {
        font-size: 18px;
        color: #94a3b8;
        margin-bottom: 32px;
        font-weight: 400;
        letter-spacing: 0.5px;
    }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.5em;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white !important;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
        transform: translateY(-1px);
        color: white !important;
    }

    /* Input Styling */
    .stTextInput label {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
    }
    
    .stTextInput div[data-baseweb="input"] {
        background-color: #0f172a !important;
        border-radius: 12px;
        border: 1px solid #334155;
        transition: all 0.3s ease;
    }

    .stTextInput div[data-baseweb="input"]:focus-within {
        border-color: #8b5cf6;
        box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.2);
    }

    .stTextInput div[data-baseweb="input"] input {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        caret-color: #8b5cf6 !important;
        background-color: transparent !important;
        padding: 14px;
    }

    /* Source Box Styling */
    .source-container {
        font-size: 14px;
        color: #cbd5e1;
        background-color: #1e293b;
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #8b5cf6;
        margin-top: 16px;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Status & Spinner text */
    .stSpinner>div>div {
        color: #8b5cf6 !important;
    }
    .stAlert p {
        color: #e2e8f0 !important;
        font-weight: 500;
    }
    .stAlert {
        background-color: rgba(139, 92, 246, 0.1) !important;
        border: 1px solid rgba(139, 92, 246, 0.2) !important;
        border-radius: 12px;
    }
    
    /* File uploader styling */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #334155 !important;
        border-radius: 16px !important;
        background-color: #0f172a !important;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #8b5cf6 !important;
        background-color: rgba(139, 92, 246, 0.05) !important;
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
if 'active_chat_idx' not in st.session_state:
    st.session_state.active_chat_idx = None

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
            <h3 style='color: #f8fafc; margin-bottom: 16px;'>Welcome to the Financial Analyzer</h3>
            <p style='color: #94a3b8; line-height: 1.6;'>Please upload a financial report in the sidebar to begin your intelligence session. 
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
                    
                    # Results are now displayed by the active chat section below
                    
                    st.session_state.history.append({"q": user_query, "a": answer, "sources": sources})
                    st.session_state.active_chat_idx = len(st.session_state.history) - 1
            else:
                st.warning("Please enter a query.")
                
        # Display the active chat (either just generated or selected from history)
        if st.session_state.active_chat_idx is not None and st.session_state.active_chat_idx < len(st.session_state.history):
            active_item = st.session_state.history[st.session_state.active_chat_idx]
            st.markdown("### 🔍 Result")
            st.markdown(f"**Q:** {active_item['q']}")
            st.write(active_item['a'])
            
            if 'sources' in active_item:
                with st.expander("📚 View Supporting Evidence"):
                    for i, doc in enumerate(active_item['sources']):
                        st.markdown(f"<div class='source-container'><strong>Reference {i+1}:</strong><br>{doc.page_content}</div>", unsafe_allow_html=True)
                        
        st.markdown("</div>", unsafe_allow_html=True)

    with col_hist:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.subheader("📜 Recent Activity")
        if not st.session_state.history:
            st.write("No queries yet.")
        start_idx = max(0, len(st.session_state.history) - 5)
        for i in range(len(st.session_state.history) - 1, start_idx - 1, -1):
            item = st.session_state.history[i]
            # Use a button for each history item to load it into the main view
            if st.button(f"💬 {item['q'][:45]}{'...' if len(item['q'])>45 else ''}", key=f"hist_{i}", use_container_width=True):
                st.session_state.active_chat_idx = i
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# --- Footer ---
