import fitz  # PyMuPDF
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyMuPDF."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def chunk_text(text):
    """Splits text into chunks for RAG."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(chunks):
    """Creates a FAISS vector store from text chunks."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

def load_llm():
    """Loads the FLAN-T5 model for answer generation."""
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
    
    # Create an answer generation pipeline
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.1,
        do_sample=True # Set to True to avoid warning with temperature
    )
    
    return HuggingFacePipeline(pipeline=pipe)

def get_answer(query, vector_store, llm):
    """Retrieves relevant chunks and generates a detailed financial answer."""
    # Retrieve top 3 relevant chunks
    docs = vector_store.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    # Enhanced prompt template for professional financial analysis
    prompt = f"""
    You are an expert Financial Analyst. 
    Instructions: 
    - If the user asks about a financial ratio or metric, first define it and state its standard formula.
    - Then, calculate or extract the specific answer using the provided context.
    - Be professional and concise.

    Context: {context}
    
    Question: {query}
    
    Detailed Answer:"""
    
    response = llm.invoke(prompt)
    return response, docs
