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
    # Retrieve top 2 relevant chunks to stay within model's 512 token limit
    docs = vector_store.similarity_search(query, k=2)
    
    # Format context with explicit reference numbers
    context_parts = []
    for i, doc in enumerate(docs):
        context_parts.append(f"[Reference {i+1}]: {doc.page_content}")
    context = "\n\n".join(context_parts)
    
    # Strict prompt for exact extraction and citations
    prompt = f"""
    Answer the Question based strictly on the provided Context. 
    Extract the exact numerical values and provide a clear, accurate sentence.
    You MUST cite your source using the reference number (e.g., [Reference 1]).
    Do not hallucinate or make up information. If the answer is not in the context, state "Information not available."

    Context:
    {context}
    
    Question:
    {query}
    
    Answer with citations:"""
    
    response = llm.invoke(prompt)
    return response, docs
