import fitz  # PyMuPDF
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-small"  # Changed from flan-t5-base to flan-t5-small for memory efficiency

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
        max_length=256,
        min_length=20,
        temperature=0.3,
        do_sample=True,
        top_p=0.95
    )
    
    return HuggingFacePipeline(pipeline=pipe)

def get_answer(query, vector_store, llm):
    """Retrieves relevant chunks and generates a detailed financial answer."""
    import numpy as np
    
    # Set relevance threshold - if score is below this, the query is not relevant to the document
    # Lower threshold allows metadata queries (like company name) to pass through
    RELEVANCE_THRESHOLD = 0.3
    
    # Initialize embeddings to compute query embedding
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Get query embedding
    query_embedding = np.array(embeddings.embed_query(query))
    
    # Retrieve top 5 chunks (more than needed to filter by threshold)
    docs = vector_store.similarity_search(query, k=5)
    
    # Compute similarity scores for each document
    docs_with_scores = []
    for doc in docs:
        # Get document embedding
        doc_embedding = np.array(embeddings.embed_query(doc.page_content))
        
        # Compute cosine similarity manually
        dot_product = np.dot(query_embedding, doc_embedding)
        norm_query = np.linalg.norm(query_embedding)
        norm_doc = np.linalg.norm(doc_embedding)
        score = dot_product / (norm_query * norm_doc) if (norm_query * norm_doc) > 0 else 0
        
        docs_with_scores.append((doc, float(score)))
    
    # Sort by score (descending) and take top 2
    docs_with_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Filter documents by relevance threshold
    relevant_docs = [(doc, score) for doc, score in docs_with_scores[:2] if score >= RELEVANCE_THRESHOLD]
    
    # If no relevant documents found, return "Information not available"
    if not relevant_docs:
        return "Information not available in the provided document.", []
    
    # Extract just the documents for context
    filtered_docs = [doc for doc, _ in relevant_docs]
    
    # Format context with explicit reference numbers
    context_parts = []
    for i, doc in enumerate(filtered_docs):
        context_parts.append(f"[Reference {i+1}]: {doc.page_content}")
    context = "\n\n".join(context_parts)
    
    # Strict prompt for exact extraction and citations
    prompt = f"""Based on the Context provided below, answer the Question directly and clearly.

IMPORTANT INSTRUCTIONS:
- Provide a complete, detailed answer (at least 2-3 sentences)
- Extract exact numerical values when relevant
- Always cite your source using the reference number (e.g., [Reference 1])
- Do not make up or hallucinate information
- If the answer is not in the context, state "Information not available"

Context:
{context}

Question: {query}

Answer:"""
    
    response = llm.invoke(prompt)
    return response, filtered_docs
