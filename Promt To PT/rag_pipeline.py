# rag_pipeline.py
import os
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from prompts import SYSTEM_PROMPT

DB_DIR = "db"

# Initialize embeddings & vectorstore
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embedding)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# -----------------------------
def ask_question(question: str):
    """
    Ask a question using RAG retrieval.

    Returns:
        response_text: str
        sources: list of document sources
    """
    # Retrieve relevant documents
    try:
        docs_list = retriever.invoke(question)
    except:
        try:
            docs_list = retriever.get_relevant_documents(question)
        except:
            docs_list = vectorstore.similarity_search(question, k=4)
    
    if len(docs_list) == 0:
        return "Information not available in provided documents.", []

    # Relevance check: verify that question keywords appear in the results
    question_words = set(question.lower().split())
    # Remove common stop words
    stop_words = {'what', 'is', 'a', 'the', 'in', 'of', 'to', 'and', 'for', 'with', 'on', 'by', 'about', 'how', 'why', 'when', 'where', 'which'}
    question_words = question_words - stop_words
    
    # Check if any significant question word appears in top results
    found_match = False
    for doc in docs_list[:2]:
        content_lower = doc.page_content.lower()
        for word in question_words:
            if word in content_lower and len(word) > 2:  # Only meaningful words
                found_match = True
                break
        if found_match:
            break
    
    # If no keywords match, information not available
    if not found_match and len(question_words) > 0:
        return "Information not available in provided documents.", []

    # Filter and clean retrieved chunks
    valid_chunks = []
    for d in docs_list:
        content = d.page_content.strip()
        
        # Skip chunks that are mostly numbers, symbols, or very short
        word_count = len(content.split())
        if word_count < 20:  # Minimum 20 words
            continue
        
        # Skip chunks that are mostly formatting characters
        symbol_ratio = sum(1 for c in content if not c.isalnum() and c != ' ' and c != '.') / len(content)
        if symbol_ratio > 0.3:  # More than 30% symbols
            continue
        
        valid_chunks.append(content)
    
    if not valid_chunks:
        # Fallback: return first chunk if no valid ones found
        valid_chunks = [d.page_content for d in docs_list if len(d.page_content) > 50]
    
    if not valid_chunks:
        return "Information not available in provided documents.", []
    
    # Extract and format key points from the best chunk
    primary_context = valid_chunks[0]
    
    # Clean up Q&A format: remove "What is X? Answer:" patterns
    text = primary_context
    # Remove Q&A patterns like "What is X? Answer: content"
    text = re.sub(r'[Ww]hat\s+is\s+[^?]*\?\s*[Aa]nswer:\s*', '', text)
    text = re.sub(r'[Qq]uestion:\s*[^?]*\?\s*[Aa]nswer:\s*', '', text)
    # Also handle plain "Answer: X" patterns
    text = re.sub(r'[Aa]nswer:\s+', '', text)
    
    # Split by sentence delimiters
    sentences = [s.strip() for s in re.split(r'[\.\n]+', text) if s.strip()]

    # Basic candidate filtering (length, not labels, not examples)
    candidates = []
    for sentence in sentences:
        if len(sentence) < 15 or len(sentence) > 350:
            continue
        if sentence[0].isdigit() or sentence.isupper() or ':' in sentence[:50]:
            continue
        if sentence.lower().startswith(('examples', 'e.g.', 'for example')):
            continue
        candidates.append(sentence)

    # If no candidates, fallback to using raw primary_context split
    if not candidates:
        candidates = [s.strip() for s in primary_context.replace('\n', ' ').split('.') if len(s.strip()) > 30]

    # Compute embeddings and score sentences by semantic similarity to the question
    try:
        question_emb = embedding.embed_documents([question])[0]
        sent_embs = embedding.embed_documents(candidates)
        qarr = np.array(question_emb)
        sims = []
        for emb in sent_embs:
            sarr = np.array(emb)
            # cosine similarity
            denom = (np.linalg.norm(qarr) * np.linalg.norm(sarr))
            sim = float(np.dot(qarr, sarr) / denom) if denom != 0 else 0.0
            sims.append(sim)

        # Select top sentences by similarity (threshold + top-k)
        ranked = sorted(zip(candidates, sims), key=lambda x: x[1], reverse=True)
        # prefer those above a reasonable threshold, otherwise top 4
        selected = [s for s,score in ranked if score >= 0.45]
        if not selected:
            selected = [s for s,score in ranked][:4]

        key_points = selected
    except Exception:
        # If embeddings fail for any reason, fallback to simple keyword/topic filtering
        q_words = [w for w in re.findall(r"\w+", question.lower()) if w not in stop_words and len(w) > 2]
        topic_word = sorted(q_words, key=len, reverse=True)[0] if q_words else None
        key_points = []
        for sentence in candidates:
            sent_low = sentence.lower()
            has_topic = topic_word and topic_word in sent_low
            keyword_matches = sum(1 for w in q_words if w in sent_low)
            if has_topic or keyword_matches > 0:
                key_points.append(sentence)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_points = []
    for pt in key_points:
        pt_lower = pt.lower().strip()
        if pt_lower not in seen:
            seen.add(pt_lower)
            unique_points.append(pt)
    
    # Limit to 5 concise key points
    key_points = unique_points[:5]
    
    # Format as professional bullet points
    if key_points:
        # Create formatted bullet points
        formatted_points = []
        for pt in key_points:
            clean_pt = pt.strip()
            # Remove trailing single/double digit numbers
            clean_pt = re.sub(r'\s+\d{1,2}\s*$', '', clean_pt)
            if len(clean_pt) > 15:
                formatted_points.append(f"• {clean_pt}")
        
        if formatted_points:
            bullet_text = "\n\n".join(formatted_points)
            response_text = bullet_text
        else:
            response_text = primary_context[:250] + "..."
    else:
        response_text = primary_context[:250] + "..."
    
    # Extract unique source (only one)
    # Since we're retrieving chunks from the same document, we only need one source
    source = None
    for d in docs_list:
        if hasattr(d, "metadata") and isinstance(d.metadata, dict):
            src = d.metadata.get("source", None)
            if src and src != "Unknown":
                source = src.replace("\\", "/")
                break  # Get first valid source and stop
    
    # Return as list for compatibility
    sources = [source] if source else []

    return response_text, sources
