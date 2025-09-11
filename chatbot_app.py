import os
import re
import json
import requests
import trafilatura
import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse, urljoin

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from langchain.retrievers import BM25Retriever, EnsembleRetriever
from transformers import pipeline

# -----------------------
# Config
# -----------------------
st.set_page_config(page_title="💬 RAG Chatbot", layout="wide")
st.title("💬 RAG Chatbot")

INDEX_DIR = Path(r"D:\SpikeUp.AI\Project Futere Facts\indexes")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# API Key Helpers
# -----------------------
def has_openai_key():
    key = os.environ.get("OPENAI_API_KEY", "")
    return bool(key and key.startswith("sk-"))

# -----------------------
# Utils
# -----------------------
def crawl_website(start_url, max_pages=5):
    from collections import deque
    visited, q = set(), deque([start_url])
    results = []
    while q and len(visited) < max_pages:
        url = q.popleft()
        if url in visited:
            continue
        visited.add(url)
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            downloaded = trafilatura.fetch_url(url)
            text = trafilatura.extract(downloaded) if downloaded else None
            if not text:
                soup = BeautifulSoup(r.text, "html.parser")
                text = soup.get_text(separator="\n")
            if text and text.strip():
                results.append((url, text))
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                link = urljoin(url, a["href"])
                if urlparse(link).netloc == urlparse(start_url).netloc:
                    if link not in visited:
                        q.append(link)
        except Exception:
            continue
    return results

def extract_file_text(uploaded_file):
    if uploaded_file.type == "text/plain":
        text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        from PyPDF2 import PdfReader
        reader = PdfReader(uploaded_file)
        text = " ".join((page.extract_text() or "") for page in reader.pages)
    else:
        return uploaded_file.name, None
    text = re.sub(r"\s{2,}", " ", text.replace("\n", " ")).strip()
    return uploaded_file.name, text

def chunk_labeled_texts(labeled_texts, chunk_size=900, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    docs = []
    for source, text in labeled_texts:
        if not text:
            continue
        for c in splitter.split_text(text):
            if len(c.strip()) > 50:
                docs.append(Document(page_content=c, metadata={"source": source}))
    return docs

def get_embeddings():
    st.sidebar.info("✅ Using HuggingFace embeddings (free)")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def save_metadata(index_path: Path, description: str):
    meta = {
        "name": index_path.name,
        "description": description,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(index_path / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def load_metadata(index_path: Path):
    meta_file = index_path / "meta.json"
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"name": index_path.name, "description": "", "last_updated": "Unknown"}

@st.cache_resource
def get_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_documents(query, docs, top_k=8):
    if not docs:
        return []
    model = get_reranker()
    pairs = [(query, d.page_content) for d in docs]
    scores = model.predict(pairs)
    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:top_k]]

def deduplicate_chunks(docs, min_diff=60):
    seen, unique = set(), []
    for doc in docs:
        snippet = doc.page_content[:min_diff]
        if snippet not in seen:
            unique.append(doc)
            seen.add(snippet)
    return unique

# -----------------------
# Hybrid Retriever + Multi-query Expansion
# -----------------------
def get_hybrid_retriever(db):
    faiss_retriever = db.as_retriever(search_kwargs={"k": 30})
    docs = db.similarity_search("dummy", k=30)
    bm25_retriever = BM25Retriever.from_documents(docs)
    return EnsembleRetriever(retrievers=[faiss_retriever, bm25_retriever], weights=[0.6, 0.4])

@st.cache_resource
def get_query_gen():
    return pipeline("text2text-generation", model="google/flan-t5-base")

def expand_queries(question: str, n_variants=2):
    query_gen = get_query_gen()
    prompt = f"Generate {n_variants} different rephrasings of this question:\n{question}"
    outputs = query_gen(prompt, max_new_tokens=100)
    variants = [o["generated_text"].strip() for o in outputs]
    return [question] + variants

def hybrid_multiquery_search(db, question, top_k=6):
    retriever = get_hybrid_retriever(db)
    queries = expand_queries(question, n_variants=2)
    all_docs = []
    for q in queries:
        all_docs.extend(retriever.get_relevant_documents(q))
    unique_docs = deduplicate_chunks(all_docs)
    return rerank_documents(question, unique_docs, top_k=top_k)

# -----------------------
# HuggingFace LLM Fallback
# -----------------------
@st.cache_resource
def get_local_llm():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        device_map="auto",
    )

# -----------------------
# Sidebar: Build Index
# -----------------------
st.sidebar.header("⚙️ Build / Ingest")
source_mode = st.sidebar.radio("📂 Select Source:", ["Website", "Upload File"])
max_pages = st.sidebar.slider("📄 Max Pages to Crawl", 1, 20, 3)

website_url, uploaded_files = None, None
if source_mode == "Website":
    website_url = st.sidebar.text_input("🌍 Website URL", value="https://www.example.com")
else:
    uploaded_files = st.sidebar.file_uploader("📂 Upload PDF/TXT", type=["pdf", "txt"], accept_multiple_files=True)

index_description = st.sidebar.text_input("🏷️ Index Description", value="")
can_build = (source_mode == "Website" and website_url) or (source_mode == "Upload File" and uploaded_files)

if st.sidebar.button("⚡ Build Index", disabled=not can_build):
    labeled, index_name = [], None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if source_mode == "Website" and website_url:
        with st.spinner(f"Crawling {website_url}..."):
            labeled = crawl_website(website_url, max_pages=max_pages)
        index_name = "web_" + urlparse(website_url).netloc.replace(".", "_") + f"_{timestamp}"

    elif source_mode == "Upload File" and uploaded_files:
        with st.spinner("Extracting text from files..."):
            for uf in uploaded_files:
                name, txt = extract_file_text(uf)
                if txt and txt.strip():
                    labeled.append((name, txt))
        base_name = "multi_files" if len(uploaded_files) > 1 else uploaded_files[0].name.replace(".", "_")
        index_name = "pdf_" + base_name + f"_{timestamp}"

    if labeled and index_name:
        docs = chunk_labeled_texts(labeled)
        embeddings = get_embeddings()
        index_path = INDEX_DIR / f"{index_name}_hf"
        db = FAISS.from_documents(docs, embedding=embeddings)
        db.save_local(str(index_path))
        save_metadata(index_path, index_description or "No description")

        st.session_state["active_index"] = str(index_path)
        st.session_state["db"] = None
        st.session_state["messages"] = []
        st.session_state["force_select_active"] = True
        st.sidebar.success(f"✅ Index built: {index_path.name}")
    else:
        st.sidebar.error("❌ Nothing to index")

# -----------------------
# Index Management
# -----------------------
st.sidebar.header("📁 Index Management")
available_indexes = [p for p in INDEX_DIR.glob("*") if p.is_dir()]

if not available_indexes:
    st.warning("⚠️ No indexes found. Build one first.")
    st.stop()

metadata_list = [load_metadata(p) for p in available_indexes]
options = [f"{m['name']} → {m['description']} (📅 {m['last_updated']})" for m in metadata_list]

# auto-select active index
default_idx = 0
if "active_index" in st.session_state:
    active_name = Path(st.session_state["active_index"]).name
    for i, m in enumerate(metadata_list):
        if m["name"] == active_name:
            default_idx = i
            break

selected_display = st.sidebar.selectbox("📑 Select Index", options, index=default_idx)
selected_index = metadata_list[options.index(selected_display)]
index_path = INDEX_DIR / selected_index["name"]

embeddings = get_embeddings()
db = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
st.session_state["db"] = db

# -----------------------
# Chatbot
# -----------------------
st.sidebar.header("🧭 App Mode")
app_mode = st.sidebar.radio("Select Mode", ["Chatbot"], index=0)

if app_mode == "Chatbot":
    st.markdown(f"### 💬 Chatbot (Active Index: `{selected_index['name']}`)")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask me something...")
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        docs = hybrid_multiquery_search(st.session_state["db"], user_input, top_k=6)

        if has_openai_key():
            from langchain_openai import ChatOpenAI
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain

            structured_prompt = """
            You are a helpful assistant. Use ONLY the provided context.

            Question:
            {question}

            Answer:
            - Give a clear, direct answer.
            - Use bullet points if listing multiple facts.
            - If not in context, reply: "I don’t know from the given documents."

            Sources:
            - List supporting sources from metadata.
            - If no sources, write: "No sources found."

            Context:
            {context}
            """
            PROMPT = PromptTemplate(template=structured_prompt, input_variables=["context", "question"])
            llm = ChatOpenAI(model="gpt-3.5-turbo")
            context = "\n\n".join(d.page_content for d in docs)
            chain = LLMChain(llm=llm, prompt=PROMPT)
            answer = chain.run({"context": context, "question": user_input})
        else:
            llm = get_local_llm()
            context = "\n\n".join(d.page_content for d in docs)
            prompt = (
                "Answer the question using ONLY the context.\n"
                "If the answer is not in context, reply: 'I don’t know from the documents.'\n"
                "Use bullet points, be concise, and cite sources from metadata.\n\n"
                f"Context:\n{context}\n\nQuestion: {user_input}\nAnswer:"
            )
            result = llm(prompt, max_new_tokens=300, temperature=0.0)
            answer = result[0]["generated_text"]

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state["messages"].append({"role": "assistant", "content": answer})
