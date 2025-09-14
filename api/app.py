# services/api/app.py
# Free version: no OpenAI needed, uses heuristics + sentence-transformers

from dotenv import load_dotenv
load_dotenv()

import json
import traceback
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Free embeddings
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FAISS_PATH = DATA_DIR / "faiss.index"
META_PATH = DATA_DIR / "faiss_meta.json"

app = FastAPI(title="Atlan Support Copilot API (Free Demo)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class TicketIn(BaseModel):
    title: str = ""
    body: str = ""
    channel: str = "email"

class QueryIn(BaseModel):
    q: str

# Config
TOPIC_TAGS = [
    "How-to", "Product", "Connector", "Lineage", "API/SDK",
    "SSO", "Glossary", "Best practices", "Sensitive data"
]
RAG_TOPICS = {"How-to", "Product", "Best practices", "API/SDK", "SSO"}

# Heuristic classification
def classify_text_heuristic(subject: str, body: str) -> Dict[str,Any]:
    s = (subject + " " + body).lower()
    if "sso" in s or "okta" in s or "single sign" in s:
        topics = ["SSO"]; sentiment = "Frustrated"; priority = "P0"
    elif "connector" in s or "sync" in s:
        topics = ["Connector"]; sentiment = "Frustrated"; priority = "P1"
    elif "lineage" in s:
        topics = ["Lineage"]; sentiment = "Curious"; priority = "P2"
    elif "api" in s or "sdk" in s or "developer" in s:
        topics = ["API/SDK"]; sentiment = "Curious"; priority = "P1"
    elif "how" in s or "how to" in s or "where" in s:
        topics = ["How-to"]; sentiment = "Curious"; priority = "P2"
    else:
        topics = ["Product"]; sentiment = "Neutral"; priority = "P2"
    return {
        "topics": topics,
        "sentiment": sentiment,
        "priority": priority,
        "confidence": 0.6,
        "explanation": f"Heuristic matched topics {topics}"
    }

# Free embeddings with sentence-transformers
LOCAL_EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts: List[str]) -> np.ndarray:
    return LOCAL_EMBED_MODEL.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")

# Crawl docs
def fetch_pages(domain: str, max_pages=3, timeout=5):
    to_visit = [domain]; visited = set(); pages = []
    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited: continue
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code != 200: continue
            soup = BeautifulSoup(r.text, "html.parser")
            for s in soup(["script","style","header","footer","nav","aside"]): s.decompose()
            text = soup.get_text(separator="\n")
            pages.append({"url": url, "text": text}); visited.add(url)
            for a in soup.find_all("a", href=True):
                href = urljoin(url, a["href"])
                if urlparse(href).netloc == urlparse(domain).netloc:
                    if href not in visited and href not in to_visit:
                        to_visit.append(href)
        except: continue
    return pages

def chunk_text(text: str, chunk_size=400, overlap=50):
    tokens = text.split(); chunks = []; i = 0
    while i < len(tokens):
        chunks.append(" ".join(tokens[i:i+chunk_size]))
        i += chunk_size - overlap
    return chunks

def build_index_from_domains(domains: List[str], max_pages_per_domain=2, save=True):
    docs = []
    for domain in domains:
        pages = fetch_pages(domain, max_pages=max_pages_per_domain)
        for p in pages:
            for c in chunk_text(p["text"]):
                docs.append({"text": c, "url": p["url"]})
    if not docs: raise RuntimeError("No docs fetched")
    embeddings = embed_texts([d["text"] for d in docs])
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    if save:
        faiss.write_index(index, str(FAISS_PATH))
        json.dump(docs, open(META_PATH, "w", encoding="utf-8"))
    return index, docs

def load_index_if_exists():
    if FAISS_PATH.exists() and META_PATH.exists():
        try:
            return faiss.read_index(str(FAISS_PATH)), json.load(open(META_PATH))
        except: return None, None
    return None, None

# RAG retrieval
def retrieve(index, docs, query, k=4):
    q_emb = embed_texts([query])
    _, idxs = index.search(q_emb, k)
    return [docs[i] for i in idxs[0] if i < len(docs)]

def generate_answer_with_sources(question: str, hits: List[Dict[str,str]]) -> Dict[str,Any]:
    answer = "Based on the docs, here are some helpful points:\n\n"
    for h in hits[:3]:
        answer += f"- {h['text'][:200]}...\n(Source: {h['url']})\n\n"
    return {"answer": answer, "sources": list({h["url"] for h in hits}), "explainability": answer}

# App state
INDEX, DOCS_META = None, None

@app.on_event("startup")
def startup_event():
    global INDEX, DOCS_META
    INDEX, DOCS_META = load_index_if_exists()
    if INDEX is not None:
        app.state.index_ready = True; return
    try:
        domains = ["https://docs.atlan.com/", "https://developer.atlan.com/"]
        INDEX, DOCS_META = build_index_from_domains(domains, max_pages_per_domain=1, save=True)
        app.state.index_ready = True
    except Exception as e:
        app.state.index_ready = False
        traceback.print_exc()

# Endpoints
@app.get("/health")
def health():
    return {"status":"ok", "index_ready": bool(app.state.__dict__.get("index_ready", False))}

@app.post("/classify")
def classify(ticket: TicketIn):
    return {"analysis": classify_text_heuristic(ticket.title, ticket.body)}

@app.post("/rag")
def rag_answer(ticket: TicketIn):
    txt = (ticket.title + "\n\n" + ticket.body).strip()
    analysis = classify_text_heuristic(ticket.title, ticket.body)
    if not any(t in RAG_TOPICS for t in analysis["topics"]):
        return {"analysis": analysis, "answer": f"This ticket has been classified as {analysis['topics'][0]} and routed to the team.", "sources": [], "explainability": None}
    if INDEX is None or DOCS_META is None:
        raise HTTPException(status_code=503, detail="KB index not ready.")
    hits = retrieve(INDEX, DOCS_META, txt, k=5)
    return {"analysis": analysis, **generate_answer_with_sources(txt, hits)}

@app.post("/rag/build")
def rag_build():
    global INDEX, DOCS_META
    try:
        domains = ["https://docs.atlan.com/", "https://developer.atlan.com/"]
        INDEX, DOCS_META = build_index_from_domains(domains, max_pages_per_domain=1, save=True)
        app.state.index_ready = True
        return {"status":"ok", "chunks": len(DOCS_META)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics")
def analytics():
    stickets = BASE_DIR / "sample_tickets.json"
    if not stickets.exists():
        return {"tickets":0,"by_topic":{},"by_priority":{}}
    data = json.load(open(stickets))
    counts_topic, counts_pr = {}, {}
    for t in data:
        analysis = classify_text_heuristic(t.get("subject",""), t.get("body",""))
        for top in analysis["topics"]:
            counts_topic[top] = counts_topic.get(top,0)+1
        counts_pr[analysis["priority"]] = counts_pr.get(analysis["priority"],0)+1
    return {"tickets": len(data), "by_topic": counts_topic, "by_priority": counts_pr}



