# services/ui/app.py
# Streamlit UI + embedded classification & RAG logic (merged backend)
import os
import json
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any
import streamlit as st
import requests

# Optional OpenAI import (only used if OPENAI_API_KEY is provided in secrets)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") if st.secrets else os.getenv("OPENAI_API_KEY")
USE_OPENAI = bool(OPENAI_API_KEY)
if USE_OPENAI:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        # old client fallback
        import openai
        openai.api_key = OPENAI_API_KEY
        client = openai

# Minimal dependencies for RAG fallback
try:
    import numpy as np
    import faiss
except Exception:
    faiss = None
    np = None

# Paths and data
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DOCS_PATH = DATA_DIR / "atlan_docs.json"   # optional pre-scraped docs
SAMPLE_TICKETS = ROOT / "sample_tickets.json"

TOPIC_TAGS = [
    "How-to", "Product", "Connector", "Lineage", "API/SDK",
    "SSO", "Glossary", "Best practices", "Sensitive data"
]
RAG_TOPICS = {"How-to", "Product", "Best practices", "API/SDK", "SSO"}

# -------------------------
# Classification helpers
# -------------------------
CLASS_PROMPT_TEMPLATE = """
You are an assistant that classifies support tickets for Atlan.

Task:
- Choose one or more TOPIC tags from: {topic_list}
- Detect SENTIMENT: one of [Frustrated, Curious, Angry, Neutral]
- Suggest PRIORITY: P0 (High/Urgent), P1 (Medium), P2 (Low)
- Provide a CONFIDENCE score between 0.0 and 1.0
- Provide a short EXPLANATION (1-2 sentences) of why

Return JSON exactly in this format (no other text):
{{
  "topics": ["..."], 
  "sentiment":"...", 
  "priority":"P0/P1/P2", 
  "confidence": 0.0,
  "explanation":"..."
}}

Ticket subject: "{subject}"
Ticket body: "{body}"
""".strip()

def classify_text_heuristic(subject: str, body: str) -> Dict[str,Any]:
    s = (subject + " " + body).lower()
    if "sso" in s or "okta" in s or "single sign" in s:
        topics = ["SSO"]
        sentiment = "Frustrated"
        priority = "P0" if "urgent" in s or "cannot" in s else "P1"
    elif "connector" in s or "sync" in s:
        topics = ["Connector"]
        sentiment = "Frustrated"
        priority = "P0" if "failed" in s or "stopped" in s else "P1"
    elif "lineage" in s:
        topics = ["Lineage"]
        sentiment = "Curious"
        priority = "P2"
    elif "api" in s or "sdk" in s or "developer" in s:
        topics = ["API/SDK"]
        sentiment = "Curious"
        priority = "P1"
    elif "how" in s or "how to" in s or "where" in s:
        topics = ["How-to"]
        sentiment = "Curious"
        priority = "P2"
    else:
        topics = ["Product"]
        sentiment = "Neutral"
        priority = "P1"
    return {
        "topics": topics,
        "sentiment": sentiment,
        "priority": priority,
        "confidence": 0.6,
        "explanation": f"Heuristic matched topics {topics}"
    }

def classify_text_llm(subject: str, body: str) -> Dict[str,Any]:
    """Use OpenAI if available; otherwise fallback to heuristic."""
    if not USE_OPENAI:
        return classify_text_heuristic(subject, body)
    prompt = CLASS_PROMPT_TEMPLATE.format(
        topic_list=", ".join(TOPIC_TAGS),
        subject=subject.replace('"','\\"'),
        body=body.replace('"','\\"')
    )
    try:
        # Support both new OpenAI client and older openai library
        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            resp = client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role":"user","content":prompt}],
                temperature=0.0, max_tokens=220
            )
            choice = resp.choices[0]
            content = None
            if hasattr(choice, "message") and getattr(choice.message, "content", None) is not None:
                content = choice.message.content
            else:
                content = choice.get("message", {}).get("content") if isinstance(choice, dict) else None
            text = content or ""
        else:
            resp = client.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], temperature=0.0, max_tokens=220)
            text = resp.choices[0].message.content
        import re, json as _json
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            j = _json.loads(m.group(0))
            if "topics" not in j:
                j["topics"] = ["Product"]
            if "confidence" not in j:
                j["confidence"] = 0.5
            return j
        else:
            return classify_text_heuristic(subject, body)
    except Exception as e:
        # On any LLM error, fallback to heuristic
        st.warning("LLM classify failed; using heuristic fallback.")
        return classify_text_heuristic(subject, body)

# -------------------------
# RAG helpers (minimal)
# -------------------------
RAG_PROMPT = """
You are an Atlan support assistant. Use the passages provided to answer the user's question concisely (3-6 sentences).
At the end, include a SOURCES list with the URLs you used.

Question:
{question}

Passages:
{passages}
""".strip()

def embed_texts_fallback(texts: List[str]):
    """Simple fallback if no OpenAI available: return random vectors (deterministic-ish)."""
    if USE_OPENAI:
        try:
            # try to call embeddings on new client
            resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
            vecs = [item.embedding for item in resp.data]
            import numpy as _np
            return _np.array(vecs).astype("float32")
        except Exception:
            pass
    # fallback random vectors (stable seed)
    import numpy as _np
    rng = _np.random.RandomState(42)
    return rng.rand(len(texts), 1536).astype("float32")

def generate_answer_with_sources(question: str, hits: List[Dict[str,str]]) -> Dict[str,Any]:
    passages = "\n\n---\n\n".join([f"URL: {h.get('url','')}\n\n{h.get('text','')[:1500]}" for h in hits])
    if not USE_OPENAI:
        sources = list({h.get("url","") for h in hits if h.get("url")})
        return {
            "answer": "I don't have an OpenAI key configured. Please consult the listed docs or add an API key to Streamlit secrets.",
            "sources": sources,
            "explainability": passages[:800]
        }
    prompt = RAG_PROMPT.format(question=question, passages=passages)
    try:
        # new client / older client compatibility
        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], temperature=0.0, max_tokens=600)
            choice = resp.choices[0]
            ans = None
            if hasattr(choice, "message") and getattr(choice.message, "content", None) is not None:
                ans = choice.message.content.strip()
            else:
                ans = str(choice)
        else:
            resp = client.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], temperature=0.0, max_tokens=600)
            ans = resp.choices[0].message.content.strip()
        sources = list({h.get("url","") for h in hits if h.get("url")})
        return {"answer": ans, "sources": sources, "explainability": passages[:800]}
    except Exception:
        st.warning("RAG LLM failed; returning fallback.")
        sources = list({h.get("url","") for h in hits if h.get("url")})
        return {
            "answer": "Failed to generate answer with LLM. Routing fallback applied.",
            "sources": sources,
            "explainability": passages[:800]
        }

# -------------------------
# Optional: load pre-scraped docs if present
# -------------------------
def load_docs_if_exist():
    if DOCS_PATH.exists():
        try:
            docs = json.load(open(DOCS_PATH, "r", encoding="utf-8"))
            return docs
        except Exception:
            return []
    return []

DOCS_META = load_docs_if_exist()

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(layout="wide", page_title="Atlan Support Copilot — Demo")

st.sidebar.header("Backend Status")
index_ready = bool(DOCS_META)
st.sidebar.json({"status": "ok" if True else "unreachable", "index_ready": index_ready})

st.title("💬 Interactive AI Agent")

with st.form("agent_form"):
    st.subheader("Submit a ticket / question")
    subject = st.text_input("Ticket Title", "")
    body = st.text_area("Ticket Body", "", height=180)
    channel = st.selectbox("Channel", ["email", "chat", "whatsapp"])
    col1, col2 = st.columns([1, 1])
    with col1:
        classify_btn = st.form_submit_button("Classify")
    with col2:
        ask_btn = st.form_submit_button("Ask Copilot")

# Result panels
analysis_placeholder = st.empty()
final_response_placeholder = st.empty()

def do_classify(local_subject, local_body):
    # prefer LLM classify if key is present
    try:
        res = classify_text_llm(local_subject, local_body)
    except Exception as e:
        st.error(f"Classification failed: {e}")
        res = classify_text_heuristic(local_subject, local_body)
    return res

def do_rag(local_subject, local_body):
    analysis = do_classify(local_subject, local_body)
    topics = analysis.get("topics", [])
    use_rag = any(t in RAG_TOPICS for t in topics)
    if not use_rag:
        route_msg = f"This ticket has been classified as a '{topics[0] if topics else 'Product'}' issue and routed to the appropriate team."
        return {"analysis": analysis, "answer": route_msg, "sources": [], "explainability": None}
    # Build hits from docs if available
    if not DOCS_META:
        return {"analysis": analysis, "answer": "No KB available. Please rebuild the index or consult docs: https://docs.atlan.com/ and https://developer.atlan.com/", "sources": [], "explainability": None}
    # naive retrieval: return top few docs (no faiss) -- just choose first k
    hits = DOCS_META[:5]
    gen = generate_answer_with_sources(local_subject + "\n\n" + local_body, hits)
    return {"analysis": analysis, "answer": gen["answer"], "sources": gen["sources"], "explainability": gen.get("explainability","")}

if classify_btn:
    analysis = do_classify(subject, body)
    analysis_placeholder.subheader("🔍 Internal Analysis View")
    analysis_placeholder.json(analysis, expanded=False)

if ask_btn:
    with st.spinner("Generating answer..."):
        result = do_rag(subject, body)
    analysis_placeholder.subheader("🔍 Internal Analysis View")
    analysis_placeholder.json(result["analysis"], expanded=False)

    final_response_placeholder.subheader("✅ Final Response View")
    if result.get("answer"):
        final_response_placeholder.markdown("**RAG Answer:**")
        final_response_placeholder.write(result["answer"])
    if result.get("sources"):
        final_response_placeholder.markdown("**Sources:**")
        for s in result["sources"]:
            final_response_placeholder.write(f"- {s}")
    if result.get("explainability"):
        final_response_placeholder.markdown("**Explainability (excerpt):**")
        final_response_placeholder.write(result["explainability"])
