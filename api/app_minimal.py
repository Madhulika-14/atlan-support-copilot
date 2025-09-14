from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import json, os

app = FastAPI(title="Atlan Support Copilot - Minimal Demo")

BASE = Path(__file__).resolve().parents[2]
SAMPLE = BASE / "sample_tickets.json"

class TicketIn(BaseModel):
    title: str = ""
    body: str = ""
    channel: str = "email"

@app.get("/health")
def health():
    # index_ready = False because full RAG is disabled in this minimal backend
    return {"status":"ok", "index_ready": False}

@app.post("/classify")
def classify(ticket: TicketIn):
    s = (ticket.title + " " + ticket.body).lower()
    if "sso" in s or "okta" in s:
        topics=["SSO"]; sentiment="Frustrated"; priority="P0"; confidence=0.8; explanation="heuristic"
    elif "connector" in s or "sync" in s:
        topics=["Connector"]; sentiment="Frustrated"; priority="P1"; confidence=0.8; explanation="heuristic"
    elif "lineage" in s:
        topics=["Lineage"]; sentiment="Curious"; priority="P2"; confidence=0.6; explanation="heuristic"
    elif "api" in s or "sdk" in s:
        topics=["API/SDK"]; sentiment="Curious"; priority="P1"; confidence=0.7; explanation="heuristic"
    elif "how" in s or "how to" in s:
        topics=["How-to"]; sentiment="Curious"; priority="P2"; confidence=0.6; explanation="heuristic"
    else:
        topics=["Product"]; sentiment="Neutral"; priority="P2"; confidence=0.5; explanation="fallback"
    return {"analysis": {"topics": topics, "sentiment": sentiment, "priority": priority, "confidence": confidence, "explanation": explanation}}

@app.post("/rag")
def rag(ticket: TicketIn):
    # minimal backend can't run RAG — respond with routing fallback so UI still works
    return {"analysis": {"topics":["Product"], "sentiment":"Neutral", "priority":"P2", "confidence":0.5, "explanation":"fallback"}, "answer": f\"This ticket has been classified as a 'Product' issue and routed to the appropriate team.\", "sources": [], "explainability": None}

@app.get("/analytics")
def analytics():
    if not SAMPLE.exists():
        return {"tickets":0, "by_topic":{}, "by_priority":{}}
    data = json.load(open(SAMPLE, "r", encoding="utf-8"))
    by_topic = {}
    by_priority = {}
    for t in data:
        txt = (t.get("subject","") + " " + t.get("body","")).lower()
        if "how" in txt or "how to" in txt:
            top="How-to"
            pr="P2"
        elif "connector" in txt:
            top="Connector"; pr="P1"
        elif "api" in txt or "sdk" in txt:
            top="API/SDK"; pr="P1"
        elif "sso" in txt:
            top="SSO"; pr="P0"
        else:
            top="Product"; pr="P2"
        by_topic[top] = by_topic.get(top,0)+1
        by_priority[pr] = by_priority.get(pr,0)+1
    return {"tickets": len(data), "by_topic": by_topic, "by_priority": by_priority}
