# ui/app.py
import streamlit as st
import requests
import json

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Atlan Support Copilot", layout="wide")

st.title("🛠️ Atlan Support Copilot")

# --- Health check ---
health = {}
try:
    health = requests.get(f"{API_BASE}/health").json()
except Exception:
    health = {"status":"unreachable", "index_ready": False}

st.sidebar.header("Backend Status")
st.sidebar.json(health)

# --- Bulk classification dashboard ---
st.header("📊 Bulk Ticket Classification")

if st.button("Load sample tickets"):
    try:
        data = requests.get(f"{API_BASE}/analytics").json()
        st.write(f"Total Tickets: {data['tickets']}")
        col1, col2 = st.columns(2)
        with col1: st.write("By Topic"); st.json(data["by_topic"])
        with col2: st.write("By Priority"); st.json(data["by_priority"])
    except Exception as e:
        st.error(f"Error fetching analytics: {e}")

# --- Interactive Agent ---
st.header("💬 Interactive AI Agent")

title = st.text_input("Ticket Title")
body = st.text_area("Ticket Body")
channel = st.selectbox("Channel", ["email","chat","voice","whatsapp"])

if st.button("Classify"):
    payload = {"title": title, "body": body, "channel": channel}
    try:
        res = requests.post(f"{API_BASE}/classify", json=payload).json()
        st.subheader("Internal Analysis View")
        st.json(res["analysis"])
    except Exception as e:
        st.error(f"Error: {e}")

if st.button("Ask Copilot"):
    payload = {"title": title, "body": body, "channel": channel}
    try:
        res = requests.post(f"{API_BASE}/rag", json=payload).json()
        st.subheader("Internal Analysis View")
        st.json(res["analysis"])
        st.subheader("Final Response View")
        st.write(res["answer"])
        if res.get("sources"):
            st.markdown("**Sources:**")
            for s in res["sources"]:
                st.markdown(f"- [{s}]({s})")
    except Exception as e:
        st.error(f"Error: {e}")

# --- Build KB Button ---
st.sidebar.header("Knowledge Base")
if st.sidebar.button("Rebuild KB Index"):
    try:
        res = requests.post(f"{API_BASE}/rag/build").json()
        st.sidebar.success(f"Index rebuilt with {res['chunks']} chunks")
    except Exception as e:
        st.sidebar.error(f"Error building index: {e}")




