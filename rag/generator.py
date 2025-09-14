# core/rag/generator.py
import os
from openai import OpenAI

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

def generate_answer_with_sources(question, passages):
    context = "\n\n".join([p["text"] for p in passages])
    sources = list({p["meta"].get("url") for p in passages if p["meta"].get("url")})
    prompt = f"""You are an Atlan support assistant.
Use the passages below to answer the user's question briefly (3-6 sentences).
At the end include a SOURCES list with URLs used.

Passages:
{context}

Question: {question}

Answer:"""
    if client:
        try:
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                max_tokens=400,
                temperature=0.0
            )
            ans = r.choices[0].message.content.strip()
        except Exception as e:
            ans = f"(OpenAI call failed: {e})\n\n" + (context[:800] + "..." if len(context) > 800 else context)
    else:
        ans = (context[:800] + "... (truncated)\n\n") if len(context) > 800 else context
        ans += "\n\n(Install OPENAI_API_KEY for better answers.)"
    return {"answer": ans, "sources": [s for s in sources if s]}

