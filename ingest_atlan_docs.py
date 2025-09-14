import requests
from bs4 import BeautifulSoup
from pathlib import Path
import json

seed_urls = [
    "https://docs.atlan.com/",
    "https://developer.atlan.com/"
]

def fetch_text(url):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        texts = [p.get_text(strip=True) for p in soup.find_all("p")]
        return "\n\n".join([t for t in texts if t])
    except Exception:
        return ""

out = []
for u in seed_urls:
    t = fetch_text(u)
    if t:
        out.append({"url": u, "text": t})

Path("data").mkdir(parents=True, exist_ok=True)
with open("data/atlan_docs.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("Saved", len(out), "docs to data/atlan_docs.json")
