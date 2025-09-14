import json
from typing import List, Dict

def load_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

def normalize_ticket(t: Dict) -> Dict:
    return {
        "id": t.get("id"),
        "channel": t.get("channel", "unknown"),
        "title": t.get("title",""),
        "body": t.get("body",""),
        "text": (t.get("title","") + "\n\n" + t.get("body","")).strip()
    }
