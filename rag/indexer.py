from sentence_transformers import SentenceTransformer
import numpy as np

EMB_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

class LocalVectorStore:
    def __init__(self):
        self.embs = []
        self.docs = []

    def add(self, doc_text, metadata=None):
        if metadata is None:
            metadata = {}
        v = EMB_MODEL.encode(doc_text)
        self.embs.append(v)
        self.docs.append({"text": doc_text, "meta": metadata})

    def search(self, query, k=5):
        if not self.embs:
            return []
        qv = EMB_MODEL.encode(query)
        embs = np.array(self.embs)
        scores = (embs @ qv) / (np.linalg.norm(embs, axis=1) * (np.linalg.norm(qv) + 1e-9) + 1e-9)
        idx = scores.argsort()[::-1][:k]
        results = []
        for i in idx:
            results.append({"score": float(scores[i]), "text": self.docs[i]["text"], "meta": self.docs[i]["meta"]})
        return results

def build_index_from_docs(list_of_docs):
    vs = LocalVectorStore()
    for d in list_of_docs:
        vs.add(d["text"], {"url": d.get("url")})
    return vs
