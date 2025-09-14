import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

MODEL_PATH = "core/classifier_model.joblib"

def heuristic_tags(text):
    t = text.lower()
    tags = set()
    if re.search(r"\b(api|sdk|endpoint|curl)\b", t): tags.add("API/SDK")
    if re.search(r"\b(sso|saml|oauth|openid)\b", t): tags.add("SSO")
    if re.search(r"\b(connector|snowflake|bigquery|redshift)\b", t): tags.add("Connector")
    if re.search(r"\b(glossary|term|taxonomy)\b", t): tags.add("Glossary")
    if re.search(r"\b(lineage|trace|lineage graph)\b", t): tags.add("Lineage")
    return list(tags)

def simple_sentiment(text):
    t = text.lower()
    if any(x in t for x in ["not working","error","fail","unable","urgent","crash","angry","frustrat","429","401"]):
        return "Frustrated"
    if any(x in t for x in ["how to","how do i","where do i","help me","can i","how can i"]):
        return "Curious"
    return "Neutral"

def train_demo_model():
    examples = [
        ("how to create a glossary term", "How-to"),
        ("snowflake connector auth error", "Connector"),
        ("api sdk example code", "API/SDK"),
        ("sso login failing", "SSO"),
        ("how to set up lineage", "Lineage"),
        ("best practice for data governance", "Best practices"),
    ]
    texts, labels = zip(*examples)
    vec = TfidfVectorizer()
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=200)
    clf.fit(X, labels)
    joblib.dump((vec, clf), MODEL_PATH)
    return vec, clf

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        return train_demo_model()

vec, clf = load_model()

def classify_text(text):
    tags = heuristic_tags(text)
    sent = simple_sentiment(text)
    try:
        X = vec.transform([text])
        pred = clf.predict(X)[0]
    except Exception:
        pred = None
    if pred and pred not in tags:
        tags.append(pred)
    priority = "P2"
    if sent == "Frustrated" or "Connector" in tags or "API/SDK" in tags:
        priority = "P1"
    return {"topics": tags or ["Product"], "sentiment": sent, "priority": priority}
