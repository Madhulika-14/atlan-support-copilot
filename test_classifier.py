from core.classifier import classify_text

def test_classifier_basic():
    res = classify_text("How to create glossary term")
    assert "How-to" in res["topics"] or "Glossary" in res["topics"]
    assert "sentiment" in res
    assert "priority" in res
