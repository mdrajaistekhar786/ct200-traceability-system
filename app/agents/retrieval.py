"""
Cross-requirement retrieval.

Gives the Validation Agent context beyond a single requirement's own text:
for a given requirement, finds other requirements in the same document whose
extracted text is lexically closest to it.

This is intentionally dependency-light (no embeddings/vector store) --
pure lexical (Jaccard) similarity over tokenized title+body text is enough
to catch the two failure modes this is meant for:
  1. A test case that's actually grounded in a *different* requirement
     (misattribution) rather than the one it's attached to.
  2. A requirement whose correct testing depends on a neighboring section
     (e.g. a sub-requirement referencing a shared table defined elsewhere).
"""

import re
from app.models import RequirementNode

STOPWORDS = set("""
a an the and or of to in on for with is are be shall must should this that
these those as by from at into your device unit system it its not may can
will if then than when where which who whom while such per
""".split())


def _tokenize(text: str):
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", (text or "").lower())
    return [w for w in words if w not in STOPWORDS]


def _jaccard(a_tokens, b_tokens) -> float:
    a, b = set(a_tokens), set(b_tokens)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def retrieve_related_requirements(
    db,
    document_id: int,
    current_node_id: str,
    query_text: str,
    top_k: int = 3,
) -> list:
    """
    Returns up to `top_k` related requirements as
    [{"node_id", "title", "text", "score"}], sorted by descending similarity.
    Requirements with zero lexical overlap are excluded.
    """
    query_tokens = _tokenize(query_text)

    if not query_tokens:
        return []

    candidates = (
        db.query(RequirementNode)
        .filter(
            RequirementNode.document_id == document_id,
            RequirementNode.node_id != current_node_id,
        )
        .all()
    )

    scored = []

    for cand in candidates:
        cand_tokens = _tokenize(f"{cand.title} {cand.text or ''}")
        score = _jaccard(query_tokens, cand_tokens)
        if score > 0:
            scored.append((score, cand))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    return [
        {
            "node_id": cand.node_id,
            "title": cand.title,
            "text": (cand.text or "")[:600],
            "score": round(score, 3),
        }
        for score, cand in scored[:top_k]
    ]
