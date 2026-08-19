"""
Document Gate -- CRAG (Corrective RAG) node.

Decides whether an uploaded PDF is actually a CT-200 manual worth extracting
requirements from. A cheap structural/keyword heuristic runs first; the LLM
is only used to adjudicate when the heuristic is inconclusive, or to give a
human-readable reason either way.
"""

from app.llm_client import call_llm_json

EXPECTED_DOC_KEYWORDS = ["CT-200", "CT200"]


def _collect_titles(tree, limit=25):
    titles = []

    def walk(nodes):
        for n in nodes:
            if len(titles) >= limit:
                return
            titles.append(f'{n["number"]} {n["title"]}')
            if n.get("children"):
                walk(n["children"])

    walk(tree or [])
    return titles


def check_document_relevance(tree, filename: str, sample_text: str = "") -> dict:
    """
    Returns {"is_relevant": bool, "reason": str}.
    """
    titles = _collect_titles(tree)

    keyword_hit = any(
        kw.lower() in filename.lower() or kw.lower() in sample_text.lower()
        for kw in EXPECTED_DOC_KEYWORDS
    )

    # Nothing extracted and no keyword match anywhere -> reject without
    # spending an LLM call.
    if not titles and not keyword_hit:
        return {
            "is_relevant": False,
            "reason": (
                "No requirement-like numbered headings were extracted and "
                "the document does not reference CT-200 in its filename or "
                "visible text."
            ),
        }

    prompt = f"""
You are the Document Gate (CRAG) of a QA traceability system that only
accepts CT-200 product manuals for requirement extraction.

Filename: {filename}

Extracted section headings (sample):
{titles}

Sample text from the document:
{sample_text[:1500]}

Decide whether this document is a CT-200 manual or specification suitable
for extracting testable requirements from.

Return ONLY valid JSON in this exact shape, nothing else:
{{"is_relevant": true, "reason": "short justification"}}
"""

    result = call_llm_json(prompt)

    if not isinstance(result, dict) or "is_relevant" not in result:
        # LLM unavailable/malformed -> fall back to the heuristic rather
        # than blocking the pipeline.
        return {
            "is_relevant": keyword_hit or bool(titles),
            "reason": (
                "LLM relevance check unavailable or returned malformed "
                "output; fell back to filename/structure heuristic."
            ),
        }

    return result
