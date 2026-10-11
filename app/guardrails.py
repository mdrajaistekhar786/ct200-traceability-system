"""
Guardrails for protecting LLM prompts from prompt injection
and jailbreak-style instructions contained in untrusted content.
"""

import re


class GuardrailViolation(Exception):
    """Raised when suspicious instructions are detected in untrusted content."""
    pass


# Common prompt-injection / jailbreak patterns
INJECTION_PATTERNS = [
    r"\bignore\s+(all\s+)?previous\s+instructions\b",
    r"\bignore\s+(all\s+)?prior\s+instructions\b",
    r"\bdisregard\s+(all\s+)?previous\s+instructions\b",
    r"\bdisregard\s+(all\s+)?prior\s+instructions\b",
    r"\bforget\s+(all\s+)?previous\s+instructions\b",
    r"\bignore\s+the\s+system\s+prompt\b",
    r"\bignore\s+the\s+developer\s+message\b",
    r"\boverride\s+(the\s+)?system\b",
    r"\breveal\s+(the\s+)?system\s+prompt\b",
    r"\breveal\s+(the\s+)?developer\s+prompt\b",
    r"\bshow\s+(me\s+)?the\s+system\s+prompt\b",
    r"\bprint\s+(the\s+)?system\s+prompt\b",
    r"\byou\s+are\s+now\s+(a|an)\b",
    r"\bact\s+as\s+(a|an)\b",
    r"\bpretend\s+you\s+are\b",
    r"\bjailbreak\b",
    r"\bdeveloper\s+mode\b",
    r"\bsystem\s+override\b",
    r"\bdo\s+not\s+follow\s+(the\s+)?instructions\b",
    r"\bfollow\s+these\s+instructions\s+instead\b",
    r"\bexecute\s+the\s+following\s+instructions\b",
    r"\byour\s+new\s+instructions\s+are\b",
]


def detect_prompt_injection(text: str) -> list[str]:
    """
    Detect common prompt-injection patterns in untrusted text.
    """

    if not text:
        return []

    normalized = text.lower()
    matches = []

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, normalized):
            matches.append(pattern)

    return matches


def validate_untrusted_content(
    requirement: dict,
    related_evidence: list,
    feedback: str | None = None,
) -> None:
    """
    Validate requirement/evidence/feedback before sending them to the LLM.

    Raises:
        GuardrailViolation: if suspicious prompt-injection content is found.
    """

    content_parts = []

    # Requirement metadata and content
    content_parts.append(str(requirement.get("id", "")))
    content_parts.append(str(requirement.get("title", "")))

    text = requirement.get("text", [])
    if isinstance(text, list):
        content_parts.append(" ".join(map(str, text)))
    else:
        content_parts.append(str(text))

    content_parts.append(str(requirement.get("tables", [])))

    # Retrieved evidence
    content_parts.append(str(related_evidence))

    # Validation feedback
    if feedback:
        content_parts.append(str(feedback))

    combined_text = "\n".join(content_parts)

    matches = detect_prompt_injection(combined_text)

    if matches:
        raise GuardrailViolation(
            "Potential prompt injection detected in "
            "document/evidence content. "
            "LLM generation blocked."
        )