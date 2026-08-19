"""
Single entry point for all Gemini LLM calls in the system.

Every agent goes through call_llm / call_llm_json so that
model configuration and JSON parsing stay consistent.
"""

import json
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

DEFAULT_MODEL = os.getenv(
    "LLM_MODEL",
    "gemini-3.5-flash-lite"
)


def call_llm(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2
) -> str:
    """Send a prompt to Gemini and return the text response."""

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "temperature": temperature,
        },
    )

    return response.text or ""


def call_llm_json(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0
):
    """
    Call Gemini and parse a JSON response.

    Returns None if the model call fails or the response
    cannot be parsed as JSON.
    """

    try:
        raw = call_llm(
            prompt,
            model=model,
            temperature=temperature
        )

    except Exception as exc:
        print("LLM call failed:", exc)
        return None

    raw = raw.strip()

    # Remove Markdown code fences if Gemini returns them.
    if raw.startswith("```"):
        raw = raw.strip("`")

        if raw.lower().startswith("json"):
            raw = raw[4:]

        raw = raw.strip()

    try:
        return json.loads(raw)

    except json.JSONDecodeError:
        return None