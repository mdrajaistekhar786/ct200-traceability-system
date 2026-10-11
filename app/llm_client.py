"""
Single entry point for all Gemini LLM calls in the system.

Every agent goes through call_llm / call_llm_json so that
model configuration, retry handling, and JSON parsing stay consistent.
"""

import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.getenv("GEMINI_API_KEY"),
    timeout=60.0,
    max_retries=0,
)


DEFAULT_MODEL = os.getenv(
    "LLM_MODEL",
    "gemini-3.8-flash",
)


MAX_RETRIES = 3
INITIAL_BACKOFF = 2


def call_llm(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
) -> str:
    """
    Send a prompt to Gemini.

    Retries temporary 503/unavailable failures using
    exponential backoff.

    Quota/rate-limit 429 errors are not retried because
    retrying immediately cannot resolve the exhausted quota.
    """

    for attempt in range(MAX_RETRIES + 1):

        try:

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=temperature,
                max_tokens=2048,
            )

            return (
                response.choices[0].message.content
                or ""
            )

        except Exception as exc:

            error_text = str(exc)

            is_quota_error = (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
            )

            is_temporary_error = (
                "503" in error_text
                or "UNAVAILABLE" in error_text
            )

            # Do not retry exhausted API quota.
            if is_quota_error:

                print(
                    "LLM quota exceeded. "
                    "Not retrying this request."
                )

                raise

            # Retry temporary provider failures.
            if (
                not is_temporary_error
                or attempt >= MAX_RETRIES
            ):
                raise

            wait_time = (
                INITIAL_BACKOFF
                * (2 ** attempt)
            )

            print(
                f"LLM temporary failure "
                f"(attempt {attempt + 1}/{MAX_RETRIES + 1}). "
                f"Retrying in {wait_time}s..."
            )

            time.sleep(
                wait_time
            )


def call_llm_json(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
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
            temperature=temperature,
        )

    except Exception as exc:

        print(
            "LLM call failed:",
            exc,
        )

        return None

    raw = raw.strip()

    # Remove Markdown code fences if the model returns them.
    if raw.startswith("```"):

        raw = raw.strip("`")

        if raw.lower().startswith("json"):
            raw = raw[4:]

        raw = raw.strip()

    try:

        return json.loads(raw)

    except json.JSONDecodeError:

        print(
            "LLM returned invalid JSON."
        )

        print(
            "RAW LLM RESPONSE:"
        )

        print(raw)

        return None