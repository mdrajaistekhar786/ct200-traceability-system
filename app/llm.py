import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


def build_prompt(requirement):

    return f"""
You are a senior QA Engineer.

Generate test cases for the following software requirement.

Requirement ID:
{requirement["id"]}

Title:
{requirement["title"]}

Description:
{" ".join(requirement["text"])}

Tables:
{json.dumps(requirement["tables"], indent=2)}

Return ONLY valid JSON.

Format:

[
  {{
    "id": "TC-001",
    "title": "",
    "preconditions": "",
    "steps": [],
    "expected_result": "",
    "priority": ""
  }}
]
"""


def generate_test_cases(requirement, model="llama-3.3-70b-versatile"):

    prompt = build_prompt(requirement)

    response = client.responses.create(
        model=model,
        input=prompt
    )

    output = response.output_text

    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return []