from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict
import os
import json
import time
import traceback
from openai import OpenAI, RateLimitError

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "dynamic-extract"
    }

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Comma separated list in Render env:
# OPENROUTER_MODELS=google/gemma-4-26b-a4b-it:free,meta-llama/llama-3.3-70b-instruct:free,openai/gpt-oss-120b:free
MODELS = [
    m.strip()
    for m in os.getenv(
        "OPENROUTER_MODELS",
        "google/gemma-4-26b-a4b-it:free,meta-llama/llama-3.3-70b-instruct:free,openai/gpt-oss-120b:free"
    ).split(",")
]


class RequestBody(BaseModel):
    text: str
    schema_: Dict[str, str] = Field(alias="schema")

    class Config:
        populate_by_name = True


LOCATION_SUFFIXES = [
    " warehouse",
    " office",
    " branch",
    " depot",
    " hub",
    " center",
    " centre",
]


def clean_string(field_name: str, value):
    if value is None:
        return None

    if not isinstance(value, str):
        return value

    value = value.strip()

    # ONLY clean location-related fields
    if field_name.lower() in {
        "origin",
        "destination",
        "city",
        "location",
        "state",
        "country",
    }:
        for suffix in LOCATION_SUFFIXES:
            if value.lower().endswith(suffix):
                value = value[:-len(suffix)].strip()
                break

    return value


@app.post("/dynamic-extract")
def dynamic_extract(req: RequestBody):

    try:

        prompt = f"""
You are an information extraction engine.

Return ONLY a valid JSON object.

Rules:

- Return EXACTLY the keys from the schema.
- Never return extra keys.
- Missing values must be null.
- Return integers as JSON integers.
- Return floats as JSON numbers.
- Return booleans as true/false.
- Dates must be YYYY-MM-DD.
- Arrays must be JSON arrays.
- No markdown.
- No explanations.
- No code fences.

IMPORTANT

Return the exact field value.

Only normalize location fields.

Examples:

origin:
"Mumbai warehouse" -> "Mumbai"

destination:
"Delhi office" -> "Delhi"

Do NOT shorten values such as:

"sick leave"
"running shoes"
"Alpha Store"
"BlueDart"
"approved"

Schema:

{json.dumps(req.schema_, indent=2)}

Text:

{req.text}
"""

        response = None
        last_error = None

        # Try every configured model
        for model in MODELS:

            print(f"Trying model: {model}")

            for attempt in range(3):

                try:

                    response = client.chat.completions.create(
                        model=model,
                        temperature=0,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                    )

                    print(f"Success using {model}")

                    break

                except RateLimitError as e:

                    last_error = e

                    retry = 30

                    try:
                        retry = int(
                            e.response.headers.get(
                                "Retry-After",
                                "30"
                            )
                        )
                    except Exception:
                        pass

                    print(
                        f"Rate limited on {model}. "
                        f"Retrying in {retry}s..."
                    )

                    time.sleep(retry)

                except Exception as e:

                    print(e)

                    last_error = e

                    break

            if response is not None:
                break

        if response is None:
            raise last_error

        content = response.choices[0].message.content.strip()

        print(content)

        # Remove markdown if model returns ```json
        if content.startswith("```"):

            content = content.split("```")[1]

            if content.startswith("json"):
                content = content[4:]

            content = content.strip()

        data = json.loads(content)

        result = {}

        for key, typ in req.schema_.items():

            value = data.get(key)

            if value is None:
                result[key] = None
                continue

            try:

                if typ == "string":
                    result[key] = clean_string(key, str(value))

                elif typ == "integer":
                    result[key] = int(value)

                elif typ == "float":
                    result[key] = float(value)

                elif typ == "boolean":
                    result[key] = bool(value)

                elif typ == "date":
                    result[key] = str(value)

                elif typ == "array[string]":
                    result[key] = [str(x) for x in value]

                elif typ == "array[integer]":
                    result[key] = [int(x) for x in value]

                else:
                    result[key] = None

            except Exception:
                result[key] = None

        return result

    except Exception:

        print(traceback.format_exc())

        raise
