from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict
import os
import json
import traceback
from openai import OpenAI

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

MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "google/gemma-4-26b-a4b-it:free"
)

class RequestBody(BaseModel):
    text: str
    schema_: Dict[str, str] = Field(alias="schema")

    class Config:
        populate_by_name = True


CITY_SUFFIXES = [
    " warehouse",
    " office",
    " branch",
    " depot",
    " hub",
    " center",
    " centre",
]


def clean_string(field, value):
    if value is None or not isinstance(value, str):
        return value

    value = value.strip()
    field = field.lower()

    # Only clean location-like fields
    location_fields = {
        "origin",
        "destination",
        "city",
        "location",
        "state",
        "country"
    }

    if field in location_fields:
        suffixes = [
            " warehouse",
            " office",
            " branch",
            " depot",
            " hub",
            " center",
            " centre"
        ]

        for suffix in suffixes:
            if value.lower().endswith(suffix):
                value = value[:-len(suffix)].strip()

    return value
    
@app.post("/dynamic-extract")
def dynamic_extract(req: RequestBody):

    try:

        prompt = f"""
You are a deterministic information extraction engine.

Extract information from the text.

IMPORTANT RULES

- Return ONLY valid JSON.
- No markdown.
- No code fences.
- No explanation.
- Return EXACTLY the keys in the schema.
- Never add extra keys.
- Missing values must be null.

TYPE RULES

string:
Return the canonical value only.
Remove unnecessary descriptive words.

Examples:
Return the exact field value.

Only remove location descriptors like:
"Mumbai warehouse" -> "Mumbai"

Do not shorten field values such as:
"sick leave"
"running shoes"
"Alpha Store"

integer:
Return JSON integer.

float:
Return JSON number.

boolean:
Return true or false.

date:
Return ISO format YYYY-MM-DD.

array[string]:
Return JSON array.

array[integer]:
Return JSON array.

Schema:

{json.dumps(req.schema_, indent=2)}

Text:

{req.text}
"""

        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        content = response.choices[0].message.content.strip()

        print(content)

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
