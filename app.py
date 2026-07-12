from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import os
import json
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL = "openai/gpt-oss-120b:free"


class RequestBody(BaseModel):
    text: str
    schema: Dict[str, str]


@app.post("/dynamic-extract")
def dynamic_extract(req: RequestBody):

    prompt = f"""
You are a strict information extraction engine.

Extract ONLY the requested fields.

Rules:

- Return EXACTLY the keys from schema.
- No additional keys.
- Missing values -> null.
- integer -> JSON integer.
- float -> JSON number.
- boolean -> true/false.
- date -> YYYY-MM-DD.
- array[string] -> JSON array of strings.
- array[integer] -> JSON array of integers.

Schema:

{json.dumps(req.schema, indent=2)}

Text:

{req.text}

Return ONLY JSON.
"""

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={"type":"json_object"}
    )

    data = json.loads(response.choices[0].message.content)

    result = {}

    for key, typ in req.schema.items():

        value = data.get(key)

        if value is None:
            result[key] = None
            continue

        try:
            if typ == "string":
                result[key] = str(value)

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

        except:
            result[key] = None

    return result
