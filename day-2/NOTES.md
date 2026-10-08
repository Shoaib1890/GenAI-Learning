# Day 2 — Structured Output and JSON Schema

## What you built

A job/resume analyzer that:

1. Reads a pasted job description and resume
2. Asks Groq to extract fields into JSON
3. Enforces a JSON Schema on the response
4. Computes a **skill match percentage in Python**
5. Prints matched vs missing skills

Code: `day-2/main.py`

This is the first step toward AI Job Copilot: LLM extracts language; Python scores.

---

## Core idea

Free-text LLM output is hard to use in code:

```text
"The candidate seems pretty strong in Python..."
```

You cannot reliably `result["matched_skills"]` from that.

**Structured output** means the model must return JSON that matches a contract:

```json
{
  "job_title": "...",
  "experience": "...",
  "skills": ["Python", "FastAPI"],
  "responsibilities": ["..."],
  "matched_skills": ["Python"],
  "missing_skills": ["FastAPI"]
}
```

Then Python can parse it with `json.loads` and compute:

```text
match % = matched_skills / skills * 100
```

---

## JSON Schema vs “please return JSON”

Two layers:

### 1. Prompt instructions

The system prompt lists the exact fields and rules:

- extract title, experience, skills, responsibilities
- `matched_skills` = JD skills also in resume
- `missing_skills` = JD skills not in resume
- no extra fields

Prompts alone are **soft**. Models still sometimes add markdown or extra keys.

### 2. `response_format` with `json_schema`

```python
response_format={
    "type": "json_schema",
    "json_schema": {
        "name": "job_description",
        "schema": schema
    }
}
```

This is a **hard constraint** from the API:

- output must be a JSON object
- required keys must exist
- `additionalProperties: False` blocks extra keys
- array fields must be arrays of strings

Day 7 uses the same pattern for requirement extraction.

---

## Schema pieces you should remember

| Keyword | Meaning |
| --- | --- |
| `type: object` | Root is a JSON object |
| `properties` | Allowed keys |
| `required` | Keys that must appear |
| `additionalProperties: False` | No surprise keys |
| `type: array` + `items` | List of a given type |
| `enum` | Value must be one of a fixed set (Day 7 uses this for `required` / `preferred`) |

Your Day 2 schema required:

- `job_title`, `experience` — strings
- `skills`, `responsibilities`, `matched_skills`, `missing_skills` — string arrays

---

## System vs user content

**System:** the analyzer role and output rules.

**User:** the actual documents:

```text
JOB DESCRIPTION:
...

RESUME:
...
```

Keep documents in the user message so the system prompt stays reusable.

---

## Deterministic scoring in Python

```python
match_percentage = matched_skills / total_skills * 100
```

This is the same design principle as Day 7:

- LLM: language understanding (extract / compare text)
- Python: numbers, percentages, business rules

Why? LLMs are probabilistic. They may round, invent a score, or change format. Python is repeatable.

Limitation of Day 2 scoring:

- it trusts the LLM’s `matched_skills` list
- it does **not** independently search the resume
- one missed skill in extraction changes the percentage

Days 3 and 7 add matching that is less “LLM said so.”

---

## Parsing the response

```python
content = response.choices[0].message.content
result = json.loads(content)
```

With JSON schema, `content` should already be valid JSON text. Still parse it — it is a string, not a Python dict, until `json.loads`.

---

## Common pitfalls

- Asking for JSON in the prompt but not setting `response_format` → markdown fences (` ```json `) break `json.loads`
- Schema and prompt disagree on field names
- Letting the LLM invent the match percentage
- Treating `matched_skills` as ground truth without checking the resume
- `skills` empty → divide-by-zero (your code returns `0`)

---

## How this shows up later

- Day 3: same structured analysis, plus embeddings as a second signal
- Day 6: tool parameters are also JSON schemas
- Day 7: schema extracts `{ skill, priority }`; Python does weighted scoring; verification uses structured JSON too

---

## Revision checklist

- [ ] Why is free-text output a problem for applications?
- [ ] Difference between prompting for JSON and JSON Schema?
- [ ] What does `additionalProperties: False` prevent?
- [ ] Why compute match % in Python?
- [ ] What is still “LLM judgement” in Day 2 matching?
