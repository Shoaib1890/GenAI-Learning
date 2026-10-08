# Day 7 — AI Job Copilot (Hybrid System)

## What you built

An end-to-end resume ↔ job matcher that **combines every earlier idea**:

| Earlier day | Used in Day 7 |
| --- | --- |
| Day 1 | Groq chat completions, system/user messages, `.env` |
| Day 2 | JSON Schema extraction, Python scoring |
| Day 3 | MiniLM embeddings, `cos_sim` |
| Day 4 | Chunking, top-k retrieval, grounded LLM prompt |
| Day 5 | Vector retrieval *concept* (here: in-memory embeddings) |
| Day 6 | “LLM is not the source of truth” — Python owns the score |

Code: `day-7/main.py`  
Inputs: `resume.txt`, `job_description.txt`  
Project write-up: `day-7/readme.md`

---

## Pipeline (memorize this)

```text
1. Load resume + JD
2. LLM extracts { skill, priority } with JSON Schema
3. Map skills into REQUIREMENT_GROUPS
4. Exact / alias match on the resume text
5. Embed resume lines (chunks)
6. For groups still missing: keyword + semantic evidence
7. LLM verifies evidence strictly (JSON)
8. Python weighted score (required=2, preferred=1)
9. LLM writes recommendations from matched/missing lists
```

`main()` is only orchestration. Each step is a function.

---

## 1. Configuration

- Fail fast if `GROQ_API_KEY` is missing
- `temperature=0` on LLM calls that must be stable
- `REQUIRED_WEIGHT = 2`, `PREFERRED_WEIGHT = 1`
- `REQUIREMENT_GROUPS` — canonical buckets (Python, Cloud, RAG, …)
- `SKILL_ALIASES` — lowercase phrases that count as a hit for a group

Normalization is **predefined**, not fully LLM-driven. That is a deliberate limit (see README).

---

## 2. Structured extraction (Day 2 skill)

The LLM may **only** extract requirements. System prompt:

- classify `required` vs `preferred`
- do **not** decide if the candidate has the skill

Schema:

```json
{
  "requirements": [
    { "skill": "Python", "priority": "required" }
  ]
}
```

`enum` on priority blocks random strings like `"must-have"`.

Extraction is language understanding. Matching is a later, separate stage. Mixing them (Day 2 did both) makes the model judge and extract at once — easier to be sloppy.

---

## 3. Exact / alias matching (not embeddings)

```python
normalized_resume = resume.lower()
matched = any(alias in normalized_resume for alias in aliases)
```

This is substring search, not an LLM.

- Fast, explainable
- Catches “python”, “github”, “groq”
- Misses paraphrase without an alias
- Can false-positive (“auth” inside another word — aliases should be chosen carefully)

Groups with no aliases (e.g. Backend System Design in your map) tend to fall through to retrieval + verification.

---

## 4. Chunking and embeddings (Days 3–4)

Resume lines longer than 30 characters become chunks. Each is encoded with MiniLM.

Why lines, not the whole resume? Day 3’s whole-doc vector diluted specific evidence. Chunks let you point at **a sentence** as evidence.

`prepare_resume_embeddings` runs **once**. `retrieve_evidence` reuses those vectors (do not re-encode per skill group).

---

## 5. Hybrid evidence retrieval (Day 4 RAG shape)

For each **missing** group:

**Keyword retrieval**

- Use `SKILL_ALIASES` for that group
- If an alias appears in a chunk, keep it with `similarity: 1.0`, `source: "keyword"`

**Semantic retrieval**

- Query: `"Experience with {group}: " + skills`
- `cos_sim(query, resume_embeddings)`
- Top 3 chunks, `source: "semantic"`

**Merge**

- Dedupe by chunk text
- Keep up to 5 evidence items

Important: retrieval is **candidate evidence**, not a match. High cosine similarity ≠ “has Docker.”

---

## 6. LLM evidence verification (strict RAG)

`verify_evidence` sends the evidence JSON to Groq with hard rules, including:

- no assumed experience
- named tech (Docker, Redis, AWS, RAG, …) needs **explicit** mention
- APIs ≠ backend system design
- Express can count as REST API if JD allows “FastAPI or similar”
- frontend + backend can count as full-stack
- LLM API usage can support AI/LLM APIs
- search/filtering ≠ embeddings/RAG
- never treat similarity scores as proof
- `matched=false` if insufficient

This is the Day 4 “answer only from context” idea, used as a **classifier** instead of a Q&A bot.

Verification JSON:

```json
{
  "results": [
    { "group": "...", "matched": true, "reason": "..." }
  ]
}
```

Python still validates `results` is a list of dicts — models can still mis-shape output.

---

## 7. Deterministic weighted scoring (Days 2 + 6 lesson)

```text
final_matched = exact matches ∪ verified matches
final_missing = groups not in final_matched

weight(group) = 2 if any mapped requirement is required else 1
score = matched_weight / total_weight * 100
```

`group_priorities`: if any skill in the group was extracted as `required`, the whole group is required.

The LLM never outputs the percentage. That keeps scoring inspectable and stable.

---

## 8. Recommendations

A last Groq call gets **only**:

- match score
- matched skill names
- missing skill names

Rules: do not invent extra skills, do not change the score, keep actions practical.

This is generation **after** the facts are frozen — same idea as RAG answering after retrieval.

---

## Hybrid matching — why all four parts

| Method | Strength | Weakness |
| --- | --- | --- |
| Exact / alias | Precise named skills | Misses paraphrase |
| Semantic retrieval | Finds related wording | False-positive evidence |
| LLM verification | Reads nuance, applies rules | Still probabilistic |
| Python scoring | Repeatable numbers | Only as good as the match lists |

Day 2 trusted the LLM for matched/missing. Day 7 **narrows** the LLM: extract requirements, verify evidence, write advice. Search and arithmetic stay in Python.

---

## Mental model: probabilistic vs deterministic

```text
Probabilistic (LLM):
  extract skills from messy JD text
  judge if a resume line is enough evidence
  phrase career advice

Deterministic (Python):
  alias substring checks
  cosine ranking
  weights and percentage
  union of match sets
```

Interview one-liner: *“I don’t let the model invent the score. I use it for language, and I keep scoring in code.”*

---

## Limitations (know these for revision)

- Groups/aliases are hand-written for this JD family
- Text resume only, not PDF parsing
- CLI, not a web app
- Extraction errors propagate into weights
- Verification can still be wrong
- Not a hiring system of record
- Day 7 does not call Chroma; retrieval is in-process MiniLM

---

## How the 7 days fit one sentence each

1. Call a chat LLM with roles and history  
2. Force JSON so programs can parse the result  
3. Embeddings compare meaning, not just keywords  
4. Retrieve chunks, then generate from context (RAG)  
5. Store and search those vectors in a vector DB  
6. Let the model call tools; loop until a final answer  
7. Combine extraction, retrieval, verification, and Python scoring  

---

## Revision checklist

- [ ] Draw the 9-step pipeline from memory
- [ ] Why extract requirements without matching in the same prompt?
- [ ] Alias match vs semantic retrieval vs verification — who does what?
- [ ] Why is cosine similarity not treated as proof?
- [ ] How is the 61.9%-style score computed?
- [ ] What did each of Days 1–6 contribute to this file?
