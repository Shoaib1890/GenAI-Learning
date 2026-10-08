# Day 3 — Embeddings and Cosine Similarity

## What you built

Day 2’s structured skill match, plus a **semantic similarity score** between the full job description and the full resume.

Pipeline:

1. LLM extracts skills / matches (JSON Schema)
2. Python computes skill match %
3. `SentenceTransformer("all-MiniLM-L6-v2")` embeds JD and resume
4. `cos_sim` produces a similarity in roughly `[-1, 1]` (usually `0` to `1` for this model)
5. Threshold `0.70` decides semantic match yes/no

Code: `day-3/main.py`

---

## Core idea

Keyword / LLM skill lists miss **paraphrase**.

```text
JD:      "REST API development"
Resume:  "Built backend services and integrated REST APIs"
```

Those are related even if the exact phrase differs.

An **embedding** maps text → a dense vector of numbers (here **384 dimensions** for MiniLM). Texts with similar meaning land closer in that vector space.

You then measure closeness with **cosine similarity**.

---

## What an embedding model is (vs a chat LLM)

| Chat LLM (Groq) | Embedding model (MiniLM) |
| --- | --- |
| Generates text | Generates a vector |
| Used for extraction, answers, verification | Used for similarity and retrieval |
| Slow / costly per call | Fast, local in your project |
| Probabilistic wording | Deterministic vector for the same text (same model) |

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode(text)  # length 384
```

You do **not** send MiniLM through Groq. It runs locally via `sentence-transformers`.

---

## Cosine similarity

For vectors `A` and `B`:

```text
cos(A, B) = (A · B) / (|A| |B|)
```

It measures **angle**, not Euclidean distance. Two long documents can still be similar if they point the same direction.

In code:

```python
cos_sim(embedding1, embedding2).item()
```

`.item()` turns a 1-element tensor into a Python float.

Interpretation for this setup (rough, not a law):

| Score | Meaning |
| --- | --- |
| ~0.8–1.0 | Very similar wording/meaning |
| ~0.6–0.8 | Related |
| below ~0.5 | Weak overlap |

Your threshold: `similarity >= 0.70` → semantic match.

Thresholds are **chosen**, not magic. Day 4 used `0.50` for retrieval because “is this chunk useful?” is a looser bar than “is this whole resume a match?”

---

## Two different signals

Day 3 prints both:

- **Skill match %** — from LLM-extracted lists
- **Semantic similarity %** — from embeddings of the *entire* JD vs *entire* resume

They can disagree:

- High skill match, lower semantic score: resume lists skills but is short / differently worded
- High semantic score, missing skills: resume talks about similar work but not the named tools

That disagreement is why Day 7 becomes **hybrid**: exact aliases + retrieval + LLM verification + Python weights.

---

## Limitations of embedding the whole documents

Day 3 encodes:

- one vector for the whole JD
- one vector for the whole resume

Problems:

- Long resume averages many topics into one vector
- A single Docker mention can get diluted
- Similarity ≠ proof the person has Docker

Days 4–5 and 7 **chunk** text, embed each chunk, and retrieve the top-k. That is the move from “document similarity” to “retrieval.”

---

## Encode, similarity, threshold — mental model

```text
text  →  encode()  →  vector
two vectors  →  cos_sim  →  float
float  →  compare to threshold  →  boolean
```

Same three steps appear in RAG:

```text
chunks → encode
question → encode
cos_sim(question, all chunks) → top-k
```

---

## Common pitfalls

- Thinking cosine similarity “proves” a skill
- Using one global threshold for every task
- Comparing embeddings from **different** models (spaces are not compatible)
- Forgetting `.item()` / tensor vs float
- Embedding huge concatenated strings when you needed chunk-level retrieval

---

## How this shows up later

- Day 4: `cos_sim(question, chunk_embeddings)` + top-3
- Day 5: ChromaDB stores embeddings and does nearest-neighbor search
- Day 7: resume lines are chunked, missing skill groups are queried semantically, evidence is verified by an LLM

---

## Revision checklist

- [ ] Embedding vs chat completion — what does each return?
- [ ] Why 384 dimensions for MiniLM?
- [ ] What does cosine similarity measure?
- [ ] Why can whole-document similarity miss a specific skill?
- [ ] Why is a similarity threshold a product choice, not a fact?
