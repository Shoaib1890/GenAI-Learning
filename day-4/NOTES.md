# Day 4 — RAG Fundamentals and Retrieval

## What you built

A question-answering program over `knowledge.txt` (company leave policy).

Flow:

1. Load the policy document
2. Split it into sentence **chunks**
3. Embed each chunk with MiniLM
4. Embed the user question
5. Rank chunks by cosine similarity
6. If the best score `< 0.50`, stop (“no relevant information”)
7. Otherwise take **top 3** chunks as **context**
8. Ask Groq to answer **only** from that context

Code: `day-4/main.py`  
Data: `day-4/knowledge.txt`

This is classic **RAG**: Retrieval-Augmented Generation.

---

## Core idea

LLMs do not magically know your company’s leave policy. If you ask without context, they may **hallucinate**.

RAG pattern:

```text
Question
   → retrieve relevant text from YOUR documents
   → put that text in the prompt as CONTEXT
   → LLM generates an answer grounded in CONTEXT
```

Two stages:

| Stage | Job | Model |
| --- | --- | --- |
| Retrieval | Find relevant passages | Embeddings + cosine similarity |
| Generation | Write the answer | Groq chat LLM |

The LLM never searches a database in Day 4. Python searches, then stuffs text into the prompt.

---

## Chunking

```python
sentences = document.replace("\n", " ").split(". ")
chunks = [s.strip() for s in sentences if s.strip()]
```

You split on `". "` so each sentence is a retrieval unit.

Why chunk at all?

- Embeddings work better on focused passages
- You can send only the useful parts (saves tokens)
- Ranked chunks are inspectable

Chunking tradeoff:

| Too small | Too large |
| --- | --- |
| Loses surrounding meaning | Dilutes the topic; more irrelevant tokens |

Sentence splitting is a simple starter. Production RAG often uses overlapping windows, markdown headers, or recursive splitters.

---

## Retrieval: top-k + threshold

```python
similarities = cos_sim(question_embedding, embeddings)[0]
top_indices = similarities.argsort(descending=True)[:3]
```

- Embed once per chunk (document side)
- Embed the question once (query side)
- Compare query vs **every** chunk
- `argsort(descending=True)[:3]` = top-k = 3

**Threshold `0.50`:** if even the best chunk is weakly related, refuse to answer. That is better than generating a confident wrong policy.

Then:

```python
context = "\n\n".join(chunks[index] for index in top_indices)
```

Only those strings go to the LLM.

---

## Grounded generation prompt

System:

```text
Answer using only the provided context.
If the answer is not present, say you don't know.
```

User:

```text
CONTEXT:
...retrieved chunks...

QUESTION:
...user question...
```

This is **prompt stuffing**, not fine-tuning. You are not training the model on the policy. You are giving it a cheat sheet for this request.

If retrieval picked the wrong chunks, the answer will be wrong even with a perfect prompt. Garbage in, garbage out.

---

## Why RAG beats “paste the whole file”

For a tiny `knowledge.txt`, you could dump the whole policy every time.

RAG still matters because:

- Real docs are bigger than the context window
- Extra irrelevant text distracts the model
- You can debug *which* passages were used
- The same retrieve-then-generate idea scales to vector DBs (Day 5)

---

## Hallucination vs grounding

| Without RAG | With RAG (done well) |
| --- | --- |
| Model uses training-time knowledge | Model is instructed to use CONTEXT |
| May invent leave days | Can quote 24 paid leaves if retrieved |
| Hard to audit | You printed the retrieved chunks |

RAG reduces hallucination; it does not eliminate it. The model can still ignore context. That is why Day 7 adds a **verification** step for evidence, not just retrieval.

---

## Common pitfalls

- No threshold → answering from weakly related sentences
- Chunking on `. ` breaks abbreviations / numbered lists
- `top_k` too small misses a needed sentence
- `top_k` too large wastes context and adds noise
- Forgetting “if not in context, say you don’t know”
- Assuming retrieval is exact keyword search — it is semantic

---

## How this shows up later

- Day 5: same RAG loop, retrieval moved into ChromaDB
- Day 7: for each missing skill group, retrieve resume lines as evidence, then **verify** instead of answering a user question

Day 7 RAG is “retrieve evidence for a requirement,” not “answer a policy question.” Same retrieve-then-LLM shape.

---

## Revision checklist

- [ ] What do R, A, G stand for in RAG?
- [ ] Why chunk before embedding?
- [ ] What are top-k and a similarity threshold doing?
- [ ] Where does the retrieved text go in the Groq messages?
- [ ] Why can RAG still produce a wrong answer?
