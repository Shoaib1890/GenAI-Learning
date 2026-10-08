# Day 5 — ChromaDB and Vector Search

## What you built

The same leave-policy RAG app as Day 4, but chunks are stored in **ChromaDB** and retrieved with `collection.query`.

Code: `day-5/main.py`  
Data: `day-5/knowledge.txt`

---

## Core idea

Day 4 kept embeddings in a Python list and ran `cos_sim` yourself.

That does not scale:

- restarting the script recomputes everything
- comparing a query to 10k+ vectors in a naive loop is slow
- you have no ids, metadata, or persistence

A **vector database** stores:

- the original text (`documents`)
- an embedding vector per document
- an `id`
- optional metadata

Query = “find nearest vectors to this question.”

Chroma is an embedding database you can run locally (in-memory client in your script).

---

## Chroma objects

```python
chroma_client = chromadb.Client()

collection = chroma_client.create_collection(
    name="company_knowledge"
)
```

- **Client:** connection to Chroma (here, default in-memory)
- **Collection:** like a table of vectors with a name

```python
collection.add(
    documents=chunks,
    ids=[str(i) for i in range(len(chunks))]
)
```

- `documents`: the sentence strings
- `ids`: unique string ids (`"0"`, `"1"`, ...)
- You did not pass `embeddings=` — Chroma embeds `documents` with its **default embedding function**

Then query:

```python
results = collection.query(
    query_texts=[question],
    n_results=3
)

retrieved_chunks = results["documents"][0]
```

`query_texts` means: embed this question with the same embedding function, then nearest-neighbor search.

`n_results=3` is top-k.

`results["documents"]` is a list of lists because you can query multiple texts at once. `[0]` is the first query.

---

## Vector search vocabulary

| Term | Meaning |
| --- | --- |
| Vector / embedding | Numeric representation of text |
| Index | Data structure for fast nearest-neighbor lookup |
| k-NN / top-k | Return the k closest vectors |
| Distance vs similarity | Chroma ranks by distance internally; closer = more similar |
| Metadata | Extra fields (source, page) for filtering — not used in Day 5 |
| Persistence | Saving the index to disk so you do not re-add every run |

Your client is in-memory: **process exit wipes the collection**. A persistent Chroma path would survive restarts. Day 7’s README lists “persistent vector database” as a future improvement.

---

## Day 4 vs Day 5 retrieval

| | Day 4 | Day 5 |
| --- | --- | --- |
| Store | Python lists | Chroma collection |
| Embed chunks | `SentenceTransformer.encode` | Chroma default embedder on `add` |
| Query | `cos_sim` + `argsort` | `collection.query` |
| Threshold | yes (`0.50`) | not applied in your Day 5 script |
| Context → Groq | same pattern | same pattern |

Your Day 5 file still creates a `SentenceTransformer` and encodes chunks for printing dimensions. **Retrieval itself uses Chroma**, not those MiniLM tensors. That leftover is a good revision catch: two embedding pipelines can exist in one file; only one is used for search.

If Chroma’s default model ≠ MiniLM, scores are not comparable to Day 4’s cosine values. Same idea (nearest neighbors), maybe different embedding space.

---

## RAG loop with a vector DB

```text
1. Chunk document
2. collection.add(documents, ids)
3. User question
4. collection.query(query_texts, n_results)
5. Join retrieved documents as CONTEXT
6. Chat completion: answer only from CONTEXT
```

The LLM still does not “call Chroma.” Python queries Chroma, then calls Groq. Day 6 is when the LLM starts requesting tools.

---

## Why ids matter

Every vector needs a unique id so you can:

- update or delete one chunk
- avoid duplicates on re-ingest
- map a hit back to a source sentence

Using `"0"`, `"1"`, ... is enough for a demo. Production ids often include filename + chunk index.

---

## Common pitfalls

- Re-`create_collection` with the same name after data exists (use `get_or_create_collection`)
- Duplicate ids on `add`
- Forgetting that in-memory Chroma is empty next run
- Mixing embedding models between insert and query
- Not checking distances — Day 5 always takes top 3, even if they are irrelevant
- Confusing `query_texts` (Chroma embeds for you) with `query_embeddings` (you pass vectors)

---

## How this shows up later

- Day 6: different skill — tools / agent loop, not vectors
- Day 7: retrieval is MiniLM + `cos_sim` over resume lines (in-process, like Day 4), plus keyword hits. Chroma is part of the 7-day journey even though Day 7 does not import it yet

Conceptually Day 7 still does vector retrieval; it just keeps the index in RAM like Day 4.

---

## Revision checklist

- [ ] What does a vector DB store besides the numbers?
- [ ] What does `collection.add` vs `collection.query` do?
- [ ] Why `results["documents"][0]`?
- [ ] In-memory vs persistent Chroma?
- [ ] Who embeds the query when you pass `query_texts`?
