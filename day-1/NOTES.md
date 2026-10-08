# Day 1 — LLM API Fundamentals with Groq

## What you built

A terminal chatbot that talks to Groq (`openai/gpt-oss-20b`), keeps conversation history, and exits when you type `exit`.

Code: `day-1/main.py`

---

## Core idea

An LLM is a next-token prediction model. You do not “run the model on your laptop” here. You send a **messages array** to an API, and the API returns the next assistant message.

Your app’s job is:

1. Load the API key from `.env`
2. Build a `messages` list
3. Call `client.chat.completions.create(...)`
4. Read `response.choices[0].message.content`
5. Append that reply back into `messages` so the next turn has context

---

## Groq client setup

```python
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
```

- `python-dotenv` loads `GROQ_API_KEY` from `.env`
- Never commit `.env`
- If the key is missing, the API call fails with an auth error

The model name is a Groq-hosted model id, not a local file:

```python
model="openai/gpt-oss-20b"
```

---

## Chat messages and roles

A chat completion is a list of message objects. Each has:

| Role | Who it represents | Typical use |
| --- | --- | --- |
| `system` | Hidden instructions | Personality, rules, style |
| `user` | The human | Questions / input |
| `assistant` | The model | Previous replies, kept for memory |

In your code:

```python
messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant. Explain technical concepts clearly..."
    }
]
```

Then each loop:

1. Append `{ "role": "user", "content": question }`
2. Call the API with the **full** `messages` list
3. Append `{ "role": "assistant", "content": answer }`

That last step is **conversation memory**. Without it, every question is a new isolated chat.

---

## Why the system prompt matters

The system message is the model’s standing instruction. It is sent on every request because it lives at the start of `messages`.

Examples of what a system prompt can control:

- Tone (“explain simply”)
- Output format (“return JSON only”) — Day 2
- Grounding (“use only the provided context”) — Day 4
- Strict evaluation rules — Day 7

The user prompt is the changing input. Keep stable rules in `system`, changing data in `user`.

---

## Request / response shape

**You send:**

- `model`
- `messages`

**You get back:**

- `choices` — usually one choice
- `choices[0].message.content` — the text reply
- usage/token metadata (not printed in Day 1)

LLMs are **stateless on the server** unless you send history. Memory is just the messages you keep locally.

---

## Tokens, context window, cost (revision)

- Text is split into **tokens** (subword pieces), not always whole words
- Input tokens + output tokens both count toward cost/limits
- The **context window** is the max tokens the model can see at once
- A growing `messages` list will eventually overflow the window

Day 1 does not truncate history. For a long chat you would later:

- keep only the last N turns, or
- summarize old turns, or
- move long knowledge into retrieval (Days 4–5)

---

## Temperature and determinism

Day 1 uses default sampling.

| Setting | Effect |
| --- | --- |
| `temperature=0` | More greedy / repeatable |
| Higher temperature | More varied wording |

Day 7 later sets `temperature=0` for extraction and verification, because those steps should be stable.

---

## Common pitfalls

- Forgetting to append the assistant reply → the model “forgets” what it just said
- Putting secrets in code instead of `.env`
- Treating the model as a database — it can hallucinate facts
- Assuming the API stores your chat — it does not
- Sending only the latest user message and losing the system prompt / history

---

## How this shows up later

- Day 2: same API, but `response_format` forces JSON
- Day 4–5: extra **context** is stuffed into the user message
- Day 6: assistant messages can contain `tool_calls` instead of (or besides) text
- Day 7: multiple specialized LLM calls, each with a different system prompt

---

## Revision checklist

- [ ] What are `system`, `user`, and `assistant` roles?
- [ ] Why must you resend the full `messages` list every turn?
- [ ] Where does conversation memory actually live?
- [ ] What does `choices[0].message.content` contain?
- [ ] Why keep the API key in `.env`?
- [ ] What happens if the chat becomes very long?
