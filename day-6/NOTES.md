# Day 6 — Tool Calling and Agent Loops

## What you built

A company assistant that can answer questions by **calling Python functions**:

- `get_leave_balance(employee_name)`
- `get_leave_policy()`
- `get_working_hours()`

The LLM does not contain Shoaib’s leave count. It **requests a tool**, your code runs the function, then the LLM writes the final English answer.

Code: `day-6/main.py`

---

## Core idea

Chat LLMs only generate tokens. They cannot truly query a database unless **you** run code.

**Tool calling** (function calling):

1. You describe tools (name, description, JSON parameters)
2. You send the user question **plus** the tool list
3. The model may return `tool_calls` instead of a finished answer
4. Your program executes the matching Python function
5. You send the tool result back as a `tool` message
6. The model answers using that result

That loop can repeat (multiple tools / follow-up calls). That is an **agent loop**.

---

## Tool schema

Tools are JSON Schema for arguments:

```python
{
    "type": "function",
    "function": {
        "name": "get_leave_balance",
        "description": "Get the remaining leave balance for an employee.",
        "parameters": {
            "type": "object",
            "properties": {
                "employee_name": {
                    "type": "string",
                    "description": "The name of the employee."
                }
            },
            "required": ["employee_name"]
        }
    }
}
```

The **description** is how the model decides *when* to call it. Vague descriptions → wrong or missing calls.

`tool_choice="auto"` lets the model answer directly **or** call tools.

---

## Message roles in a tool conversation

| Role | Meaning |
| --- | --- |
| `user` | Question (“How many leaves does Shoaib have?”) |
| `assistant` + `tool_calls` | Model’s request to run functions |
| `tool` | Your function output, linked by `tool_call_id` |
| `assistant` (final) | Natural-language answer |

Your loop:

```python
while message.tool_calls:
    messages.append({ role: assistant, tool_calls: [...] })

    for tool_call in message.tool_calls:
        arguments = json.loads(tool_call.function.arguments)
        result = ...  # run Python
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": function_name,
            "content": str(result)
        })

    response = client.chat.completions.create(..., tools=tools, ...)
    message = response.choices[0].message
```

Important details:

- Re-send `tools=` on the follow-up call so the model can call again
- `tool_call_id` must match the id from the assistant message
- `arguments` arrive as a **JSON string** → `json.loads`
- One assistant turn can request **several** tools; you run each

---

## Why this is an “agent”

A single LLM call is not an agent.

An agent is:

```text
observe → decide (maybe call tool) → act → observe result → decide again → ...
until it produces a final message without tool_calls
```

Your `while message.tool_calls` **is** that loop.

Example:

- User: “What is Shoaib’s leave balance and the leave policy?”
- Model may call `get_leave_balance` and `get_leave_policy` in one turn
- You execute both
- Model writes one combined answer

---

## Tools vs RAG

| RAG (Days 4–5) | Tools (Day 6) |
| --- | --- |
| Retrieve text, stuff into prompt | Execute functions / APIs |
| Good for documents | Good for live or structured data |
| You choose chunks with similarity | Model chooses which function to call |
| Answer grounded in passages | Answer grounded in function return values |

You can combine them (a `search_docs` tool). Day 6 keeps tools as simple Python dicts / strings.

The model should **not** invent “Shoaib has 12 leaves” from memory if a tool exists — it should call `get_leave_balance`. If the tool schema/description is poor, it might skip the tool and hallucinate.

---

## Safety notes (conceptual)

You only expose three harmless functions. In real systems:

- never let the model run arbitrary shell commands
- validate arguments
- do not put secrets in tool results you later log
- treat tool names as an allowlist (`else: result = "Unknown tool"`)

---

## Common pitfalls

- Forgetting to append the assistant `tool_calls` message before tool results
- Mismatched `tool_call_id`
- Not `json.loads` on arguments
- Dropping `tools` on the second API call
- Infinite loop if the model keeps requesting tools (need a max-iteration cap in production)
- Putting real data only in the system prompt instead of a tool — it will go stale

---

## How this shows up later

Day 7 does not use Groq tools. It uses a **pipeline you wrote**: extract → match → retrieve → verify → score → recommend.

Still, Day 6’s lesson applies: **the LLM should not be the system of record**. Python owns leave balances in Day 6 and owns the match score in Day 7.

---

## Revision checklist

- [ ] What does `tools` + `tool_choice="auto"` tell the model?
- [ ] Difference between assistant `tool_calls` and role `tool`?
- [ ] Why parse `tool_call.function.arguments` with `json.loads`?
- [ ] What makes the `while` loop an agent loop?
- [ ] When would you use a tool instead of RAG?
