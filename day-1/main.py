import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)
# messages = []
messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant. Explain technical concepts clearly and use simple examples when appropriate."
    }
]
while True:
    question = input("You: ")
    if question.lower() == "exit":
        break

    messages.append({
        "role": "user",
        "content": question
    })
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages
    )
    answer = response.choices[0].message.content
    messages.append({
        "role": "assistant",
        "content": answer
    })
    print("AI:", answer)

