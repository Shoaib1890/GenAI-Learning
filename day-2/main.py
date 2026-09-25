import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

job_description = input("Paste the job description:\n")

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "system",
            "content": """
            You are a job description analyzer.
            Analyze the job description provided by the user.
            """
        },
        {
            "role": "user",
            "content": job_description
        }
    ]
)

print(response.choices[0].message.content)