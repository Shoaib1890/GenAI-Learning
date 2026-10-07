import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def get_leave_balance(employee_name):
    leave_balances = {
        "Shoaib": 12,
        "Rahul": 8,
        "Priya": 15
    }

    return leave_balances.get(employee_name, 0)

def get_leave_policy():
    return "Employees should apply for planned leave at least 3 days in advance."


def get_working_hours():
    return "Standard working hours are 9 AM to 6 PM, Monday to Friday."


tools = [
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
    },
        {
        "type": "function",
        "function": {
            "name": "get_leave_policy",
            "description": "Get the company policy for planned leave.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_working_hours",
            "description": "Get the standard company working hours.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


question = input("Ask a question: ")

messages = [
    {
        "role": "user",
        "content": question
    }
]

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

message = response.choices[0].message

while message.tool_calls:

    # Add the assistant's tool-call request to the conversation
    messages.append({
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
            }
            for tool_call in message.tool_calls
        ]
    })

    # Execute every tool requested by the LLM
    for tool_call in message.tool_calls:

        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        if function_name == "get_leave_balance":
            result = get_leave_balance(
                arguments["employee_name"]
            )

        elif function_name == "get_leave_policy":
            result = get_leave_policy()

        elif function_name == "get_working_hours":
            result = get_working_hours()

        else:
            result = "Unknown tool"

        print("\nTool called:", function_name)
        print("Tool result:", result)

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": function_name,
            "content": str(result)
        })

    # Ask the LLM again using all the information collected so far
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    message = response.choices[0].message


print("\nFinal Answer:")
print(message.content)