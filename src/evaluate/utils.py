from openai import AzureOpenAI
import env
import json

def extract_answer(agent_response):
    client = AzureOpenAI(
        api_key=env.API_KEY,
        azure_endpoint=env.BASE_URL,
        api_version=env.API_VERSION,
    )
    tool = {
        "type": "function",
        "function": {
            "name": "Answer",
            "description": "Return the answer to the math problem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "integer",
                        "description": "The final answer to the problem."
                    }
                },
                "required": ["answer"],
                "additionalProperties": False
            }
        }
    }
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {
                "role": "system",
                "content": "You have a long detailed solution. Please return just the answer, no explanation, in JSON format.",
            },
            {
                "role": "user",
                "content": agent_response,
            },
        ],
        tools=[tool],
        tool_choice={"type": "function", "function": {"name": "Answer"}}
    )
    arguments_str = response.choices[0].message.tool_calls[0].function.arguments
    return json.loads(arguments_str)["answer"]