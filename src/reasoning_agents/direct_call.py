from pydantic import BaseModel, Field
from openai import AzureOpenAI
import env
import json


class Answer(BaseModel):
    answer: int = Field(description="The answer to the math competition problem.")


class BaselineLLMAgent:
    def __init__(self, model: str = "gpt-4o"):
        self.client = AzureOpenAI(
            api_key=env.API_KEY,
            azure_endpoint=env.BASE_URL,
            api_version=env.API_VERSION,
        )
        self.model = model

        # Prepare the tool/function spec
        self.tool = {
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

    def solve(self, problem: str) -> int:
        reasoning = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are solving math competition problems. Please "
                        "think about the problem very carefully and tackle the "
                        "problem step by step."
                    ),
                },
                {
                    "role": "user",
                    "content": problem,
                },
            ],
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Extract the answer from the following response.",
                },
                {
                    "role": "user",
                    "content": reasoning.choices[0].message.content,
                },
            ],
            tools=[self.tool],
            tool_choice={"type": "function", "function": {"name": "Answer"}}
        )

        arguments_str = response.choices[0].message.tool_calls[0].function.arguments
        parsed = Answer.model_validate_json(arguments_str)
        return parsed.answer


if __name__ == "__main__":
    agent = BaselineLLMAgent()
    result = agent.solve("If x/4 = 2, what is x?")
    print("Answer:", result)
