from pydantic import BaseModel, Field
from openai import AzureOpenAI
import env
import json


class Answer(BaseModel):
    answer: int = Field(description="The answer to the math competition problem.")


class Code(BaseModel):
    answer: str = Field(description="The complete code implementation.")


class BaselineLLMAgent:
    def __init__(self, model: str = "gpt-4o"):
        self.client = AzureOpenAI(
            api_key=env.API_KEY,
            azure_endpoint=env.BASE_URL,
            api_version=env.API_VERSION,
        )
        self.model = model

    def solve(self, problem: str, coding=False) -> int | str:
        if coding:
            parameters = {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "integer",
                        "description": "The complete code implementation.",
                    }
                },
                "required": ["code"],
                "additionalProperties": False,
            }
            tool_choice = "Code"
        else:
            parameters = {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "integer",
                        "description": "The final answer to the problem.",
                    }
                },
                "required": ["answer"],
                "additionalProperties": False,
            }
            tool_choice = "Answer"

        tool = {
            "type": "function",
            "function": {
                "name": "Answer",
                "description": "Return the answer to the math problem.",
                "parameters": parameters,
            },
        }

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are solving a logical task. Respond only with the final answer (a number or code), no explanation.",
                },
                {
                    "role": "user",
                    "content": problem,
                },
            ],
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": tool_choice}},
        )

        arguments_str = response.choices[0].message.tool_calls[0].function.arguments
        if coding:
            parsed = Code.model_validate_json(arguments_str)
        else:
            parsed = Answer.model_validate_json(arguments_str)
        return parsed.answer


if __name__ == "__main__":
    agent = BaselineLLMAgent()
    result = agent.solve("If x/4 = 2, what is x?")
    print("Answer:", result)
