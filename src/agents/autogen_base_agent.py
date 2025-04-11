from pydantic import BaseModel, Field
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
import env
import asyncio
from autogen_core.models import (
    SystemMessage,
    UserMessage,
)


class Answer(BaseModel):
    answer: int = Field(description="The answer to the math competition problem.")


class AutoGenAgent:
    def __init__(self, model: str = "gpt-4o"):
        self.agent = AzureOpenAIChatCompletionClient(
            model=model,
            azure_endpoint=env.BASE_URL,
            api_version=env.API_VERSION,
            api_key=env.API_KEY,
            response_format={"type": "json_object"}
        )

    async def solve(self, problem: str) -> int:
        parameters = Answer.model_json_schema()
        parameters["additionalProperties"] = False
        response = await self.agent.create(
            messages=[
                SystemMessage(content="You are solving math competition problems. Respond only with the final answer (an integer), no explanation in JSON format."),
                UserMessage(content=problem, source="user")
            ],
            tools=[
                {
                    "name": "Answer",
                    "description": "Return the answer to the math problem",
                    "parameters": parameters,
                    "strict": True
                }
            ],
        )
        args_str: Answer = response.content[0].arguments
        parsed: Answer = Answer.model_validate_json(args_str)
        return parsed.answer


async def main():
    agent = AutoGenAgent()
    response = await agent.solve("We have the following: x / 4 = 2. What is x?")

    print(response)

asyncio.run(main())