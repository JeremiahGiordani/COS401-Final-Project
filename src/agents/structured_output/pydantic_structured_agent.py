from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import env
from openai import AsyncAzureOpenAI


class Answer(BaseModel):
    answer: int = Field(description="The answer to the math competition problem")


class Code(BaseModel):
    answer: str = Field(description="The complete code implementation.")


class PydanticAgent:
    def __init__(self, model: str = "gpt-4o"):
        client = AsyncAzureOpenAI(
            azure_endpoint=env.BASE_URL,
            api_version=env.API_VERSION,
            api_key=env.API_KEY,
        )
        self.model = OpenAIModel(
            model,
            provider=OpenAIProvider(openai_client=client),
        )

    def solve(self, problem: str, coding=False) -> int | str:
        if coding:
            result_type = Code
        else:
            result_type = Answer
        agent = Agent(
            self.model,
            system_prompt="You are solving a logical task. Respond only with the final answer (a number or code), no explanation.",
            result_type=result_type,
        )
        response = agent.run_sync(problem)
        return response.data.answer


if __name__ == "__main__":
    agent = PydanticAgent()
    response = agent.solve("We have the following: x / 4 = 2. What is x?")
    print(response)
