from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import env
from openai import AsyncAzureOpenAI

from agents.structured_output.agent import Agent

class Answer(BaseModel):
    answer: int = Field(description="The answer to the math competition problem")

class PydanticAgent:
    def __init__(self, model: str = "gpt-4o"):
        client = AsyncAzureOpenAI(
            azure_endpoint=env.BASE_URL,
            api_version=env.API_VERSION,
            api_key=env.API_KEY
        )
        model = OpenAIModel(
            model,
            provider=OpenAIProvider(openai_client=client),
        )
        self.agent = Agent(
            model,
            system_prompt="You are solving math competition problems. Respond only with the final answer (an integer), no explanation.",
            result_type=Answer,
        )

    def solve(self, problem: str) -> int:
        response = self.agent.run_sync(problem)
        return response.data.answer

if __name__ == "__main__":
    agent = PydanticAgent()
    response = agent.solve("We have the following: x / 4 = 2. What is x?")
    print(response)