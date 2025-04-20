from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import env
from openai import AsyncAzureOpenAI

class PydanticAgent:
    def __init__(self, model: str = "gpt-4o"):
        client = AsyncAzureOpenAI(
            azure_endpoint=env.BASE_URL,
            api_version=env.API_VERSION,
            api_key=env.API_KEY
        )
        self.model = OpenAIModel(
            model,
            provider=OpenAIProvider(openai_client=client),
        )

    def solve(self, system_prompts: list[str], prompt: str) -> str:
        current_user_prompt = prompt
        for system_prompt in system_prompts:
            agent = Agent(
                self.model,
                system_prompt=system_prompt
            )
            response = agent.run_sync(current_user_prompt)
            current_user_prompt = response.data

        return response.data

if __name__ == "__main__":
    agent = PydanticAgent()
    problem = "If x/4 = 2, what is x?"
    system_prompts = [
        "You are a helpful assistant. Do not give the answer, give the user a hint. Then give them the answer",
        f"The user is helping a student on the following problem {problem}. The user will give a hint to the problem. Describe if the hint is helpful or not. Then describe if the hint gives too much away."
    ]
    result = agent.solve(system_prompts=system_prompts, prompt=problem)