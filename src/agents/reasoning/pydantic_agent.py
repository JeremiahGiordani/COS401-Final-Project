from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import env
from openai import AsyncAzureOpenAI

from agents.reasoning.agent import Agent

class PydanticAgent(Agent):
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

    def solve(self, system_prompt: str, prompts: list[str]) -> str:
        agent = Agent(
            self.model,
            system_prompt=system_prompt
        )

        response = None
        for prompt in prompts:
            if response:
                response = agent.run_sync(prompt, message_history=response.all_messages())
            else:
                response = agent.run_sync(prompt)

        return response.data

if __name__ == "__main__":
    agent = PydanticAgent()
    problem = "If x/4 = 2, what is x?"
    prompts = [
        f"Can you give me a hint to this problem: {problem}",
        f"Sorry, can you clarify?"
    ]
    system_prompt = "You are a helpful assistant"
    result = agent.solve(system_prompt=system_prompt, prompts=prompts)
    print(result)