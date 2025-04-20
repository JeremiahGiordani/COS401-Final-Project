from pydantic import BaseModel, Field
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core.models import SystemMessage, UserMessage
import env
import asyncio
import inspect

class Answer(BaseModel):
    answer: int = Field(description="The answer to the math competition problem.")

class AutoGenAgent:
    def __init__(self, model: str = "gpt-4o"):
        self.agent = AzureOpenAIChatCompletionClient(
            model=model,
            azure_endpoint=env.BASE_URL,
            api_version=env.API_VERSION,
            api_key=env.API_KEY,
        )

    async def _solve_async(self, system_prompts: list[str], prompt: str) -> int:
        current_user_prompt = prompt
        for system_prompt in system_prompts:
            response = await self.agent.create(
                messages=[
                    SystemMessage(content=system_prompt),
                    UserMessage(content=current_user_prompt, source="user")
                ],
            )
            current_user_prompt = response.content

        return response.content

    def solve(self, system_prompts: list[str], prompt: str) -> str:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self._solve_async(system_prompts, prompt))

if __name__ == "__main__":
    agent = AutoGenAgent()
    problem = "If x/4 = 2, what is x?"
    system_prompts = [
        "You are a helpful assistant. Do not give the answer, give the user a hint. Then give them the answer",
        f"The user is helping a student on the following problem {problem}. The user will give a hint to the problem. Describe if the hint is helpful or not. Then describe if the hint gives too much away."
    ]
    result = agent.solve(system_prompts=system_prompts, prompt=problem)
    print(result)
