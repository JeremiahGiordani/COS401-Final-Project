from pydantic import BaseModel, Field
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core.models import SystemMessage, UserMessage, AssistantMessage
import env
import asyncio

class AutoGenAgent:
    def __init__(self, model: str = "gpt-4o"):
        self.agent = AzureOpenAIChatCompletionClient(
            model=model,
            azure_endpoint=env.BASE_URL,
            api_version=env.API_VERSION,
            api_key=env.API_KEY,
        )

    async def _solve_async(self, system_prompt: str, prompts: list[str]) -> str:
        messages = [SystemMessage(content=system_prompt)]

        for prompt in prompts:
            messages.append(UserMessage(content=prompt, source="user"))
            response = await self.agent.create(messages=messages)
            response_content = response.content
            messages.append(AssistantMessage(content=response_content, source="assistant"))

        return response_content

    def solve(self, system_prompt: str, prompts: list[str]) -> str:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self._solve_async(system_prompt, prompts))

if __name__ == "__main__":
    agent = AutoGenAgent()
    problem = "If x/4 = 2, what is x?"
    prompts = [
        f"Can you give me a hint to this problem: {problem}",
        f"Sorry, can you clarify?"
    ]
    system_prompt = "You are a helpful assistant"
    result = agent.solve(system_prompt=system_prompt, prompts=prompts)
    print(result)