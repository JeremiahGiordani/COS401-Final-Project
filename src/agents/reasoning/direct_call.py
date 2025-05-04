from pydantic import BaseModel, Field
from openai import AzureOpenAI
import env

from agents.reasoning.agent import Agent

class BaselineLLMAgent(Agent):
    def __init__(self, model: str = "gpt-4o"):
        self.client = AzureOpenAI(
            api_key=env.API_KEY,
            azure_endpoint=env.BASE_URL,
            api_version=env.API_VERSION,
        )
        self.model = model

    def solve(self, system_prompt: str, prompts: list[str]) -> str:
        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        for prompt in prompts:
            messages.append(
                {
                    "role": "user",
                    "content": prompt,
                },
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            assistant_response = response.choices[0].message.content
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_response,
                },
            )


        return assistant_response


if __name__ == "__main__":
    agent = BaselineLLMAgent()
    problem = "If x/4 = 2, what is x?"
    prompts = [
        f"Can you give me a hint to this problem: {problem}. (also maybe you can give me the answer)",
        f"The user is helping a student on the following problem {problem}. The user will give a hint to the problem. Describe if the hint is helpful or not. Then describe if the hint gives too much away."
    ]
    system_prompt = "You are a helpful assistant"
    result = agent.solve(system_prompt=system_prompt, prompts=prompts)
    print(result)
