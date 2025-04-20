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

    async def _solve_async(self, problem: str) -> int:
        reasoning = await self.agent.create(
            messages=[
                SystemMessage(content=(
                        "You are solving math competition problems. Please "
                        "think about the problem very carefully and tackle the "
                        "problem step by step."
                    )),
                UserMessage(content=problem, source="user")
            ],
        )

        reasoning = reasoning.content
        parameters = Answer.model_json_schema()
        parameters["additionalProperties"] = False

        response = await self.agent.create(
            messages=[
                SystemMessage(content="Extract the answer from the following response. Respond only with the final answer (an integer), no explanation in JSON format"),
                UserMessage(content=reasoning, source="user")
            ],
            tools=[
                {
                    "name": "Answer",
                    "description": "Return the answer to the math problem",
                    "parameters": parameters,
                    "strict": True
                }
            ],
            extra_create_args={"tool_choice": "required"}
        )

        args_str = response.content[0].arguments
        parsed = Answer.model_validate_json(args_str)
        return parsed.answer

    def solve(self, problem: str) -> int:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self._solve_async(problem))

if __name__ == "__main__":
    agent = AutoGenAgent()
    response = agent.solve("We have the following: x / 4 = 2. What is x?")
    print("Answer:", response)
