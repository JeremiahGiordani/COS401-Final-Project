from pydantic import BaseModel, Field
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core.models import SystemMessage, UserMessage
import env
import asyncio
import inspect


class Answer(BaseModel):
    answer: int = Field(description="The answer to the math competition problem.")


class Code(BaseModel):
    answer: str = Field(description="The complete code implementation.")


class AutoGenAgent:
    def __init__(self, model: str = "gpt-4o"):
        self.agent = AzureOpenAIChatCompletionClient(
            model=model,
            azure_endpoint=env.BASE_URL,
            api_version=env.API_VERSION,
            api_key=env.API_KEY,
        )

    async def _solve_async(self, problem: str, coding=False) -> int | str:
        if coding:
            parameters = Answer.model_json_schema()
            tool = {
                "name": "Code",
                "description": "Return the code implementation",
                "parameters": parameters,
                "strict": True,
            }
        else:
            parameters = Code.model_json_schema()
            tool = {
                "name": "Answer",
                "description": "Return the answer to the math problem",
                "parameters": parameters,
                "strict": True,
            }
        parameters["additionalProperties"] = False

        response = await self.agent.create(
            messages=[
                SystemMessage(
                    content="You are solving a logical task. Respond only with the final answer (a number or code), no explanation."
                ),
                UserMessage(content=problem, source="user"),
            ],
            tools=[tool],
            extra_create_args={"tool_choice": "required"},
        )
        print(response)

        args_str = response.content[0].arguments
        if coding:
            parsed = Code.model_validate_json(args_str)
        else:
            parsed = Answer.model_validate_json(args_str)
        return parsed.answer

    def solve(self, problem: str, coding=False) -> int | str:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self._solve_async(problem, coding=coding))


if __name__ == "__main__":
    agent = AutoGenAgent()
    response = agent.solve("We have the following: x / 4 = 2. What is x?")
    print("Answer:", response)
