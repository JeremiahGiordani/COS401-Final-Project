from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import PydanticOutputParser
import env
import json


class Answer(BaseModel):
    answer: int = Field(description="The answer to the math competition problem.")


class Code(BaseModel):
    answer: str = Field(description="The complete code implementation.")


class LangchainAgent:
    def __init__(self, model_name: str = "gpt-4o"):
        self.llm = AzureChatOpenAI(
            model=model_name,
            temperature=0,
            azure_endpoint=env.BASE_URL,
            api_key=env.API_KEY,
            api_version=env.API_VERSION,
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are solving a logical task. Respond only with the final answer (a number or code), no explanation.",
                ),
                ("human", "{problem}"),
            ]
        )

    def solve(self, problem: str, coding=False) -> int | str:
        if coding:
            tools = [Code]
        else:
            tools = [Answer]

        messages = self.prompt.invoke({"problem": problem})
        structured_llm = self.llm.bind_tools(tools, tool_choice=True)
        result = structured_llm.invoke(messages)
        result = result.tool_calls[0]["args"]
        return result["answer"]


if __name__ == "__main__":
    agent = LangchainAgent()
    response = agent.solve("We have the following: x / 4 = 2. What is x?")
    print(response)
