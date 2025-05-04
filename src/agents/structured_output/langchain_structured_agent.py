from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import PydanticOutputParser
import env
import json

class Answer(BaseModel):
    answer: int = Field(description="The answer to the math competition problem.")

class LangchainAgent:
    def __init__(self, model_name: str = "gpt-4o"):
        llm = AzureChatOpenAI(
            model=model_name,
            temperature=0,
            azure_endpoint=env.BASE_URL,
            api_key=env.API_KEY,
            api_version=env.API_VERSION,
        )

        tools = [Answer]

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are solving math competition problems. Respond only with the final answer in JSON format."),
            ("human", "{problem}")
        ])

        self.structured_llm = llm.bind_tools(tools, tool_choice=True)


    def solve(self, problem: str) -> int:
        messages = self.prompt.invoke({"problem": problem})
        result = self.structured_llm.invoke(messages)
        result = result.tool_calls[0]["args"]
        return result["answer"]

if __name__ == "__main__":
    agent = LangchainAgent()
    response = agent.solve("We have the following: x / 4 = 2. What is x?")
    print(response)