from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import PydanticOutputParser
import env
import json

class LangchainAgent:
    def __init__(self, model_name: str = "gpt-4o"):
        self.llm = AzureChatOpenAI(
            model=model_name,
            temperature=0,
            azure_endpoint=env.BASE_URL,
            api_key=env.API_KEY,
            api_version=env.API_VERSION,
        )

        # From prior implementation
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("human", "{user_prompt}")
        ])


    def solve(self, system_prompt: str, prompts: list[str]) -> str:
        messages = [{"role": "system", "content": system_prompt}]

        for prompt in prompts:
            messages.append({"role": "user", "content": prompt})
            response = self.llm.invoke(messages)
            response = response.content
            messages.append({"role": "assistant", "content": response})

        return response

if __name__ == "__main__":
    agent = LangchainAgent()
    problem = "If x/4 = 2, what is x?"
    prompts = [
        f"Can you give me a hint to this problem: {problem}",
        f"Sorry, can you clarify?"
    ]
    system_prompt = "You are a helpful assistant"
    result = agent.solve(system_prompt=system_prompt, prompts=prompts)
    print(result)