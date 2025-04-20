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


        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("human", "{user_prompt}")
        ])


    def solve(self, system_prompts: list[str], prompt: str) -> str:

        current_user_prompt = prompt
        for system_prompt in system_prompts:
            messages = self.prompt.invoke({"system_prompt": system_prompt, "user_prompt": current_user_prompt})
            response = self.llm.invoke(messages)
            current_user_prompt = response.content

        return response.content

if __name__ == "__main__":
    agent = LangchainAgent()
    problem = "If x/4 = 2, what is x?"
    system_prompts = [
        "You are a helpful assistant. Do not give the answer, give the user a hint. Then give them the answer",
        f"The user is helping a student on the following problem {problem}. The user will give a hint to the problem. Describe if the hint is helpful or not. Then describe if the hint gives too much away."
    ]
    result = agent.solve(system_prompts=system_prompts, prompt=problem)
    print(result)