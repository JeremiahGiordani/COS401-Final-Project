from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import Runnable
import env

class Answer(BaseModel):
    answer: int = Field(description="The answer to the math competition problem.")

class LangchainAgent:
    def __init__(self, model_name: str = "gpt-4o"):
        self.llm = AzureChatOpenAI(
            model=model_name,
            temperature=0,
            azure_endpoint=env.BASE_URL,
            api_key=env.API_KEY,
            api_version=env.API_VERSION
        )
        
        # This automatically creates a chain that uses the prompt + LLM + parses the output as Answer
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are solving math competition problems. Respond only with the final answer."),
            ("human", "{problem}")
        ])
        
        self.chain: Runnable = self.prompt | self.llm.with_structured_output(Answer, method="function_calling")

    def solve(self, problem: str) -> int:
        result: Answer = self.chain.invoke({"problem": problem})
        return int(result.answer)

if __name__ == "__main__":
    agent = LangchainAgent()
    response = agent.solve("We have the following: x / 4 = 2. What is x?")
    print(response)