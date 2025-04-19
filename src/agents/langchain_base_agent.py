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
    # response = agent.solve("We have the following: x / 4 = 2. What is x?")
    # response = agent.solve("Let $x,y$ and $z$ be positive real numbers that satisfy the following system of equations: \n\\[\\log_2\\left({x \\over yz}\\right) = {1 \\over 2}\\]\n\\[\\log_2\\left({y \\over xz}\\right) = {1 \\over 3}\\]\n\\[\\log_2\\left({z \\over xy}\\right) = {1 \\over 4}\\]\nThen the value of $\\left|\\log_2(x^4y^3z^2)\\right|$ is $\\tfrac{m}{n}$ where $m$ and $n$ are relatively prime positive integers. Find $m+n$.")
    # response = agent.solve("Let $O(0,0), A(\\tfrac{1}{2}, 0),$ and $B(0, \\tfrac{\\sqrt{3}}{2})$ be points in the coordinate plane. Let $\\mathcal{F}$ be the family of segments $\\overline{PQ}$ of unit length lying in the first quadrant with $P$ on the $x$-axis and $Q$ on the $y$-axis. There is a unique point $C$ on $\\overline{AB}$, distinct from $A$ and $B$, that does not belong to any segment from $\\mathcal{F}$ other than $\\overline{AB}$. Then $OC^2 = \\tfrac{p}{q}$, where $p$ and $q$ are relatively prime positive integers. Find $p + q$.")
    response = agent.solve("Jen enters a lottery by picking $4$ distinct numbers from $S=\\{1,2,3,\\cdots,9,10\\}.$ $4$ numbers are randomly chosen from $S.$ She wins a prize if at least two of her numbers were $2$ of the randomly chosen numbers, and wins the grand prize if all four of her numbers were the randomly chosen numbers. The probability of her winning the grand prize given that she won a prize is $\\tfrac{m}{n}$ where $m$ and $n$ are relatively prime positive integers. Find $m+n$.")


    print(response)