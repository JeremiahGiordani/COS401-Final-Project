from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
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
            api_version=env.API_VERSION,
        )

        self.reasoning_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are solving math competition problems. Think step-by-step before giving an answer."),
            ("human", "{problem}")
        ])

        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract the final answer from the response."),
            ("human", "{reasoning}")
        ])

    def solve(self, problem: str) -> int:

        messages = self.reasoning_prompt.invoke({"problem": problem})
        reasoning_response = self.llm.invoke(messages)
        reasoning_text = reasoning_response.content

        tools = [Answer]
        structured_llm = self.llm.bind_tools(tools, tool_choice=True)
        result = structured_llm.invoke(self.answer_prompt.invoke({"reasoning": reasoning_text}))
        result = result.tool_calls[0]["args"]
        return result["answer"]

if __name__ == "__main__":
    agent = LangchainAgent()
    response = agent.solve("We have the following: x / 4 = 2. What is x?")
    print("Answer:", response)
