from crewai import Agent, Task, Crew, LLM
from pydantic import BaseModel, Field


class Answer(BaseModel):
    answer: int = Field(description="The answer to the math competition problem.")

class Code(BaseModel):
    answer: str = Field(description="The complete code implementation.")

# CrewAI uses env vars for the token and endpoint
class CrewAIAgent:
    def __init__(self, model: str = "gpt-4o"):

        self.llm = LLM(
            model=f"azure/{model}"
        )


        self.agent = Agent(
            role="Logic Solver",
            goal="You are solving a logical task. Respond only with the final answer (a number or code), no explanation.",
            backstory="You are a highly trained reasoning model.",
            verbose=False,
            allow_delegation=False,
            llm=self.llm,
            function_calling_llm=self.llm,
        )

    def solve(self, problem: str, coding=False) -> int | str:
        if coding:
            expected_output = "A JSON schema with the code implementation"
            output_schema = Code
        else:
            expected_output="A JSON schema with the answer as an integer"
            output_schema = Answer

        task = Task(
            description=(
                f"Solve the following task and return only the final answer: {problem}"
            ),
            agent=self.agent,
            expected_output=expected_output,
            output_pydantic=output_schema
        )

        crew = Crew(
            agents=[self.agent],
            tasks=[task],
            verbose=False,
            chat_llm=self.llm,
            manager_llm=self.llm,
            planning_llm=self.llm,
            function_calling_llm=self.llm
        )

        result = crew.kickoff()
        return result.pydantic.answer


if __name__ == "__main__":
    agent = CrewAIAgent()
    result = agent.solve("We have the following: x / 4 = 2. What is x?")
    print(result)
