from crewai import Agent, Task, Crew, LLM
from pydantic import BaseModel, Field

class Answer(BaseModel):
    answer: int = Field(description="The answer to the math competition problem.")


class CrewAIAgent:
    def __init__(self, model: str = "gpt-4o"):

        self.llm = LLM(
            model=f"azure/{model}"
        )

        # Define the CrewAI agent
        self.agent = Agent(
            role="Math Solver",
            goal="You are solving math competition problems. Respond only with the final answer (an integer), no explanation.",
            backstory="You are a highly trained mathematical reasoning model.",
            verbose=False,
            allow_delegation=False,
            llm=self.llm,
            function_calling_llm=self.llm,
        )

    def solve(self, problem: str) -> int:
        # Wrap the problem in a Task
        task = Task(
            description=(
                f"Solve the following math problem and return only the final integer answer: {problem}"
            ),
            agent=self.agent,
            expected_output="A JSON schema with the answer as an integer",
            output_pydantic=Answer
        )

        # Create and run a Crew
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
