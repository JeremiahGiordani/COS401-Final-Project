from crewai import Agent, Task, Crew, LLM
from pydantic import BaseModel, Field


class Answer(BaseModel):
    answer: int = Field(description="The answer to the math competition problem.")


class CrewAIAgent:
    def __init__(self, model: str = "gpt-4o"):
        self.llm = LLM(model=f"azure/{model}")

        # Reasoning agent
        self.reasoning_agent = Agent(
            role="Math Thinker",
            goal="Solve math problems step by step.",
            backstory="You're an expert in mathematical problem solving, explaining each step carefully.",
            verbose=False,
            llm=self.llm,
            function_calling_llm=self.llm
        )

        # Answer extractor
        self.answer_agent = Agent(
            role="Answer Extractor",
            goal="Read the reasoning and extract the final answer.",
            backstory="You specialize in parsing solutions and returning structured answers.",
            verbose=False,
            llm=self.llm,
            function_calling_llm=self.llm
        )

    def solve(self, problem: str) -> int:
        # First task: reasoning
        reasoning_task = Task(
            description=f"Solve the following math problem step by step: {problem}",
            agent=self.reasoning_agent,
            expected_output="A full explanation of how to solve the problem, step by step.",
        )

        # Second task: answer extraction
        answer_task = Task(
            description="Now extract the final answer from the previous explanation. "
                        "Return it as JSON with the format: {\"answer\": <integer>}.",
            agent=self.answer_agent,
            context=[reasoning_task],
            expected_output="Final JSON answer.",
            output_pydantic=Answer
        )

        crew = Crew(
            agents=[self.reasoning_agent, self.answer_agent],
            tasks=[reasoning_task, answer_task],
            chat_llm=self.llm,
            manager_llm=self.llm,
            function_calling_llm=self.llm,
            verbose=False,
        )

        result = crew.kickoff()
        return result.pydantic.answer


if __name__ == "__main__":
    agent = CrewAIAgent()
    result = agent.solve("We have the following: x / 4 = 2. What is x?")
    print("Answer:", result)
