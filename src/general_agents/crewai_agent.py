from crewai import Agent, Task, Crew, LLM
from pydantic import BaseModel, Field

class CrewAIAgent:
    def __init__(self, model: str = "gpt-4o",):
        self.llm = LLM(
            model=f"azure/{model}"
        )

    def solve(self, system_prompts: list[str], prompt: str) -> str:
        agents = []
        tasks = []
        for i, system_prompt in enumerate(system_prompts):
            agent = Agent(
                role=f"Agent_{i}",
                goal=system_prompt,
                backstory=system_prompt,
                verbose=False,
                allow_delegation=False,
                llm=self.llm,
                function_calling_llm=self.llm,
            )
            agents.append(agent)
            if i == 0:
                task = Task(
                    description=prompt,
                    agent=agent,
                    expected_output=system_prompt,
                )
            else:
                task = Task(
                    description=system_prompt,
                    agent=agent,
                    expected_output=system_prompt,
                    context=tasks
                )
            tasks.append(task)

        crew = Crew(
            agents=agents,
            tasks=tasks,
            verbose=False,
            chat_llm=self.llm,
            manager_llm=self.llm,
            planning_llm=self.llm,
            function_calling_llm=self.llm
        )

        result = crew.kickoff()
        return result.raw


if __name__ == "__main__":
    agent = CrewAIAgent()
    problem = "If x/4 = 2, what is x?"
    system_prompts = [
        "You are a helpful assistant. First, give the user a hint to solve the problem. Then give them the answer",
        f"The user is helping a student on the following problem {problem}. The user will give a hint to the problem. Describe if the hint is helpful or not. Then describe if the hint gives too much away."
    ]
    result = agent.solve(system_prompts=system_prompts, prompt=problem)
    print(result)
