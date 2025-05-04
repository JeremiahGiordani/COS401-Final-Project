from crewai import Agent, Task, Crew, LLM

class CrewAIAgent:
    def __init__(self, model: str = "gpt-4o",):
        self.llm = LLM(
            model=f"azure/{model}"
        )

    def solve(self, system_prompt: str, prompts: list[str]) -> str:
        agent = Agent(
            role="Reasoning Agent",
            goal=system_prompt,
            backstory=system_prompt,
            verbose=False,
            allow_delegation=False,
            llm=self.llm,
            function_calling_llm=self.llm,
        )

        tasks = []
        for i, prompt in enumerate(prompts):
            task = Task(
                description=prompt,
                agent=agent,
                expected_output="Respond clearly and helpfully.",
                context=tasks if i > 0 else None,
            )
            tasks.append(task)

        crew = Crew(
            agents=[agent],
            tasks=tasks,
            verbose=False,
            chat_llm=self.llm,
            manager_llm=self.llm,
            planning_llm=self.llm,
            function_calling_llm=self.llm,
        )

        result = crew.kickoff()
        return result.raw


if __name__ == "__main__":
    agent = CrewAIAgent()
    problem = "If x/4 = 2, what is x?"
    prompts = [
        f"Can you give me a hint to this problem: {problem}",
        f"Sorry, can you clarify? Can you walk me through with more clear steps. Give me a list"
    ]
    system_prompt = "You are a helpful assistant"
    result = agent.solve(system_prompt=system_prompt, prompts=prompts)
    print(result)
