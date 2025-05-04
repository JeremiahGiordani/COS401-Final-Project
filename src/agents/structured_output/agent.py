from abc import ABC, abstractmethod

class Agent(ABC):
    @abstractmethod
    def __init__(self, model: str = "gpt-4o") -> None:
        """
        Initialize the agent.
        """
        pass

    @abstractmethod
    def solve(self, problem: str, coding=False) -> str:
        """
        Solve a given task.
        """
        pass
