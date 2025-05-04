import ast
import re
import argparse
import importlib.util
import tempfile
from datasets import load_dataset
from typing import Callable
from typing import List, Dict
import types
import sys
import io
import contextlib

from agents.structured_output.agent import Agent

from agents.structured_output.pydantic_structured_agent import PydanticAgent
from agents.structured_output.langchain_structured_agent import LangchainAgent
from agents.structured_output.crewai_structured_agent import CrewAIAgent
from agents.structured_output.autogen_structured_agent import AutoGenAgent
from agents.structured_output.direct_structured_call import BaselineLLMAgent


def extract_python_code(raw_response: str) -> str:
    match = re.search(r"```python\s*(.*?)```", raw_response, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else raw_response.strip()


def load_test_function(test_code: str) -> Callable:
    """Dynamically load the check function from the test_code string."""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(test_code)
        temp_filename = f.name

    spec = importlib.util.spec_from_file_location("test_module", temp_filename)
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)
    return test_module.check


def evaluate_code(n: int, agent: Agent) -> float:
    instruction_dataset = load_dataset(
        "codeparrot/instructhumaneval", split=f"test[:{n}]"
    )
    test_dataset = load_dataset("evalplus/humanevalplus", split=f"test[:{n}]")
    total_correct = 0

    for instruction, test in zip(instruction_dataset, test_dataset):
        prompt = instruction["context"] + "\n" + instruction["instruction"]
        response = agent.solve(prompt, coding=True)
        code = extract_python_code(response)

        try:
            exec_globals = {"List": List, "Dict": Dict, "__builtins__": __builtins__}
            exec_globals = {}
            with (
                io.StringIO() as buf,
                contextlib.redirect_stdout(buf),
                contextlib.redirect_stderr(buf),
            ):
                exec(code, exec_globals)
            candidate_fn = next(
                v
                for k, v in exec_globals.items()
                if isinstance(v, types.FunctionType) and k != "check"
            )
            check_fn = load_test_function(test["test"])
            check_fn(candidate_fn)
            total_correct += 1
            print("Test succeded")
        except Exception as e:
            print(f"Test failed: {e}")

    return total_correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n", type=int, default=30, help="Number of coding problems to solve"
    )
    args = parser.parse_args()

    agents = {
        "baseline": BaselineLLMAgent(),
        "pydantic": PydanticAgent(),
        "langchain": LangchainAgent(),
        "crewai": CrewAIAgent(),
        "autogen": AutoGenAgent(),
    }

    for name, agent in agents.items():
        print(f"\n--- Evaluating {name} ---")
        accuracy = evaluate_code(args.n, agent)
        print(f"Accuracy for {name}: {accuracy:.2%}")


if __name__ == "__main__":
    main()
