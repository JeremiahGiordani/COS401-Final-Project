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

from agents.reasoning.agent import Agent

from agents.reasoning.pydantic_agent import PydanticAgent
from agents.reasoning.langchain_agent import LangchainAgent
from agents.reasoning.crewai_agent import CrewAIAgent
from agents.reasoning.autogen_agent import AutoGenAgent
from agents.reasoning.direct_call import BaselineLLMAgent


ONE_SHOT_PROMPT = "Given the following coding problem, please implement the solution step by step"

PLANNING_PROMPT = (
    "Given the following coding prompt, please formulate an approach "
    "to the problem. Do NOT write code yet, "
    "just come up with a plan"
)
REFLECT_PROMPT = (
    "Great, now that you've written up a plan, I want you to "
    "carefully reflect on it. It's VERY important that you get "
    "it right. Please reconsider all of your steps and aim to correct "
    "any mistakes."
)
EXECUTE_PROMPT = "Great, now given you plan, please implement the code."

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

def evaluate_code(n: int, agent: Agent, reasoning_steps: int=1) -> float:
    instruction_dataset = load_dataset("codeparrot/instructhumaneval", split=f"test[:{n}]")
    test_dataset = load_dataset("evalplus/humanevalplus", split=f"test[:{n}]")
    total_correct = 0

    for instruction, test in zip(instruction_dataset, test_dataset):
        coding_prompt = instruction["context"] + "\n" + instruction["instruction"]
        system_prompt = "You are an expert coding agent"

        if reasoning_steps == 1:
            one_shot_prompt = ONE_SHOT_PROMPT + f"Coding problem: {coding_prompt}"
            prompt = [one_shot_prompt]
        elif reasoning_steps == 2:
            planning_prompt = PLANNING_PROMPT + f"Coding problem: {coding_prompt}"
            prompt = [planning_prompt, EXECUTE_PROMPT]
        elif reasoning_steps == 3:
            planning_prompt = PLANNING_PROMPT + f"Coding problem: {coding_prompt}"
            prompt = [planning_prompt, REFLECT_PROMPT, EXECUTE_PROMPT]

        response = agent.solve(system_prompt, prompt)
        code = extract_python_code(response)

        try:
            exec_globals = {"List": List, "Dict": Dict, "__builtins__": __builtins__}
            exec_globals = {}
            with io.StringIO() as buf, contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                exec(code, exec_globals)
            candidate_fn = next(
                v for k, v in exec_globals.items()
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
    parser.add_argument("--n", type=int, default=30, help="Number of coding problems to solve")
    parser.add_argument("--reasoning", type=int, default=1, help="Number of reasoning steps")
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
        accuracy = evaluate_code(args.n, agent, reasoning_steps=args.reasoning)
        print(f"Accuracy for {name}: {accuracy:.2%}")

if __name__ == "__main__":
    main()
