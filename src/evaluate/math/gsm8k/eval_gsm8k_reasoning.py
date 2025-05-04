from datasets import load_dataset
from typing import List, Dict
import json
import argparse
import re
from tqdm import tqdm

from agents.reasoning.agent import Agent

from agents.reasoning.pydantic_agent import PydanticAgent
from agents.reasoning.langchain_agent import LangchainAgent
from agents.reasoning.crewai_agent import CrewAIAgent
from agents.reasoning.autogen_agent import AutoGenAgent
from agents.reasoning.direct_call import BaselineLLMAgent


def extract_gsm8k_answer(answer_text: str) -> str:
    """
    Extracts the final numeric answer from GSM8K-style '<<...=ANSWER>>'.
    Returns the last number found in the double angle brackets.
    """
    match = re.findall(r"<<.*?=(\-?\d+.*?)>>", answer_text)
    return match[-1].strip() if match else "N/A"


def evaluate_agent(
    agent: Agent, dataset, n_problems=10, reasoning_steps: int = 1
) -> List[Dict]:
    results = []
    for i, item in enumerate(
        tqdm(dataset, desc=f"Evaluating {agent.__class__.__name__}", total=n_problems)
    ):
        if i >= n_problems:
            break

        problem = item["question"]
        solution = extract_gsm8k_answer(item["answer"])

        system_prompt = (
            "You are an agent specialized in solving math competition problems."
        )
        one_shot_prompt = f"Given the following problem, please tackle the problem step by step. Problem: {problem}"
        planning_prompt = (
            "Given the following problem, please formulate an approach "
            f"to the problem. Problem: {problem} - Do NOT solve it yet, "
            "just come up with a plan"
        )
        reflect_prompt = (
            "Great, now that you've written up a plan, I want you to "
            "carefully reflect on it. It's VERY important that you get "
            "it right. Please reconsider all of your steps and aim to correct "
            "any mistakes. These problems are VERY challenging, and it's very "
            "likely that you made a mistake somewhere."
        )
        execute_prompt = "Great, now given you plan, please solve the problem."

        if reasoning_steps == 1:
            prompt = [one_shot_prompt]
        elif reasoning_steps == 2:
            prompt = [planning_prompt, execute_prompt]
        elif reasoning_steps == 3:
            prompt = [planning_prompt, reflect_prompt, execute_prompt]

        try:
            response = agent.solve(system_prompt, prompts=prompt)
            is_correct = int(str(response).strip()) == int(str(solution).strip())
        except Exception as e:
            response = f"ERROR: {e}"
            is_correct = False

        results.append(
            {
                "id": f"gsm8k-{i + 1}",
                "problem": problem,
                "solution": solution,
                "response": response,
                "correct": is_correct,
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n", type=int, default=30, help="Number of problems to evaluate"
    )
    parser.add_argument(
        "--reasoning", type=int, default=1, help="Number of reasoning steps"
    )
    args = parser.parse_args()

    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="train")

    agents = {
        "baseline": BaselineLLMAgent(),
        "pydantic": PydanticAgent(),
        "langchain": LangchainAgent(),
        "crewai": CrewAIAgent(),
        "autogen": AutoGenAgent(),
    }

    for name, agent in agents.items():
        print(f"\n--- Evaluating {name} ---")
        results = evaluate_agent(agent, dataset, n_problems=args.n)
        accuracy = sum(r["correct"] for r in results) / len(results)
        print(f"Accuracy for {name}: {accuracy:.2%}")

        for r in results:
            print(
                f"[{r['id']}] Correct: {r['correct']} | Response: {r['response']} | Solution: {r['solution']}"
            )


if __name__ == "__main__":
    main()
