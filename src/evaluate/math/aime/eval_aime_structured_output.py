from datasets import load_dataset
from typing import List, Dict
import json
import argparse
from tqdm import tqdm

from agents.structured_output.agent import Agent

from agents.structured_output.pydantic_structured_agent import PydanticAgent
from agents.structured_output.langchain_structured_agent import LangchainAgent
from agents.structured_output.crewai_structured_agent import CrewAIAgent
from agents.structured_output.autogen_structured_agent import AutoGenAgent
from agents.structured_output.direct_structured_call import BaselineLLMAgent


def evaluate_agent(agent: Agent, dataset, n_problems=10) -> List[Dict]:
    results = []
    for i, item in enumerate(tqdm(dataset, desc=f"Evaluating {agent.__class__.__name__}", total=n_problems)):
        if i >= n_problems:
            break

        problem = item["Problem"]
        solution = item["Answer"]

        response = agent.solve(problem)
        is_correct = str(response).strip() == str(solution).strip()

        results.append({
            "id": item["ID"],
            "problem": problem,
            "solution": solution,
            "response": response,
            "correct": is_correct,
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="Number of problems to evaluate")
    args = parser.parse_args()

    print("Loading AIME dataset...")
    dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train")

    agents = {
        "baseline": BaselineLLMAgent(),
        "reasoning": BaselineLLMAgent(model="o3-mini"),
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


if __name__ == "__main__":
    main()
