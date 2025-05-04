from datasets import load_dataset
from typing import List, Dict
import json
import argparse
import re
from tqdm import tqdm

from reasoning_agents.pydantic_agent import PydanticAgent
from reasoning_agents.langchain_agent import LangchainAgent
from reasoning_agents.crewai_agent import CrewAIAgent
from reasoning_agents.autogen_agent import AutoGenAgent
from reasoning_agents.direct_call import BaselineLLMAgent


def extract_gsm8k_answer(answer_text: str) -> str:
    """
    Extracts the final numeric answer from GSM8K-style '<<...=ANSWER>>'.
    Returns the last number found in the double angle brackets.
    """
    match = re.findall(r"<<.*?=(\-?\d+.*?)>>", answer_text)
    return match[-1].strip() if match else "N/A"


def evaluate_agent(agent, dataset, n_problems=10) -> List[Dict]:
    results = []
    for i, item in enumerate(tqdm(dataset, desc=f"Evaluating {agent.__class__.__name__}", total=n_problems)):
        if i >= n_problems:
            break

        problem = item["question"]
        solution = extract_gsm8k_answer(item["answer"])

        try:
            response = agent.solve(problem)
            is_correct = int(str(response).strip()) == int(str(solution).strip())
        except Exception as e:
            response = f"ERROR: {e}"
            is_correct = False

        results.append({
            "id": f"gsm8k-{i+1}",
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
            print(f"[{r['id']}] Correct: {r['correct']} | Response: {r['response']} | Solution: {r['solution']}")


if __name__ == "__main__":
    main()
