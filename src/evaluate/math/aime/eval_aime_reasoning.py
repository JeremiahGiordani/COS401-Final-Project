from datasets import load_dataset
from typing import List, Dict
import argparse
from tqdm import tqdm
import re

from evaluate.utils import extract_answer

from agents.reasoning.agent import Agent

from agents.reasoning.pydantic_agent import PydanticAgent
from agents.reasoning.langchain_agent import LangchainAgent
from agents.reasoning.crewai_agent import CrewAIAgent
from agents.reasoning.autogen_agent import AutoGenAgent
from agents.reasoning.direct_call import BaselineLLMAgent

ONE_SHOT_PROMPT = "Given the following problem, please tackle the problem step by step"

PLANNING_PROMPT = (
    "Given the following problem, please formulate an approach "
    "to the problem. - Do NOT solve it yet, "
    "just come up with a plan"
)
REFLECT_PROMPT = (
    "Great, now that you've written up a plan, I want you to "
    "carefully reflect on it. It's VERY important that you get "
    "it right. Please reconsider all of your steps and aim to correct "
    "any mistakes. These problems are VERY challenging, and it's very "
    "likely that you made a mistake somewhere."
)
EXECUTE_PROMPT = "Great, now given you plan, please solve the problem."

def extract_json_only(raw_response: str) -> str:
    match = re.search(r"```json\s*(.*?)```", raw_response, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else raw_response.strip()

def evaluate_agent(agent: Agent, dataset, n_problems=10, reasoning_steps: int=1) -> List[Dict]:
    results = []
    for i, item in enumerate(tqdm(dataset, desc=f"Evaluating {agent.__class__.__name__}", total=n_problems)):
        if i >= n_problems:
            break

        problem = item["Problem"]
        solution = item["Answer"]
        system_prompt = "You are an agent specialized in solving math competition problems."
    
        if reasoning_steps == 1:
            one_shot_prompt = ONE_SHOT_PROMPT + f"Problem: {problem}"
            prompt = [one_shot_prompt]
        elif reasoning_steps == 2:
            planning_prompt = PLANNING_PROMPT + f"Problem: {problem}"
            prompt = [planning_prompt, EXECUTE_PROMPT]
        elif reasoning_steps == 3:
            planning_prompt = PLANNING_PROMPT + f"Problem: {problem}"
            prompt = [planning_prompt, REFLECT_PROMPT, EXECUTE_PROMPT]

        response = agent.solve(system_prompt, prompts=prompt)

        try:
            answer = extract_answer(response)
        except Exception as ex:
            print("Exception")
            print(f"Response: {response}")
            print(f"Solution: {solution}")
            continue

        is_correct = str(answer).strip() == str(solution).strip()

        results.append({
            "id": item["ID"],
            "problem": problem,
            "solution": solution,
            "response": answer,
            "correct": is_correct,
        })

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="Number of problems to evaluate")
    parser.add_argument("--reasoning", type=int, default=1, help="Number of reasoning steps")
    args = parser.parse_args()

    print("Loading AIME dataset...")
    dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train")

    agents = {
        "baseline": BaselineLLMAgent(),
        "pydantic": PydanticAgent(),
        "langchain": LangchainAgent(),
        "crewai": CrewAIAgent(),
        "autogen": AutoGenAgent(),
    }

    for name, agent in agents.items():
        print(f"\n--- Evaluating {name} ---")
        results = evaluate_agent(agent, dataset, n_problems=args.n, reasoning_steps=args.reasoning)
        accuracy = sum(r["correct"] for r in results) / len(results)
        print(f"Accuracy for {name}: {accuracy:.2%}")

        for r in results:
            print(f"[{r['id']}] Correct: {r['correct']} | Response: {r['response']} | Solution: {r['solution']}")


if __name__ == "__main__":
    main()
