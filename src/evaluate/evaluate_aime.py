from datasets import load_dataset
from typing import List, Dict
import json
from agents.pydantic_base_agent import PydanticAgent
# from agents.langchain import LangchainAgent
# from agents.crewai import CrewAIAgent
# from agents.autogen import AutoGenAgent

def evaluate_agent(agent, dataset, n_problems=10) -> List[Dict]:
    results = []
    for i, item in enumerate(dataset):
        if i >= n_problems:
            break

        problem = item["problem"]
        solution = item["solution"]

        try:
            response = agent.solve(problem)
            is_correct = str(response).strip() == str(solution).strip()
        except Exception as e:
            response = f"ERROR: {e}"
            is_correct = False

        results.append({
            "problem": problem,
            "solution": solution,
            "response": response,
            "correct": is_correct,
        })

    return results

def main():
    dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    print(type(dataset))  # <class 'datasets.arrow_dataset.Dataset'>
    print(f"Number of entries: {len(dataset)}\n")

    # Convert to list of dicts and dump as JSON string
    dataset_as_json = dataset.to_dict()  # list of column-wise lists
    dataset_as_dicts = [dict(zip(dataset_as_json.keys(), values)) for values in zip(*dataset_as_json.values())]

    print(json.dumps(dataset_as_dicts[:3], indent=2))  # Print just the first 3 for sanity
    with open("aime.json", "w") as f:
        json.dump(dataset_as_dicts, f, indent=4)
    return

    agents = {
        "pydantic": PydanticAgent(),
        "langchain": LangchainAgent(),
        "crewai": CrewAIAgent(),
        "autogen": AutoGenAgent(),
    }

    for name, agent in agents.items():
        print(f"\n--- Evaluating {name} ---")
        results = evaluate_agent(agent, dataset)
        accuracy = sum(r["correct"] for r in results) / len(results)
        print(f"Accuracy for {name}: {accuracy:.2%}")
        for r in results:
            print(f"Problem: {r['problem'][:50]}... | Correct: {r['correct']}")

if __name__ == "__main__":
    main()
