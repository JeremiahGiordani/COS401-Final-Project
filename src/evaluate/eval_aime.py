from datasets import load_dataset
from typing import List, Dict
import json
import argparse
from tqdm import tqdm
import re

from openai import AzureOpenAI
import env

from general_agents.pydantic_agent import PydanticAgent
from general_agents.langchain_agent import LangchainAgent
from general_agents.crewai_agent import CrewAIAgent
from general_agents.autogen_agent import AutoGenAgent
from general_agents.direct_call import BaselineLLMAgent

def extract_json_only(raw_response: str) -> str:
    match = re.search(r"```json\s*(.*?)```", raw_response, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else raw_response.strip()

def extract_answer(agent_response):
    client = AzureOpenAI(
        api_key=env.API_KEY,
        azure_endpoint=env.BASE_URL,
        api_version=env.API_VERSION,
    )
    tool = {
        "type": "function",
        "function": {
            "name": "Answer",
            "description": "Return the answer to the math problem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "integer",
                        "description": "The final answer to the problem."
                    }
                },
                "required": ["answer"],
                "additionalProperties": False
            }
        }
    }
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {
                "role": "system",
                "content": "You have a long detailed solution. Please return just the answer, no explanation, in JSON format.",
            },
            {
                "role": "user",
                "content": agent_response,
            },
        ],
        tools=[tool],
        tool_choice={"type": "function", "function": {"name": "Answer"}}
    )
    arguments_str = response.choices[0].message.tool_calls[0].function.arguments
    return json.loads(arguments_str)["answer"]

def evaluate_agent(agent, dataset, n_problems=10) -> List[Dict]:
    results = []
    for i, item in enumerate(tqdm(dataset, desc=f"Evaluating {agent.__class__.__name__}", total=n_problems)):
        if i >= n_problems:
            break

        problem = item["Problem"]
        solution = item["Answer"]
        system_prompt = "You are an agent specialized in solving math competition problems."
        prompt1 = f"Given the following problem, please tackle the problem step by step. Problem: {problem}"
        # prompt2 = "Great, not please return the answer as a JSON. IMPORTANT! You must return only a JSON containing the answer in the following format {'answer': <answer>}"
        prompt2 = "Great, now that you've written up a solution, I want you to carefully reflect on it. It's VERY important that you get it right. Please reconsider all of your steps and aim to correct any mistakes. These problems are VERY challenging, and it's very likely that you made a mistake somewhere."
        response = agent.solve(system_prompt, prompts=[prompt1, prompt2])

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
        results = evaluate_agent(agent, dataset, n_problems=args.n)
        accuracy = sum(r["correct"] for r in results) / len(results)
        print(f"Accuracy for {name}: {accuracy:.2%}")

        for r in results:
            print(f"[{r['id']}] Correct: {r['correct']} | Response: {r['response']} | Solution: {r['solution']}")


if __name__ == "__main__":
    main()
