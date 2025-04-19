from datasets import load_dataset
import re

def extract_final_answer(answer_text: str) -> str:
    """
    Extracts the final answer from the GSM8K 'answer' field.
    This is usually in the format: '... The answer is 23.'
    """
    match = re.findall(r"<<.*?=(\-?\d+.*?)>>", answer_text)
    return match[-1].strip() if match else "N/A"

def main():
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="train")

    print("\nShowing 10 sample problems:\n" + "=" * 40)
    for i, item in enumerate(dataset.select(range(10))):
        question = item["question"]
        full_answer = item["answer"]
        print("question")
        print(question)
        print("full_answer")
        print(full_answer)
        extracted_answer = extract_final_answer(full_answer)

        print(f"Q{i+1}: {question.strip()}")
        print(f"A{i+1}: {extracted_answer}")
        print("-" * 40)

if __name__ == "__main__":
    main()
