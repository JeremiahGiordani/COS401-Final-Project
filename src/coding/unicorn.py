import os
import re
from general_agents.pydantic_agent import PydanticAgent
from general_agents.langchain_agent import LangchainAgent
from general_agents.crewai_agent import CrewAIAgent
from general_agents.autogen_agent import AutoGenAgent
from general_agents.direct_call import BaselineLLMAgent


UNICORN_TASK = (
    "Write a complete HTML file that renders a unicorn. "
    "Use HTML, CSS, and JavaScript as needed. "
    "All code must be embedded in the HTML file (no external resources). "
    "Begin with <!DOCTYPE html> and include all code directly. "
    "Only output the code of the HTML file. Do not include explanations or formatting."
)

UNICORN_TASK2 = (
    "You are participating in a head-to-head competition with other AI agents. "
    "Your goal is to impress a panel of human judges by generating the most visually creative and technically impressive unicorn using only web technologies. "
    "You may use HTML, CSS, and JavaScript to achieve this. Try to make your unicorn stand out with details, animations, color, or clever use of code. "
    "This is your chance to shine — think creatively and aim to outperform the others.\n\n"

    "Please output a **complete HTML file** that renders a unicorn. The unicorn can be abstract or realistic, stylized or symbolic. "
    "All code must be embedded directly in the file — no external links or dependencies.\n\n"

    "Guidelines:\n"
    "- Begin the file with <!DOCTYPE html>\n"
    "- Include all CSS in <style> tags\n"
    "- Include any JavaScript in <script> tags\n"
    "- The entire output must be just the code of the HTML file (no explanations, formatting, or commentary)\n\n"

    "Remember: you’re competing. Make it count."
)

SYSTEM_PROMPT = (
    "You are a creative frontend developer. "
    "You write beautiful, self-contained web pages on request."
)

OUTPUT_DIR = "outputs/unicorns_compete"

def extract_html_only(raw_response: str) -> str:
    """
    Extracts HTML from a string, removing ```html fences if present.
    """
    # Match content inside ```html ... ```
    match = re.search(r"```html\s*(.*?)```", raw_response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return raw_response.strip()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    agents = {
        "baseline": BaselineLLMAgent(),
        "pydantic": PydanticAgent(),
        "langchain": LangchainAgent(),
        "crewai": CrewAIAgent(),
        "autogen": AutoGenAgent(),
    }

    for name, agent in agents.items():
        print(f"Generating unicorn with: {name}...")
        try:
            html = agent.solve(system_prompts=[SYSTEM_PROMPT], prompt=UNICORN_TASK2)
            html = extract_html_only(html)

            # Ensure output is just the HTML code
            if "<!DOCTYPE html" not in html:
                print(html)
                raise ValueError("Agent did not return valid HTML.")

            output_path = os.path.join(OUTPUT_DIR, f"{name}_unicorn.html")
            with open(output_path, "w") as f:
                f.write(html)

            print(f"✅ Saved to {output_path}")

        except Exception as e:
            print(f"❌ Error generating unicorn for {name}: {e}")


if __name__ == "__main__":
    main()
