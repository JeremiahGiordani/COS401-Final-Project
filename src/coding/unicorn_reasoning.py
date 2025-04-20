import os
import re
from general_agents.pydantic_agent import PydanticAgent
from general_agents.langchain_agent import LangchainAgent
from general_agents.crewai_agent import CrewAIAgent
from general_agents.autogen_agent import AutoGenAgent
from general_agents.direct_call import BaselineLLMAgent


system_prompt1 = (
    "You are an AI competing against other agents in a unicorn-rendering competition using web technologies. "
    "Before you begin coding, you must first outline your creative and technical plan in detail. "
    "Describe how you will visually represent the unicorn, what HTML elements or structure you'll use, what CSS techniques or styles you will apply, and whether you will include any JavaScript (e.g., for animation). "
    "Consider ways to make your unicorn stand out — such as using gradients, animations, clever geometry, or symbolic representations. "
    "Be specific and thoughtful. Think step by step, and do not output any code yet — only your design plan."
)

system_prompt2 = (
    "You are a web developer. Based on the user's provided description of a unicorn implementation, you will now write the full code. "
    "Your task is to turn the design plan into a fully self-contained HTML file that renders the described unicorn. "
    "The file must include embedded HTML, CSS, and JavaScript (if needed), with no external dependencies.\n\n"
    "Guidelines:\n"
    "- Begin with <!DOCTYPE html>\n"
    "- Use <style> tags for CSS and <script> tags for JavaScript\n"
    "- The entire output must be just the code of the HTML file — do not include explanations, Markdown formatting, or commentary.\n\n"
    "Your output will be directly pasted into a file and rendered in a browser, so make it complete and functional."
)

user_prompt = "Please create the unicorn implementation"

OUTPUT_DIR = "outputs/unicorns_reasoning"

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
            html = agent.solve(system_prompts=[system_prompt1, system_prompt2], prompt=user_prompt)
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
