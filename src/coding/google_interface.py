import os
import re
from general_agents.pydantic_agent import PydanticAgent
from general_agents.langchain_agent import LangchainAgent
from general_agents.crewai_agent import CrewAIAgent
from general_agents.autogen_agent import AutoGenAgent
from general_agents.direct_call import BaselineLLMAgent

# Simple system prompt to define identity
system_prompt = "You are a frontend engineer with strong visual design skills."

# Multi-step user prompts
reasoning_prompt = (
    "You are tasked with replicating the visual layout of the Google.com homepage using only HTML and CSS. Before writing code, describe your plan in detail: What layout will you use? How will you position the logo, search bar, and buttons? What fonts and spacing will you replicate? Think carefully about structure and style."
)

code_prompt = (
    "Now, implement your plan. Write a complete HTML file that replicates the Google homepage. Start with <!DOCTYPE html>, use embedded CSS inside <style> tags, and use no external resources or libraries. Just do your best to replicate the look and layout of the real homepage using plain HTML and CSS."
)

reflect_prompt = (
    "Without seeing the output, reflect critically: What parts of your implementation may be inaccurate? Are the logo size, search bar width, button spacing, or alignment potentially off? What would you improve?"
)

recode_prompt = (
    "Now revise your code based on your reflections. Output a new, complete HTML file that more closely matches the visual layout of the Google homepage."
)

one_shot_prompt = (
    """You are a skilled frontend developer.

    Your task is to replicate the visual layout of the Google.com homepage using only HTML and CSS. 
    Do your best to match the structure, spacing, font sizes, and overall appearance as closely as possible.

    Constraints:
    - Use embedded CSS inside <style> tags (no external stylesheets)
    - Do not use any external assets — you may use placeholder text/images (e.g., “Google” as plain text or placeholder logo)
    - Include a centered search bar, two buttons beneath it, and links at the top-right and bottom of the page
    - Mimic the spacing, alignment, and styling as seen on https://www.google.com

    Your output must be a **complete, self-contained HTML file**, beginning with <!DOCTYPE html>. Do not include any explanations, formatting, or commentary — only the raw HTML code.
    """
)

OUTPUT_DIR = "outputs/google/2_steps"

def extract_html_only(raw_response: str) -> str:
    match = re.search(r"```html\s*(.*?)```", raw_response, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else raw_response.strip()

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
        print(f"Generating Google with: {name}...")

        try:
            html = agent.solve(system_prompt=system_prompt, prompts=[reasoning_prompt, code_prompt])
            html = extract_html_only(html)

            if "<!doctype html" not in html.lower():
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
