import os
import re
from general_agents.pydantic_agent import PydanticAgent
from general_agents.langchain_agent import LangchainAgent
from general_agents.crewai_agent import CrewAIAgent
from general_agents.autogen_agent import AutoGenAgent
from general_agents.direct_call import BaselineLLMAgent

system_prompt = "You are a frontend engineer with strong visual design skills."

reasoning_prompt = (
    "You are participating in a web design competition where your task is to replicate the layout of a Wikipedia article page using only HTML and CSS. The page should be on Large Language Models "
    "Before writing any code, describe in detail how you will structure the page. "
    "Include your planned approach to the header, sidebar, article title, table of contents, content sections, inline references, and footer. "
    "Think carefully about what layout techniques you will use (e.g., grid, flexbox), how you will style the typography and spacing, and how to represent Wikipedia-style structure. "
    "Do not write any code yet — only describe your plan step by step."
)

code_prompt = (
    "Now, based on your plan, write a complete HTML file that replicates the layout of a Wikipedia article page. "
    "It should include:\n"
    "- A header (e.g., Wikipedia branding as text)\n"
    "- A sidebar with links\n"
    "- An article title and subtitle\n"
    "- A table of contents\n"
    "- Multiple content sections with headings, paragraphs, and inline references\n"
    "- A footer\n\n"
    "Use only HTML and CSS — all CSS should be in a <style> tag, and your HTML must start with <!DOCTYPE html>. "
    "You may use placeholder text and boxes where needed, but do not use any images or external resources. "
    "Only output the raw HTML code with embedded styles. Do not include explanations, formatting, or Markdown fences."
)

reflect_prompt = (
    "You just wrote a complete HTML file intended to replicate a Wikipedia article layout. "
    "Now imagine the file is being rendered in a browser. Without seeing the output, reflect critically on how it might look. "
    "Does the layout feel clean and accurate? Are sections aligned properly? Does the sidebar appear in the correct position? "
    "Are the font sizes and spacing consistent with Wikipedia’s appearance? "
    "Identify any parts of the layout or style that may be inaccurate or unclear. Be specific and honest in your assessment."
)

recode_prompt = (
    "Now, revise your HTML file based on your reflections. "
    "Fix any layout issues, spacing inconsistencies, misaligned elements, or unclear section structure. "
    "Ensure the final layout more closely resembles a real Wikipedia article page. "
    "Use only embedded HTML and CSS, starting with <!DOCTYPE html>. "
    "Output only the final, updated HTML file — no commentary, no formatting, and no Markdown."
)

oneshot_prompt = (
    "You are a skilled web developer. "
    "Your task is to replicate the layout of a Wikipedia article page (e.g., https://en.wikipedia.org/wiki/Large_language_model) using only HTML and CSS. "
    "Your output should include a header with Wikipedia branding (text only), a left-hand sidebar with links, a table of contents, and multiple content sections with headings and inline references. "
    "Use only embedded CSS (via <style>) and do not use any images or external resources. "
    "Begin your output with <!DOCTYPE html> and output only the raw HTML code — no formatting or explanations."
)


OUTPUT_DIR = "outputs/wikipedia/1_shot"

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
            html = agent.solve(system_prompt=system_prompt, prompts=[oneshot_prompt])
            html = extract_html_only(html)

            if "<!doctype html" not in html.lower():
                print(html)
                raise ValueError("Agent did not return valid HTML.")

            output_path = os.path.join(OUTPUT_DIR, f"{name}_wikipedia.html")
            with open(output_path, "w") as f:
                f.write(html)

            print(f"✅ Saved to {output_path}")

        except Exception as e:
            print(f"❌ Error generating wiki page for {name}: {e}")

if __name__ == "__main__":
    main()
