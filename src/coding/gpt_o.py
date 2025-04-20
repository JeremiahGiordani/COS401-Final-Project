import os
import openai
import env
import re

# Ensure your OpenAI API key is loaded from the environment
api_key = env.OPENAI_API_KEY

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

MODEL_NAME = "o4-mini-2025-04-16"

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

client = openai.OpenAI(api_key=api_key)

def query_gpt_o3_mini(system_prompt: str, user_prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":

    html = query_gpt_o3_mini(SYSTEM_PROMPT, UNICORN_TASK2)
    html = extract_html_only(html)
    output_path = os.path.join(OUTPUT_DIR, f"o4_unicorn.html")
    with open(output_path, "w") as f:
        f.write(html)