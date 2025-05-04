import os
import re
import argparse

from agents.reasoning.pydantic_agent import PydanticAgent
from agents.reasoning.langchain_agent import LangchainAgent
from agents.reasoning.crewai_agent import CrewAIAgent
from agents.reasoning.autogen_agent import AutoGenAgent
from agents.reasoning.direct_call import BaselineLLMAgent


system_prompt = "You are a helpful web developer."


reasoning_prompt = (
    "You are participating in a space-scene-rendering competition where the goal is to create a scene of outer space using only inline SVG graphics. "
    "Before you begin coding, describe your design plan in detail. "
    "Think about what shapes (e.g., paths, circles, polygons) you'll use to represent the scene. "
    "Describe the visual style (e.g., color palette, gradients, symmetry), layering strategy, and any animation ideas. "
    "You may use SVG transforms, gradients, and stroke/fill styles. "
    "Do not write any code yet — just explain your design step by step."
)


code_prompt = (
    "Now, based on your plan, write a complete HTML file that renders your space scene using SVG. "
    "The entire scene must be created using inline SVG code — no images, external assets, or CSS classes. "
    "Use <svg> directly inside the <body> and define all shapes, styles, gradients, and animations inside the SVG tag. "
    "Begin with <!DOCTYPE html> and include everything in a single HTML file. "
    "Only output the HTML code — no commentary or Markdown formatting."
)

reflect_prompt = (
    "Now, reflect on your plan. Do you see any problems or blind spots? Can you detail the process better?"
)


svg_space_oneshot_prompt = (
    "You are participating in a space-rendering competition. "
    "Your task is to create a scene of outer space using only inline SVG, embedded inside a complete HTML file.\n\n"

    "The scene should be visually recognizable, creative, and well-structured. "
    "Use basic SVG elements such as <path>, <circle>, <rect>, <polygon>, and <line>. "
    "You may also use gradients, stroke styles, transformations, and animations inside the <svg> tag.\n\n"

    "Do not use any external images, CSS files, or JavaScript. All styling must be inside the SVG element.\n\n"

    "Your output must be a **fully self-contained HTML file** beginning with <!DOCTYPE html>, and the space scene must be rendered using a single inline <svg> block within the <body>. "
    "Do not include any explanations, Markdown formatting, or comments — only output the raw HTML file."
)

ONE_STEP_PROMPT = [svg_space_oneshot_prompt]
TWO_STEP_PROMPTS = [reasoning_prompt, code_prompt]
THREE_STEP_PROMPTS = [reasoning_prompt, reflect_prompt, code_prompt]


OUTPUT_DIR = "../outputs/svg_outputs/"

def extract_html_only(raw_response: str) -> str:
    match = re.search(r"```html\s*(.*?)```", raw_response, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else raw_response.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning", type=int, default=1, help="Number of reasoning steps")
    args = parser.parse_args()

    if args.reasoning == 1:
        out_dir = OUTPUT_DIR + "1_step"
    elif args.reasoning == 2:
        out_dir = OUTPUT_DIR + "2_step"
    else:
        out_dir = OUTPUT_DIR + "3_step"

    os.makedirs(out_dir, exist_ok=True)

    agents = {
        "baseline": BaselineLLMAgent(),
        "pydantic": PydanticAgent(),
        "langchain": LangchainAgent(),
        "crewai": CrewAIAgent(),
        "autogen": AutoGenAgent(),
    }

    for name, agent in agents.items():
        print(f"Generating Space scene with: {name}...")

        try:
            if args.reasoning == 1:
                prompts=ONE_STEP_PROMPT
            elif args.reasoning == 2:
                prompts=TWO_STEP_PROMPTS
            else:
                prompts=THREE_STEP_PROMPTS

            html = agent.solve(system_prompt=system_prompt, prompts=prompts)
            html = extract_html_only(html)

            if "<!doctype html" not in html.lower():
                raise ValueError("Agent did not return valid HTML.")

            output_path = os.path.join(out_dir, f"{name}_space.html")
            with open(output_path, "w") as f:
                f.write(html)

            print(f"Saved to {output_path}")

        except Exception as e:
            print(f"Error generating space for {name}: {e}")

if __name__ == "__main__":
    main()
