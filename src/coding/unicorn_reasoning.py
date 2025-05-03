import os
import re
from general_agents.pydantic_agent import PydanticAgent
from general_agents.langchain_agent import LangchainAgent
from general_agents.crewai_agent import CrewAIAgent
from general_agents.autogen_agent import AutoGenAgent
from general_agents.direct_call import BaselineLLMAgent

# Simple system prompt to define identity
system_prompt = "You are a helpful web developer."

# Multi-step user prompts
reasoning_prompt = (
    "You are participating in a space-scene-rendering competition where the goal is to create a scene of outer space using only inline SVG graphics. "
    "Before you begin coding, describe your design plan in detail. "
    "Think about what shapes (e.g., paths, circles, polygons) you'll use to represent the scene. "
    "Describe the visual style (e.g., color palette, gradients, symmetry), layering strategy, and any animation ideas. "
    "You may use SVG transforms, gradients, and stroke/fill styles. "
    "Do not write any code yet — just explain your design step by step."
)

robot_reasoning_prompt = (
    "You are participating in a robot-rendering competition where the goal is to create a robot using only inline SVG graphics. "
    "Before you begin coding, describe your design plan in detail. "
    "Think about what shapes (e.g., paths, circles, polygons) you'll use to represent the robot. "
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

robot_code_prompt = (
    "Now, based on your plan, write a complete HTML file that renders your robot using SVG. "
    "The entire robot must be created using inline SVG code — no images, external assets, or CSS classes. "
    "Use <svg> directly inside the <body> and define all shapes, styles, gradients, and animations inside the SVG tag. "
    "Begin with <!DOCTYPE html> and include everything in a single HTML file. "
    "Only output the HTML code — no commentary or Markdown formatting."
)


reflect_prompt = (
    "You just wrote an HTML file that renders a space scene using inline SVG. "
    "Now reflect critically on how the scene might appear when rendered in a browser. "
    "Are the parts correctly proportioned? Is the scene visually appealing? Do the colors and gradients work well together? "
    "Are any shapes misaligned, overlapping incorrectly, or visually awkward? "
    "List any potential problems with your SVG and what you would improve to make it look more like a space scene and more visually polished."
)

robot_reflect_prompt = (
    "You just wrote an HTML file that renders a robot using inline SVG. "
    "Now reflect critically on how the scene might appear when rendered in a browser. "
    "Are the parts correctly proportioned? Is the robot visually appealing? Do the colors and gradients work well together? "
    "Are any shapes misaligned, overlapping incorrectly, or visually awkward? "
    "List any potential problems with your SVG and what you would improve to make it look more like a robot and more visually polished."
    "Describe some improvements that you can make to make this better and more complex."
)

recode_prompt = (
    "Now revise your scene SVG based on your reflections. Fix any alignment issues, shape overlaps, color problems, or stylistic inconsistencies. "
    "Your goal is to make the scene look cleaner, more accurate, and more beautiful. "
    "Output a complete HTML file with everything embedded — start with <!DOCTYPE html> and include the updated SVG code inside the <body>. "
    "Do not include any extra text or formatting — only the raw HTML."
)

robot_recode_prompt = (
    "Now revise your scene SVG based on your reflections. Fix any alignment issues, shape overlaps, color problems, or stylistic inconsistencies. "
    "Your goal is to make the robot look cleaner, more accurate, more complex, and more beautiful. "
    "Output a complete HTML file with everything embedded — start with <!DOCTYPE html> and include the updated SVG code inside the <body>. "
    "Do not include any extra text or formatting — only the raw HTML."
)


svg_unicorn_oneshot_prompt = (
    "You are participating in a unicorn-rendering competition. "
    "Your task is to create a unicorn using only inline SVG, embedded inside a complete HTML file.\n\n"

    "The unicorn should be visually recognizable, creative, and well-structured. "
    "Use basic SVG elements such as <path>, <circle>, <rect>, <polygon>, and <line>. "
    "You may also use gradients, stroke styles, transformations, and animations inside the <svg> tag.\n\n"

    "Do not use any external images, CSS files, or JavaScript. All styling must be inside the SVG element.\n\n"

    "Your output must be a **fully self-contained HTML file** beginning with <!DOCTYPE html>, and the unicorn must be rendered using a single inline <svg> block within the <body>. "
    "Do not include any explanations, Markdown formatting, or comments — only output the raw HTML file."
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

svg_robot_oneshot_prompt = (
    "You are participating in a robot-rendering competition. "
    "Your task is to create a robot using only inline SVG, embedded inside a complete HTML file.\n\n"

    "The robot should be visually recognizable, creative, and well-structured. "
    "Use basic SVG elements such as <path>, <circle>, <rect>, <polygon>, and <line>. "
    "You may also use gradients, stroke styles, transformations, and animations inside the <svg> tag.\n\n"

    "Do not use any external images, CSS files, or JavaScript. All styling must be inside the SVG element.\n\n"

    "Your output must be a **fully self-contained HTML file** beginning with <!DOCTYPE html>, and the robot must be rendered using a single inline <svg> block within the <body>. "
    "Do not include any explanations, Markdown formatting, or comments — only output the raw HTML file."
    "Remember: You are participating in a head-to-head competition with other AI agents. "
    "You’re competing. Make it count! Show off your skills at coding to make the best robot!"
)


OUTPUT_DIR = "outputs/robot_svg/2_step"

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
        print(f"🦄 Generating Space scene with: {name}...")

        try:
            html = agent.solve(system_prompt=system_prompt, prompts=[robot_reasoning_prompt, robot_code_prompt])
            html = extract_html_only(html)

            if "<!doctype html" not in html.lower():
                print(html)
                raise ValueError("Agent did not return valid HTML.")

            output_path = os.path.join(OUTPUT_DIR, f"{name}_space.html")
            with open(output_path, "w") as f:
                f.write(html)

            print(f"✅ Saved to {output_path}")

        except Exception as e:
            print(f"❌ Error generating space for {name}: {e}")

if __name__ == "__main__":
    main()
