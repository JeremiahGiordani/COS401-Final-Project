import os
import re
from general_agents.pydantic_agent import PydanticAgent
from general_agents.langchain_agent import LangchainAgent
from general_agents.crewai_agent import CrewAIAgent
from general_agents.autogen_agent import AutoGenAgent
from general_agents.direct_call import BaselineLLMAgent

system_prompt = "You are a frontend engineer with strong visual design skills."

reasoning_prompt = (
    "You are tasked with building a weekly calendar scheduling interface using HTML, CSS, and JavaScript. "
    "The final interface should allow users to view and interact with a calendar grid, creating and managing events. "
    "Before you write any code, take time to reason about the implementation. Think carefully through the structure of the HTML layout, the required CSS styling, and the JavaScript logic needed to fulfill the requirements.\n\n"
    "Your goal in this step is to create a clear, structured plan that breaks down the project. "
    "For each of the following specification levels, explain how you would implement it and what challenges you might face:\n\n"
    "Basic Requirements:\n"
    "- Display weekdays Monday through Friday as columns\n"
    "- Show 30-minute time slots from 9:00 AM to 5:00 PM as rows\n"
    "- Allow users to click and drag to create an event block\n"
    "- Snap events to the nearest 30-minute interval\n"
    "- Display event time range inside the event block\n"
    "- Allow event deletion via click with confirmation prompt\n"
    "- All code should be in a single index.html file with embedded CSS and JS\n"
    "- Use only vanilla JavaScript (no external libraries)\n"
    "- ENSURE calendar is very clearly labeled!\n\n"
    "Intermediate Requirements:\n"
    "- Highlight the selected time range while dragging\n"
    "- Prevent overlapping events in the same day\n"
    "- Use CSS Grid or Flexbox for layout\n"
    "- Keep the calendar responsive to window resizing\n"
    "- Use clean, semantic HTML and modular CSS classes\n\n"
    "Advanced Requirements:\n"
    "- Allow naming events via a popup or input\n"
    "List out the high-level components you'll need, what HTML structure you plan to use, what JavaScript logic is required, and how you'll manage user interactions."
)


code_prompt = (
    "Now implement the weekly calendar scheduling interface based on your plan. "
    "Use HTML, CSS, and JavaScript to fulfill as many of the following specifications as possible. "
    "All code must be embedded in a single index.html file with no external libraries.\n\n"
    "Basic Requirements:\n"
    "- Display weekdays Monday through Friday as columns\n"
    "- Show 30-minute time slots from 9:00 AM to 5:00 PM as rows\n"
    "- Allow users to click and drag to create an event block\n"
    "- Snap events to the nearest 30-minute interval\n"
    "- Display event time range inside the event block\n"
    "- Allow event deletion via click with confirmation prompt\n"
    "- Use only vanilla JavaScript\n"
    "- ENSURE calendar is very clearly labeled!\n\n"
    "Intermediate Requirements:\n"
    "- Highlight the selected time range while dragging\n"
    "- Prevent overlapping events in the same day\n"
    "- Use CSS Grid or Flexbox for layout\n"
    "- Make the layout responsive\n"
    "- Write clean, semantic HTML and well-organized CSS\n\n"
    "Advanced Requirements:\n"
    "- Allow naming events via a popup or input\n"
    "Output only the complete code in a single index.html file."
)


reflect_prompt = (
    "Great, now that you've just written out how you plan to implement this, reflect deeper. Take a moment to think more carefully about what you will do and why.\n"
    "What will you be very careful about? What will you pay close attention to when implementing?"
)

recode_prompt = (
    "Now, revise your HTML file based on your reflections. "
    "Fix any layout issues, spacing inconsistencies, misaligned elements, or unclear section structure. "
    "Ensure the final layout more closely resembles a real Wikipedia article page. "
    "Use only embedded HTML and CSS, starting with <!DOCTYPE html>. "
    "Output only the final, updated HTML file — no commentary, no formatting, and no Markdown."
)

oneshot_prompt = (
    "Build a weekly calendar interface using HTML, CSS, and JavaScript. "
    "The calendar should display columns for Monday to Friday, and rows for time slots from 9:00 AM to 5:00 PM, divided into 30-minute intervals. "
    "Users should be able to click and drag within a column to create an event spanning a time range. "
    "The event box should snap to 30-minute intervals, show the start and end time (e.g., '10:00 AM – 11:30 AM'), and be styled with a colored background. "
    "Clicking an event should prompt the user to confirm deletion, and remove the event if confirmed. "
    "Ensure the layout is clean and responsive. Use only vanilla JavaScript, no external libraries. "
    "Include all code (HTML, CSS, JS) in a single index.html file."
    "Use only the code, starting with <!DOCTYPE html>. "
    "Output only the final, updated HTML file — no commentary, no formatting, and no Markdown."
)

complex_prompt = (
    "Create a weekly calendar scheduling interface using HTML, CSS, and JavaScript. "
    "This interface should display a grid for weekdays and allow users to create and manage events. "
    "Your solution should fulfill as many of the following specifications as possible:\n\n"
    "Basic Requirements:\n"
    "- Display weekdays Monday through Friday as columns\n"
    "- Show 30-minute time slots from 9:00 AM to 5:00 PM as rows\n"
    "- Allow users to click and drag to create an event block\n"
    "- Snap events to the nearest 30-minute interval\n"
    "- Display event time range inside the event block\n"
    "- Allow event deletion via click with confirmation prompt\n"
    "- All code should be in a single index.html file with embedded CSS and JS\n"
    "- Use only vanilla JavaScript (no external libraries)\n"
    "- ENSURE calendar is very clearly labeled!\n\n"
    "Intermediate Requirements:\n"
    "- Highlight the selected time range while dragging\n"
    "- Prevent overlapping events in the same day\n"
    "- Use CSS Grid or Flexbox for layout\n"
    "- Keep the calendar responsive to window resizing\n"
    "- Use clean, semantic HTML and modular CSS classes\n\n"
    "Advanced Requirements:\n"
    "- Allow naming events via a popup or input\n"
    "Use only the code, starting with <!DOCTYPE html>. "
    "Output only the final, updated HTML file — no commentary, no formatting, and no Markdown."
)




OUTPUT_DIR = "outputs/calendar/2_steps"

def extract_html_only(raw_response: str) -> str:
    match = re.search(r"```html\s*(.*?)```", raw_response, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else raw_response.strip()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    agents = {
        "baseline": BaselineLLMAgent(),
        # "reasoning": BaselineLLMAgent(model="o3-mini"),
        "pydantic": PydanticAgent(),
        "langchain": LangchainAgent(),
        "crewai": CrewAIAgent(),
        "autogen": AutoGenAgent(),
    }

    for name, agent in agents.items():
        print(f"Generating with: {name}...")

        try:
            html = agent.solve(system_prompt=system_prompt, prompts=[reasoning_prompt, code_prompt])
            html = extract_html_only(html)

            if "<!doctype html" not in html.lower():
                print(html)
                raise ValueError("Agent did not return valid HTML.")

            output_path = os.path.join(OUTPUT_DIR, f"{name}_calendar.html")
            with open(output_path, "w") as f:
                f.write(html)

            print(f"✅ Saved to {output_path}")

        except Exception as e:
            print(f"❌ Error generating wiki page for {name}: {e}")

if __name__ == "__main__":
    main()
