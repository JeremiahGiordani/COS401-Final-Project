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

UNICORN_TASK3 =  (
    "You are participating in a unicorn-rendering competition. "
    "Your task is to create a unicorn using only inline SVG, embedded inside a complete HTML file.\n\n"

    "The unicorn should be visually recognizable, creative, and well-structured. "
    "Use basic SVG elements such as <path>, <circle>, <rect>, <polygon>, and <line>. "
    "You may also use gradients, stroke styles, transformations, and animations inside the <svg> tag.\n\n"

    "Do not use any external images, CSS files, or JavaScript. All styling must be inside the SVG element.\n\n"

    "Your output must be a **fully self-contained HTML file** beginning with <!DOCTYPE html>, and the unicorn must be rendered using a single inline <svg> block within the <body>. "
    "Do not include any explanations, Markdown formatting, or comments — only output the raw HTML file."
)

YOUTUBE_TASK = (
    "You are a highly skilled frontend developer. Your task is to replicate the layout and design of the YouTube homepage as precisely as possible using only HTML and CSS.\n\n"

    "You should aim to reproduce the page as an **identical copy** — not an approximation. Everything should be as close as possible to the real page: the layout, the spacing, the font styles, the font sizes, the icon placement, the color scheme, the size and alignment of thumbnails, the button shapes, the sidebar structure — every visual detail.\n\n"

    "Requirements:\n"
    "- Use embedded CSS in a <style> tag (no external stylesheets)\n"
    "- Start the file with <!DOCTYPE html>\n"
    "- Do not include JavaScript\n"
    "You should use the following images for video thumbnails:\n"
    "-https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Felis_catus-cat_on_snow.jpg/1599px-Felis_catus-cat_on_snow.jpg?20190920093216\n"
    "-https://upload.wikimedia.org/wikipedia/commons/thumb/8/86/Eurasian_blue_tit_Lancashire.jpg/640px-Eurasian_blue_tit_Lancashire.jpg\n"
    "-https://upload.wikimedia.org/wikipedia/commons/thumb/9/98/Onion_domes_of_Cathedral_of_the_Annunciation.JPG/640px-Onion_domes_of_Cathedral_of_the_Annunciation.JPG\n"
    "-https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/Soyuz_TMA-16_approaching_ISS.jpg/640px-Soyuz_TMA-16_approaching_ISS.jpg\n"
    "-https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/German-Shepherd-dog-rainbow-shake.jpg/640px-German-Shepherd-dog-rainbow-shake.jpg\n"

    # "You can use the following for logos on the page:\n"
    # "- Youtube Logo: https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_Logo_2017.svg\n"
    # "- Home Icon: https://upload.wikimedia.org/wikipedia/commons/thumb/3/30/Home_free_icon.svg/640px-Home_free_icon.svg.png\n"
    # "- History Icon: https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/VK_icons_history_outline_56.svg/640px-VK_icons_history_outline_56.svg.png\n"
    # "- Bell Icon: https://upload.wikimedia.org/wikipedia/commons/thumb/a/ad/Bell-solid.svg/640px-Bell-solid.svg.png\n"
    # "- Magnifying Glass Icon: https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Magnifying_glass_icon.svg/640px-Magnifying_glass_icon.svg.png\n"
    # "- Upload Icon: https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/Noun_Project_cloud_upload_icon_411593_cc.svg/640px-Noun_Project_cloud_upload_icon_411593_cc.svg.png\n"
    # "- Clock Icon: https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/OOjs_UI_icon_clock.svg/640px-OOjs_UI_icon_clock.svg.png\n"
    # "- Profile Icon: https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/VK_icons_profile_28.svg/640px-VK_icons_profile_28.svg.png\n"
    "DO NOT USE ANY OTHER IMAGES OR ICONS. You can ONLY use the ones provided here. Keep in mind that you should be particular about how you size them on the page.\n"

    "- Do not include commentary, Markdown formatting, or code fences — only output raw HTML code\n\n"

    "The result should closely match the actual YouTube homepage in appearance, structure, and detail."
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
    "You are participating in a robot-rendering competition."
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

SYSTEM_PROMPT = (
    "You are a creative frontend developer. "
)

MODEL_NAME = "o4-mini-2025-04-16"

OUTPUT_DIR = "outputs/robot_svg"

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

    html = query_gpt_o3_mini(SYSTEM_PROMPT, svg_robot_oneshot_prompt)
    html = extract_html_only(html)
    output_path = os.path.join(OUTPUT_DIR, f"o4_robot.html")
    with open(output_path, "w") as f:
        f.write(html)