import os
import re
from general_agents.pydantic_agent import PydanticAgent
from general_agents.langchain_agent import LangchainAgent
from general_agents.crewai_agent import CrewAIAgent
from general_agents.autogen_agent import AutoGenAgent
from general_agents.direct_call import BaselineLLMAgent

system_prompt = "You are a frontend engineer with strong visual design skills."

reasoning_prompt = (
    "You are participating in a web design competition where your task is to replicate the layout of the YouTube homepage using only HTML and CSS. "
    "You should aim to reproduce the page as an **identical copy** — not an approximation. Everything should be as close as possible to the real page"
    "Before writing any code, describe your design plan in detail. "
    "Include how you will structure the top navigation bar, sidebar, and grid of video thumbnails. "
    "Describe how you will use HTML elements to represent the video cards, titles, channels, and view counts. "
    "Also explain how you will style everything using CSS — layout (grid or flex), spacing, font sizing, etc. "
    "You may use external image URLs (e.g., placeholder images or thumbnails from Wikimedia Commons). "
    "Do not write any code yet — only describe your implementation plan step by step."
)


code_prompt = (
    "Now write a complete HTML file that replicates the layout of the YouTube homepage based on your design plan. "
    "Your page should include:\n"
    "- A fixed top navigation bar with a logo (as text or image), search bar, and icons\n"
    "- A left sidebar with navigation links\n"
    "- A main area with a grid of video cards, each showing a thumbnail (via external image link), a title, a channel name, and view count\n\n"
    "You should use the following images for video thumbnails:\n"
    "-https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Felis_catus-cat_on_snow.jpg/1599px-Felis_catus-cat_on_snow.jpg?20190920093216\n"
    "-https://upload.wikimedia.org/wikipedia/commons/thumb/8/86/Eurasian_blue_tit_Lancashire.jpg/640px-Eurasian_blue_tit_Lancashire.jpg\n"
    "-https://upload.wikimedia.org/wikipedia/commons/thumb/9/98/Onion_domes_of_Cathedral_of_the_Annunciation.JPG/640px-Onion_domes_of_Cathedral_of_the_Annunciation.JPG\n"
    "-https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/Soyuz_TMA-16_approaching_ISS.jpg/640px-Soyuz_TMA-16_approaching_ISS.jpg\n"
    "-https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/German-Shepherd-dog-rainbow-shake.jpg/640px-German-Shepherd-dog-rainbow-shake.jpg\n"

    "You can use the following for logos on the page:\n"
    "- Youtube Logo: https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_Logo_2017.svg\n"
    "- Home Icon: https://upload.wikimedia.org/wikipedia/commons/thumb/3/30/Home_free_icon.svg/640px-Home_free_icon.svg.png\n"
    "- History Icon: https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/VK_icons_history_outline_56.svg/640px-VK_icons_history_outline_56.svg.png\n"
    "- Bell Icon: https://upload.wikimedia.org/wikipedia/commons/thumb/a/ad/Bell-solid.svg/640px-Bell-solid.svg.png\n"
    "- Magnifying Glass Icon: https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Magnifying_glass_icon.svg/640px-Magnifying_glass_icon.svg.png\n"
    "- Upload Icon: https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/Noun_Project_cloud_upload_icon_411593_cc.svg/640px-Noun_Project_cloud_upload_icon_411593_cc.svg.png\n"
    "- Clock Icon: https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/OOjs_UI_icon_clock.svg/640px-OOjs_UI_icon_clock.svg.png\n"
    "- Profile Icon: https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/VK_icons_profile_28.svg/640px-VK_icons_profile_28.svg.png\n"

    "Use only HTML and CSS. All CSS should be inside a <style> tag, and the HTML should begin with <!DOCTYPE html>. "
    "You may use external image links (e.g., Wikimedia, placeholder.com), but no JavaScript or interactive behavior. "
    "Only output the raw HTML — no Markdown, explanations, or extra formatting."
)


reflect_prompt = (
    "You just wrote an HTML file that replicates the layout of the YouTube homepage. "
    "Now reflect on how the layout might look when rendered in a browser. "
    "Are the navigation bar and sidebar positioned correctly? Does the video grid align well? "
    "Are the thumbnail images the right size? Is the font sizing and spacing consistent? "
    "Identify any issues you expect could cause the layout to look off, and note anything you'd want to fix or improve in a second version."
)


recode_prompt = (
    "Now revise your HTML file to address the issues you identified in your reflection. "
    "Improve any layout, spacing, alignment, or styling problems so the page looks more like the real YouTube homepage. "
    "Ensure that all video cards are well-aligned and visually balanced. "
    "Output a full, self-contained HTML file starting with <!DOCTYPE html>, using <style> tags for all CSS. "
    "Only output the raw HTML code — no commentary, Markdown, or explanations."
)


oneshot_prompt = (
    "You are a skilled frontend developer. Your task is to replicate the layout of the YouTube homepage using only HTML and CSS. "
    "Include:\n"
    "- A top navigation bar with a logo (text or image), search bar, and icon buttons\n"
    "- A sidebar with links like Home, Subscriptions, and Library\n"
    "- A grid of video thumbnails with video title, channel name, and view count below each\n\n"
    "Use the following images:\n"
    "-https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Felis_catus-cat_on_snow.jpg/1599px-Felis_catus-cat_on_snow.jpg?20190920093216\n"
    "-https://upload.wikimedia.org/wikipedia/commons/thumb/8/86/Eurasian_blue_tit_Lancashire.jpg/640px-Eurasian_blue_tit_Lancashire.jpg\n"
    "-https://upload.wikimedia.org/wikipedia/commons/thumb/9/98/Onion_domes_of_Cathedral_of_the_Annunciation.JPG/640px-Onion_domes_of_Cathedral_of_the_Annunciation.JPG\n"
    "-https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/Soyuz_TMA-16_approaching_ISS.jpg/640px-Soyuz_TMA-16_approaching_ISS.jpg\n"
    "-https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/German-Shepherd-dog-rainbow-shake.jpg/640px-German-Shepherd-dog-rainbow-shake.jpg\n"
    "All code should be inside one HTML file, starting with <!DOCTYPE html> and using <style> tags for all CSS. "
    "No JavaScript or external stylesheets. Only output the code — no Markdown or explanations."
    "Make up video names, make up channels, do whatever you want to do in order to make it look realistic"
    "IMPORTANT! It should look as close to the actual, real Youtube interface as possible."
)

oneshot_prompt = (
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

    "You can use the following for logos on the page:\n"
    "- Youtube Logo: https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_Logo_2017.svg\n"
    "- Home Icon: https://upload.wikimedia.org/wikipedia/commons/thumb/3/30/Home_free_icon.svg/640px-Home_free_icon.svg.png\n"
    "- History Icon: https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/VK_icons_history_outline_56.svg/640px-VK_icons_history_outline_56.svg.png\n"
    "- Bell Icon: https://upload.wikimedia.org/wikipedia/commons/thumb/a/ad/Bell-solid.svg/640px-Bell-solid.svg.png\n"
    "- Magnifying Glass Icon: https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Magnifying_glass_icon.svg/640px-Magnifying_glass_icon.svg.png\n"
    "- Upload Icon: https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/Noun_Project_cloud_upload_icon_411593_cc.svg/640px-Noun_Project_cloud_upload_icon_411593_cc.svg.png\n"
    "- Clock Icon: https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/OOjs_UI_icon_clock.svg/640px-OOjs_UI_icon_clock.svg.png\n"
    "- Profile Icon: https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/VK_icons_profile_28.svg/640px-VK_icons_profile_28.svg.png\n"
    "DO NOT USE ANY OTHER IMAGES OR ICONS. You can ONLY use the ones provided here.\n"
    "Keep in mind that you should be particular about how you size them on the page.\n"

    "- Do not include commentary, Markdown formatting, or code fences — only output raw HTML code\n\n"

    "The result should closely match the actual YouTube homepage in appearance, structure, and detail."
)




OUTPUT_DIR = "outputs/youtube/2_part"

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
        print(f"Generating Youtube with: {name}...")

        try:
            html = agent.solve(system_prompt=system_prompt, prompts=[reasoning_prompt, code_prompt])
            html = extract_html_only(html)

            if "<!doctype html" not in html.lower():
                print(html)
                raise ValueError("Agent did not return valid HTML.")

            output_path = os.path.join(OUTPUT_DIR, f"{name}_youtube.html")
            with open(output_path, "w") as f:
                f.write(html)

            print(f"✅ Saved to {output_path}")

        except Exception as e:
            print(f"❌ Error generating youtube page for {name}: {e}")

if __name__ == "__main__":
    main()
