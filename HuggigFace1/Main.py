from __future__ import annotations

import io
import os
from unittest.mock import patch

import cv2
import torch
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers.dynamic_module_utils import get_imports
from webdriver_manager.chrome import ChromeDriverManager

model_name = "HuggingFaceM4/VLM_WebSight_finetuned"
def capture_screenshot(url):
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)
    screenshot = driver.get_screenshot_as_png()
    with open('website_screenshot.png', 'wb') as file:
        file.write(screenshot)
    driver.quit()
    return 'website_screenshot.png'

def analyze_visual_structure(screenshot_path):
    image = cv2.imread(screenshot_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    elements = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 50:
            if w > 200 and h < 100:
                element_type = 'header'
            elif w > 50 and h > 50 and w < 200 and h < 200:
                element_type = 'button'
            else:
                element_type = 'div'
            elements.append({
                'type': element_type,
                'position': (x, y),
                'size': (w, h)
            })
    cv2.imwrite('analyzed_structure.png', image)
    return elements, 'analyzed_structure.png'

def generate_html_css(elements):
    html_elements = []
    css_rules = []
    for i, elem in enumerate(elements):
        element_type = elem['type']
        x, y = elem['position']
        w, h = elem['size']
        elem_id = f'element-{i}'

        if element_type == 'header':
            html_elements.append(f'<header id="{elem_id}">Header {i}</header>')
        elif element_type == 'button':
            html_elements.append(f'<button id="{elem_id}">Button {i}</button>')
        elif element_type == 'div':
            html_elements.append(f'<div id="{elem_id}">Div {i}</div>')

        css_rules.append(f'''
            #{elem_id} {{
                position: absolute;
                top: {y}px;
                left: {x}px;
                width: {w}px;
                height: {h}px;
                background-color: #e0e0e0;
                border: 1px solid #ccc;
                text-align: center;
                line-height: {h}px; /* Center text vertically */
            }}
        ''')

    html_output = '<!DOCTYPE html>\n<html>\n<head>\n<style>\n'
    html_output += '\n'.join(css_rules)
    html_output += '\n</style>\n</head>\n<body>\n'
    html_output += '\n'.join(html_elements)
    html_output += '\n</body>\n</html>'

    return html_output

def generate_javascript(elements):
    js_code = []
    for i, elem in enumerate(elements):
        element_type = elem['type']
        elem_id = f'element-{i}'

        if element_type == 'button':
            js_code.append(f'''
                document.getElementById('{elem_id}').addEventListener('click', function() {{
                    alert('Button {elem_id} clicked!');
                }});
            ''')

    return '\n'.join(js_code)

def integrate_javascript_into_html(html_output, js_code):
    html_with_js = html_output.replace('</body>', f'<script>\n{js_code}\n</script>\n</body>')
    return html_with_js


def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
    imports = get_imports(filename)
    if not torch.cuda.is_available() and "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports
def generate_code_with_model(image_path):
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = AutoModelForCausalLM.from_pretrained("HuggingFaceM4/VLM_WebSight_finetuned",
                                                     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                                     device_map='auto' if torch.cuda.is_available() else None,
                                                     trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True)
    # Load and preprocess the image
    with open(image_path, 'rb') as f:
        image = Image.open(io.BytesIO(f.read()))

    # Generate code from image
    code = model(image)

    return code[0]['generated_text']  # Adjust based on the model's output format
def complete_flow(url):
    screenshot_path = capture_screenshot(url)
    elements, _ = analyze_visual_structure(screenshot_path)
    print(screenshot_path)

    # use hf
    generated_code = generate_code_with_model(screenshot_path)
    print(generated_code)
    # manual
    html_output = generate_html_css(elements)
    js_code = generate_javascript(elements)
    final_html_output = integrate_javascript_into_html(html_output, js_code)

    with open('output_with_model.html', 'w') as file:
        file.write(final_html_output)

    print("Completed processing. Output saved to 'output_with_model.html'.")


# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    complete_flow('https://help.pinterest.com/en/article/pinterest-lens')
