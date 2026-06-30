import mammoth
from markdownify import markdownify as md
import os

docx_path = "HAMR_Project_Concept.docx"
output_md = "HAMR_Project_Concept.md"
image_dir = "HAMR_Project_Concept_images"

if not os.path.exists(image_dir):
    os.makedirs(image_dir)

def convert_image(image):
    with image.open() as image_bytes:
        image_name = f"image_{len(os.listdir(image_dir)) + 1}.{image.content_type.split('/')[1]}"
        image_path = os.path.join(image_dir, image_name)
        with open(image_path, "wb") as f:
            f.write(image_bytes.read())
    return {"src": image_path}

with open(docx_path, "rb") as docx_file:
    result = mammoth.convert_to_html(docx_file, convert_image=mammoth.images.img_element(convert_image))
    html = result.value
    messages = result.messages

markdown_text = md(html, heading_style="ATX")

with open(output_md, "w", encoding="utf-8") as f:
    f.write(markdown_text)

print(f"Conversion complete! Saved to {output_md}")
if messages:
    print("Messages:")
    for message in messages:
        print(message)
