from PIL import Image, ImageDraw
import io
import base64

def is_image_prompt(prompt: str):
    keywords = ["draw", "generate image", "picture", "sketch", "art"]
    return any(k in prompt.lower() for k in keywords)

def generate_image(prompt: str):
    img = Image.new('RGB', (64, 64), color='white')
    d = ImageDraw.Draw(img)
    d.text((10, 25), "Gen", fill=(0, 0, 0))

    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()
