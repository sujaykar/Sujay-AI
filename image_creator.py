# image_creator.py

import os
import hashlib
from openai import OpenAI
from PIL import Image
import requests
from io import BytesIO

class DalleImageGenerator:
    def __init__(self, model: str = "dall-e-3", size: str = "1024x1024"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment variables.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.size = size
        self.default_style = "highly detailed digital illustration"

    def generate_refined_prompt(self, slide) -> str:
        gpt_prompt = (
            "Create a vivid DALL·E prompt for an image that matches this slide.\n"
            f"Title: {slide['title']}\nContent: {slide['content']}"
        )
        try:
            completion = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You're an AI image prompt writer."},
                    {"role": "user", "content": gpt_prompt}
                ]
            )
            return completion.choices[0].message.content.strip()
        except Exception:
            return f"{self.default_style} image for the slide titled '{slide['title']}'"

    def refine_prompt(self, prompt: str) -> str:
        return f"Simple illustration. {prompt[:150]}"

    def get_prompt_hash(self, prompt: str) -> str:
        return hashlib.md5(prompt.encode()).hexdigest()

    def generate_image(self, prompt: str) -> str:
        styled_prompt = f"{self.default_style}, {prompt}"
        try:
            response = self.client.images.generate(
                model=self.model,
                prompt=styled_prompt,
                size=self.size,
                n=1
            )
            return response.data[0].url
        except Exception:
            try:
                refined = self.refine_prompt(prompt)
                styled_prompt = f"{self.default_style}, {refined}"
                response = self.client.images.generate(
                    model=self.model,
                    prompt=styled_prompt,
                    size=self.size,
                    n=1
                )
                return response.data[0].url
            except Exception as retry_e:
                raise RuntimeError(f"Both original and refined prompt failed: {str(retry_e)}")

    def download_image(self, url: str, save_path: str) -> None:
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            image.save(save_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download or save image: {str(e)}")

    def generate_images_for_slides(self, slides: list, save_dir: str = "generated_images") -> list:
        updated_slides = []
        for i, slide in enumerate(slides):
            image_prompt = slide.get("image_prompt") or self.generate_refined_prompt(slide)
            prompt_hash = self.get_prompt_hash(image_prompt)
            filename = f"{prompt_hash}.png"
            filepath = os.path.join(save_dir, filename)

            if os.path.exists(filepath):
                slide['image_path'] = filepath
                slide['image_url'] = None
            else:
                try:
                    image_url = self.generate_image(image_prompt)
                    self.download_image(image_url, filepath)
                    slide['image_url'] = image_url
                    slide['image_path'] = filepath
                except Exception as e:
                    slide['image_url'] = None
                    slide['image_path'] = None
                    slide['error'] = str(e)

            updated_slides.append(slide)
        return updated_slides
