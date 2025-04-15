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
    """Ensure prompt meets DALL-E 3 requirements"""
    # Remove any potentially blocked content
    cleaned = re.sub(r'[^a-zA-Z0-9\s,.!?-]', '', prompt)
    
    # Ensure proper length (DALL-E 3 has 400 character limit)
    if len(cleaned) > 380:  # Leave room for our style prefix
        cleaned = cleaned[:375] + "..."
        # Add basic style if missing
    if not any(style_word in cleaned.lower() 
              for style_word in ["illustration", "photo", "painting", "render"]):
        cleaned = f"digital illustration of {cleaned}"
    
    return cleaned
    
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
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Validate it's actually an image
        image = Image.open(BytesIO(response.content))
        image.verify()  # Verify it's a complete image file
        
        # Convert to RGB if needed (for PNG compatibility)
        if image.mode in ('RGBA', 'LA'):
            image = image.convert('RGB')
            
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image.save(save_path, format='PNG', quality=95)
        
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to download image: {str(e)}")
    except (IOError, Image.DecompressionBombError) as e:
        raise RuntimeError(f"Invalid image data: {str(e)}")
    finally:
        if 'image' in locals():
            image.close()

    
    def generate_images_for_slides(self, slides: list, save_dir: str = "generated_images") -> list:
    if not slides:
        return []
        
    os.makedirs(save_dir, exist_ok=True)
    updated_slides = []
    
    for i, slide in enumerate(slides, 1):
        try:
            # Generate or get cached image
            image_prompt = slide.get("image_prompt") or self.generate_refined_prompt(slide)
            prompt_hash = self.get_prompt_hash(image_prompt)
            filename = f"slide_{i}_{prompt_hash}.png"  # More descriptive
            filepath = os.path.join(save_dir, filename)
            
            if os.path.exists(filepath):
                slide['image_path'] = filepath
                slide['status'] = 'cached'
            else:
                image_url = self.generate_image(image_prompt)
                self.download_image(image_url, filepath)
                
                # Verify the downloaded image
                with Image.open(filepath) as img:
                    if img.size != (1024, 1024):  # Expected size for DALL-E 3
                        img = img.resize((1024, 1024))
                        img.save(filepath)
                
                slide.update({
                    'image_url': image_url,
                    'image_path': filepath,
                    'status': 'generated'
                })
                
        except Exception as e:
            slide.update({
                'image_url': None,
                'image_path': None,
                'error': str(e),
                'status': 'failed'
            })
            
        updated_slides.append(slide)
        
    return updated_slides
