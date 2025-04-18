import os
import re
import hashlib
from openai import OpenAI
from PIL import Image
import requests
from io import BytesIO
import traceback
import time
from typing import Optional

class DalleImageGenerator:
    """
    Enhanced DALL-E image generator with improved error handling and file management.
    """
    def __init__(self, model: str = "dall-e-3", size: str = "1024x1024", quality: str = "standard"):
        """
        Initializes the DalleImageGenerator with configurable options.
        
        Args:
            model: DALL-E model version ("dall-e-2" or "dall-e-3")
            size: Image dimensions ("1024x1024", "1024x1792", or "1792x1024")
            quality: Image quality ("standard" or "hd")
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment variables.")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.size = size
        self.quality = quality
        self.default_style = "highly detailed digital illustration"
        
        # Configure image storage
        self.image_save_directory = "generated_images"
        os.makedirs(self.image_save_directory, exist_ok=True)

    def generate_filename(self, prompt: str, prefix: str = "img") -> str:
        """
        Generates a readable filename from prompt.
        
        Args:
            prompt: The image generation prompt
            prefix: Optional prefix for filename
            
        Returns:
            str: Generated filename (without extension)
        """
        # Clean the prompt to create a valid filename
        clean_prompt = re.sub(r'[^a-zA-Z0-9]', '_', prompt)[:50]
        timestamp = str(int(time.time()))[-6:]
        return f"{prefix}_{clean_prompt}_{timestamp}"

    def generate_image(self, prompt: str) -> str:
        """
        Generates an image using DALL-E API.
        
        Args:
            prompt: Text prompt for image generation
            
        Returns:
            str: URL of the generated image
            
        Raises:
            RuntimeError: If API call fails
        """
        full_prompt = f"{self.default_style}, {prompt}" if self.default_style else prompt
        
        try:
            response = self.client.images.generate(
                model=self.model,
                prompt=full_prompt,
                size=self.size,
                quality=self.quality,
                n=1
            )
            
            if not response.data or not response.data[0].url:
                raise RuntimeError("API response did not contain image URL")
                
            return response.data[0].url
            
        except Exception as e:
            raise RuntimeError(f"DALL-E API call failed: {str(e)}")

    def download_image(self, url: str, prompt: str) -> str:
        """Alias for download_and_save_image for backward compatibility"""
        return self.download_and_save_image(url, prompt)

    def download_and_save_image(self, url: str, prompt: str) -> str:
        """
        Downloads and saves an image from URL with comprehensive error handling.
        
        Args:
            url: Image URL to download
            prompt: Original prompt for filename generation
            
        Returns:
            str: Path to saved image
            
        Raises:
            RuntimeError: If download or processing fails
        """
        image = None
        try:
            # Download image
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Process image
            image = Image.open(BytesIO(response.content))
            image.load()  # Force loading to catch corrupt files
            
            # Generate filename and path
            filename = f"{self.generate_filename(prompt)}.png"
            save_path = os.path.join(self.image_save_directory, filename)
            
            # Convert to RGB if needed
            if image.mode in ('RGBA', 'LA', 'P'):
                rgb_image = Image.new("RGB", image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[3] if image.mode == 'RGBA' else None)
                image = rgb_image
                
            # Save image
            image.save(save_path, format='PNG', quality=95)
            return save_path
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to download image: {str(e)}")
        except (IOError, SyntaxError, Image.DecompressionBombError) as e:
            raise RuntimeError(f"Invalid image data: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error: {str(e)}")
        finally:
            if image:
                try:
                    image.close()
                except Exception:
                    pass

    def generate_and_save_image(self, prompt: str) -> str:
        """
        End-to-end image generation and saving.
        
        Args:
            prompt: Text prompt for image generation
            
        Returns:
            str: Path to saved image
            
        Raises:
            RuntimeError: If any step fails
        """
        try:
            image_url = self.generate_image(prompt)
            return self.download_and_save_image(image_url, prompt)
        except Exception as e:
            raise RuntimeError(f"Failed to generate and save image: {str(e)}")

    def cleanup_old_images(self, max_age_hours: int = 24) -> None:
        """
        Cleans up images older than specified hours.
        
        Args:
            max_age_hours: Maximum age in hours to keep images
        """
        now = time.time()
        cutoff = now - (max_age_hours * 3600)
        
        for filename in os.listdir(self.image_save_directory):
            filepath = os.path.join(self.image_save_directory, filename)
            try:
                if os.path.isfile(filepath) and os.path.getmtime(filepath) < cutoff:
                    os.remove(filepath)
            except Exception:
                continue
