# image_creator.py

import os
import re # Keep if needed for refine_prompt etc.
import hashlib # Keep if needed for get_prompt_hash etc.
from openai import OpenAI
from PIL import Image # Pillow library
import requests
from io import BytesIO
import traceback # For detailed error logging

class DalleImageGenerator:
    """
    Handles generating images using OpenAI's DALL-E model and downloading them.
    """
    def __init__(self, model: str = "dall-e-3", size: str = "1024x1024", quality: str = "standard"):
        """
        Initializes the DalleImageGenerator.

        Args:
            model (str): The DALL-E model to use (e.g., "dall-e-3").
            size (str): The desired image size (e.g., "1024x1024", "1024x1792", "1792x1024").
            quality (str): The quality of the image ('standard' or 'hd').
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment variables.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.size = size
        self.quality = quality # Added quality parameter
        self.default_style = "highly detailed digital illustration" # Optional default style prefix

        # Define where generated images will be saved temporarily
        # Ensure this directory exists or is created where needed
        self.image_save_directory = "generated_images"
        os.makedirs(self.image_save_directory, exist_ok=True)

    # --- Placeholder for potentially complex prompt refinement logic ---
    # You would implement these if needed based on your agent's logic
    def refine_prompt(self, original_prompt: str) -> str:
        """
        (Optional) Refines a prompt, potentially making it more suitable for DALL-E.
        This is a basic placeholder - implement complex logic if required.
        """
        # Example: Simple truncation or keyword addition (can be much more sophisticated)
        refined = original_prompt[:1000] # DALL-E 3 has length limits
        print(f"--- DEBUG (refine_prompt): Original: '{original_prompt}', Refined: '{refined}' ---")
        return refined

    def get_prompt_hash(self, prompt: str) -> str:
        """
        (Optional) Generates a hash for a prompt, useful for unique filenames.
        """
        return hashlib.md5(prompt.encode()).hexdigest()[:10] # Short hash

    # --- Core Image Generation Logic ---
    def generate_image(self, prompt: str) -> str:
        """
        Generates an image using the DALL-E API based on the prompt.

        Args:
            prompt (str): The text prompt for image generation.

        Returns:
            str: The URL of the generated image provided by the OpenAI API.

        Raises:
            RuntimeError: If the API call fails or returns an unexpected response.
        """
        # Combine default style (if any) with the user prompt
        full_prompt = f"{self.default_style}, {prompt}" if self.default_style else prompt
        print(f"--- DEBUG (generate_image): Generating with prompt: '{full_prompt}' ---")

        try:
            response = self.client.images.generate(
                model=self.model,
                prompt=full_prompt,
                size=self.size,
                quality=self.quality, # Use quality setting
                n=1,
                # response_format='url' # Default for DALL-E 3 >= 1.0 client
            )
            # Validate the response structure
            if response and response.data and len(response.data) > 0 and response.data[0].url:
                image_url = response.data[0].url
                # Log revised prompt if model revised it (DALL-E 3 often does for safety/clarity)
                if response.data[0].revised_prompt:
                     print(f"--- INFO (generate_image): Model revised prompt to: '{response.data[0].revised_prompt}' ---")
                print(f"--- DEBUG (generate_image): API Success. URL: {image_url} ---")
                return image_url
            else:
                # Log unexpected structure and raise an error
                print(f"--- ERROR (generate_image): Unexpected API response structure: {response} ---")
                raise RuntimeError("API response did not contain expected image URL.")

        except Exception as e:
            print(f"--- ERROR (generate_image): API call failed: {e} ---")
            # Optionally add retry logic with refined prompt here if needed
            # For simplicity, we'll just raise a comprehensive error for now
            raise RuntimeError(f"DALL-E API call failed for prompt '{prompt}'. Error: {str(e)}")

    # --- Image Downloading and Saving Logic ---
    def download_and_save_image(self, url: str, prompt_for_filename: str) -> str:
        """
        Downloads an image from a URL, validates it, converts if needed,
        and saves it locally using a filename derived from the prompt.

        Args:
            url (str): The URL of the image to download.
            prompt_for_filename (str): The original prompt used to generate a unique filename.

        Returns:
            str: The local file path where the image was saved.

        Raises:
            RuntimeError: If downloading, validation, or saving fails.
        """
        image = None  # Initialize image to None for the finally block
        try:
            print(f"--- DEBUG (download_save): Downloading URL: {url} ---")
            response = requests.get(url, timeout=30) # Increased timeout for potentially large images
            response.raise_for_status()  # Check for HTTP errors (4xx, 5xx)
            print(f"--- DEBUG (download_save): Download status code: {response.status_code}, Content-Type: {response.headers.get('Content-Type')} ---")

            # Attempt to open and validate the image using PIL/Pillow
            try:
                # Use BytesIO to treat the downloaded content like a file
                img_stream = BytesIO(response.content)
                print(f"--- DEBUG (download_save): Opening image from BytesIO stream... ---")
                image = Image.open(img_stream)
                print(f"--- DEBUG (download_save): Image opened. Format: {image.format}, Mode: {image.mode}, Size: {image.size} ---")

                # Explicitly load the image data to ensure it's fully read and valid
                # This catches truncated/corrupt image data errors
                print(f"--- DEBUG (download_save): Loading image data (image.load())... ---")
                image.load()
                print(f"--- DEBUG (download_save): Image data loaded successfully. ---")

            except (IOError, SyntaxError, Image.DecompressionBombError) as pil_error:
                print(f"--- ERROR (download_save): PIL Error opening/loading image: {pil_error} ---")
                raise RuntimeError(f"Invalid or corrupt image data received from URL. PIL Error: {str(pil_error)}")
            except Exception as open_err:
                print(f"--- ERROR (download_save): Unexpected Error opening/loading image: {open_err} ---")
                print(traceback.format_exc()) # Log full traceback for unexpected errors
                raise RuntimeError(f"Unexpected error processing image data: {str(open_err)}")

            # Ensure the image object is valid after open/load attempt
            if image is None:
                 raise RuntimeError("Image object is None after attempting to open/load, indicating a processing failure.")

            # --- Image Saving ---
            # Generate a somewhat unique filename based on the prompt hash
            filename_base = self.get_prompt_hash(prompt_for_filename)
            save_filename = f"{filename_base}.png" # Save as PNG
            save_path = os.path.join(self.image_save_directory, save_filename)

            print(f"--- DEBUG (download_save): Preparing to save image to: {save_path} ---")

            # Ensure save directory exists (redundant if created in __init__, but safe)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Handle potential transparency or different modes by converting to RGB
            # PNG supports transparency (RGBA), but converting to RGB is safer for broader compatibility
            # If you NEED transparency, save without conversion or convert to RGBA explicitly.
            save_image = image
            if image.mode in ('RGBA', 'LA', 'P'): # P (Palette) often needs conversion
                print(f"--- DEBUG (download_save): Converting image from mode {image.mode} to RGB for saving. ---")
                # Create a new RGB image and paste the original onto it with white background
                # This handles transparency correctly when converting to RGB.
                save_image = Image.new("RGB", image.size, (255, 255, 255))
                try:
                    # Paste using the original image as a mask if it has alpha
                    save_image.paste(image, mask=image.split()[3] if image.mode == 'RGBA' or image.mode == 'LA' else None)
                except IndexError: # Handle cases like 'LA' mode which might only have L and A channels
                     save_image.paste(image) # Fallback paste without mask

                print(f"--- DEBUG (download_save): Conversion to RGB successful. ---")

            # Save the potentially converted image
            print(f"--- DEBUG (download_save): Attempting to save image... ---")
            save_image.save(save_path, format='PNG', quality=95) # Use high quality for PNG
            print(f"--- DEBUG (download_save): Image successfully saved to {save_path}. ---")

            return save_path # Return the path where the image was saved

        except requests.exceptions.RequestException as req_e:
            print(f"--- ERROR (download_save): RequestException downloading URL '{url}': {req_e} ---")
            raise RuntimeError(f"Failed to download image URL: {str(req_e)}")
        except Exception as e:
            print(f"--- ERROR (download_save): Unexpected Exception during download/processing/saving: {e} ---")
            print(traceback.format_exc()) # Print full traceback
            # Re-raise consistently as RuntimeError
            if not isinstance(e, RuntimeError):
                raise RuntimeError(f"Error during image download/processing/saving: {str(e)}")
            else:
                raise # Re-raise the original RuntimeError (e.g., from PIL errors)

        finally:
            # Ensure the PIL Image object is closed to free resources, regardless of success/failure
            if image:
                try:
                    print(f"--- DEBUG (download_save): Closing image object. ---")
                    image.close()
                except Exception as close_err:
                    # Log error closing, but don't overwrite primary error by raising here
                    print(f"--- WARNING (download_save): Non-critical error closing image object: {close_err} ---")


    # --- Combined Function (Example Usage Pattern) ---
    def generate_and_save_image(self, prompt: str) -> str:
        """
        Generates an image from a prompt, downloads it, and saves it locally.

        Args:
            prompt (str): The text prompt.

        Returns:
            str: The local file path of the saved image.
        """
        print(f"--- INFO (generate_and_save): Starting process for prompt: '{prompt}' ---")
        # Step 1: Generate the image URL using DALL-E API
        image_url = self.generate_image(prompt)

        # Step 2: Download the image from the URL and save it locally
        local_image_path = self.download_and_save_image(image_url, prompt)

        print(f"--- INFO (generate_and_save): Successfully generated and saved image to: {local_image_path} ---")
        return local_image_path

