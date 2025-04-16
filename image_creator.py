# image_creator.py

import os
import re
import hashlib
from openai import OpenAI
from PIL import Image # Pillow library
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

    # ... (generate_refined_prompt, refine_prompt, get_prompt_hash are less relevant to the download error) ...

    def generate_image(self, prompt: str) -> str:
        # This function calls the DALL-E API.
        # Based on logs, it SEEMS to be successfully returning a URL.
        styled_prompt = f"{self.default_style}, {prompt}"
        try:
            response = self.client.images.generate(
                model=self.model,
                prompt=styled_prompt,
                size=self.size,
                n=1
                # Consider adding response_format='url' or 'b64_json' explicitly if needed,
                # though 'url' is default for DALL-E 3 with the python client >= 1.0
            )
            # Add check for response structure and data
            if response and response.data and len(response.data) > 0 and response.data[0].url:
                 print(f"--- DEBUG (generate_image): API Success. URL: {response.data[0].url} ---")
                 return response.data[0].url
            else:
                 # Log the unexpected response structure
                 print(f"--- DEBUG (generate_image): Unexpected API response structure: {response} ---")
                 raise RuntimeError("API response did not contain expected image URL.")

        except Exception as e: # Catch initial API call error
            print(f"--- DEBUG (generate_image): Initial API call failed: {e} ---")
            # Attempt retry with refined prompt (refine_prompt has potential issues like over-truncation)
            try:
                refined = self.refine_prompt(prompt)
                print(f"--- DEBUG (generate_image): Retrying with refined prompt: {refined} ---")
                styled_prompt = f"{self.default_style}, {refined}"
                response = self.client.images.generate(
                    model=self.model,
                    prompt=styled_prompt,
                    size=self.size,
                    n=1
                )
                # Repeat check for response structure
                if response and response.data and len(response.data) > 0 and response.data[0].url:
                    print(f"--- DEBUG (generate_image): API Retry Success. URL: {response.data[0].url} ---")
                    return response.data[0].url
                else:
                    print(f"--- DEBUG (generate_image): Unexpected API response structure on retry: {response} ---")
                    raise RuntimeError("API response on retry did not contain expected image URL.")

            except Exception as retry_e: # Catch retry error
                print(f"--- DEBUG (generate_image): Retry API call failed: {retry_e} ---")
                # Raise a more informative error
                raise RuntimeError(f"DALL-E API call failed on both original and refined prompt attempts. Last error: {str(retry_e)}")


    def download_image(self, url: str, save_path: str) -> None:
        # <<< This is where the traceback indicates the error occurs >>>
        image = None # Initialize image to None for the finally block
        try:
            print(f"--- DEBUG (download_image): Downloading URL: {url} ---")
            response = requests.get(url, timeout=20) # Increased timeout
            response.raise_for_status() # Check for HTTP errors (4xx, 5xx)
            print(f"--- DEBUG (download_image): Download status code: {response.status_code} ---")

            # Validate it's actually an image using PIL/Pillow
            try:
                 # Use BytesIO to treat the downloaded content like a file
                 img_stream = BytesIO(response.content)
                 print(f"--- DEBUG (download_image): Opening image from BytesIO stream... ---")
                 image = Image.open(img_stream)
                 print(f"--- DEBUG (download_image): Image opened. Format: {image.format}, Mode: {image.mode}, Size: {image.size} ---")

                 # *** THE LIKELY CULPRIT WAS HERE ***
                 # image.verify()
                 # According to PIL docs, verify() can leave the file pointer in an undefined state.
                 # It's often unnecessary if Image.open() succeeds without error.
                 # Let's remove it and rely on Image.open() succeeding as the primary validation.

                 # Explicitly load the image data while the stream is presumably valid
                 # This reads the pixel data.
                 print(f"--- DEBUG (download_image): Loading image data (image.load())... ---")
                 image.load()
                 print(f"--- DEBUG (download_image): Image data loaded. ---")

            except (IOError, SyntaxError, Image.DecompressionBombError) as pil_error:
                # Catch specific PIL errors during open/load
                print(f"--- DEBUG (download_image): PIL Error opening/loading image: {pil_error} ---")
                raise RuntimeError(f"Invalid or corrupt image data received: {str(pil_error)}")
            except Exception as open_err:
                 print(f"--- DEBUG (download_image): Unexpected Error opening/loading image: {open_err} ---")
                 raise RuntimeError(f"Failed to process image data: {str(open_err)}")

            # Check if image object is valid after trying to open and load
            if image is None:
                 raise RuntimeError("Image object is None after attempting to open/load.")

            # Convert to RGB if needed (BEFORE saving)
            # Handle common modes including Palette ('P')
            if image.mode in ('RGBA', 'LA', 'P'):
                print(f"--- DEBUG (download_image): Converting image from mode {image.mode} to RGB ---")
                # Use a safer conversion method, especially for palette images
                image = image.convert('RGB')
                print(f"--- DEBUG (download_image): Conversion to RGB successful. ---")

            # Ensure save directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Save the image
            print(f"--- DEBUG (download_image): Attempting to save image to {save_path} ---")
            image.save(save_path, format='PNG', quality=95)
            print(f"--- DEBUG (download_image): Image save command executed. ---")

        except requests.exceptions.RequestException as req_e:
            print(f"--- DEBUG (download_image): RequestException downloading URL: {req_e} ---")
            raise RuntimeError(f"Failed to download image URL: {str(req_e)}")
        # Catch the RuntimeError or other Exceptions raised above, or new ones
        except Exception as e:
            print(f"--- DEBUG (download_image): Exception during download/processing: {e} ---")
            import traceback
            print(traceback.format_exc()) # Print traceback for unexpected errors here
            # Re-raise but ensure it's a RuntimeError for consistency if needed
            if not isinstance(e, RuntimeError):
                 raise RuntimeError(f"Error during image download/processing: {str(e)}")
            else:
                 raise # Re-raise the original RuntimeError

        finally:
            # Ensure the PIL Image object is closed to free resources
            if image:
                try:
                    print(f"--- DEBUG (download_image): Closing image object. ---")
                    image.close()
                except Exception as close_err:
                    print(f"--- WARNING (download_image): Error closing image object: {close_err} ---") # Log but don't fail

    # ... (generate_images_for_slides - relies on the above methods) ...
