import logging
import io
import os
import tempfile
from typing import Optional, Callable, Union, List, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import requests

from app.engine.model_manager import ModelManager
from app.engine.pipeline import GenerationPipeline
from app.config import ModelConfig, GenerationConfig

logger = logging.getLogger(__name__)

class InferenceEngine:
    """
    Main inference engine that coordinates model management and generation.
    Supports both local execution (GPU/CPU) and remote execution (network GPU server).
    Provides a fallback Mock Mode for offline testing/development.
    """
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.generation_pipeline: Optional[GenerationPipeline] = None
        self.config = ModelConfig()
        self.gen_config = GenerationConfig()
        self.inference_target = "local"  # "local" or "remote"
        self.remote_url = "http://localhost:8000"
        self.mock_mode = False
    
    def set_inference_target(self, target: str, remote_url: str):
        """Set the inference compute target and remote server address."""
        self.inference_target = target
        self.remote_url = remote_url.rstrip('/')
        logger.info(f"Inference target set to: {self.inference_target} ({self.remote_url if target == 'remote' else 'local'})")
        
    def test_remote_connection(self) -> Tuple[bool, str]:
        """Test connection to the remote inference server."""
        try:
            response = requests.get(f"{self.remote_url}/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                gpu_msg = "GPU Connected"
                if data.get("gpu_info", {}).get("available"):
                    gpu_msg = f"GPU: {data['gpu_info'].get('name', 'Unknown')}"
                return True, f"Connected. {gpu_msg} ({data.get('status', 'Idle')})"
            return False, f"Server returned status code {response.status_code}"
        except requests.exceptions.RequestException as e:
            return False, f"Connection failed: {str(e)}"

    def initialize(self, progress_callback: Callable = None) -> bool:
        """
        Initialize the inference engine.
        If target is local, load model weights.
        If target is remote, test server connectivity.
        """
        self.mock_mode = False
        
        if self.inference_target == "remote":
            if progress_callback:
                progress_callback(30, "Connecting to remote GPU server...")
            success, msg = self.test_remote_connection()
            if progress_callback:
                progress_callback(100, msg)
            if not success:
                logger.warning(f"Failed to connect to remote server. Remote message: {msg}")
                # Don't fail completely, allow initializing, but mark connection as failed
                raise ConnectionError(f"Could not connect to remote server at {self.remote_url}.\nDetails: {msg}")
            return True
            
        # Local mode initialization
        try:
            if progress_callback:
                progress_callback(10, "Checking ML libraries...")
            
            # Verify if torch and diffusers can be imported
            import torch
            import diffusers
            
            pipeline = self.model_manager.load_model(progress_callback)
            self.generation_pipeline = GenerationPipeline(
                pipeline, 
                self.gen_config
            )
            return True
        except (ImportError, RuntimeError) as e:
            logger.warning(f"Local model initialization failed: {e}. Switching to Mock Mode.")
            self.mock_mode = True
            if progress_callback:
                progress_callback(100, "Local GPU unavailable. Switched to Mock (Demo) Mode.")
            return True
            
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1280,
        height: int = 720,
        steps: int = 50,
        guidance: float = 7.0,
        seed: Optional[int] = None,
        progress_callback: Callable = None,
    ) -> Image.Image:
        """
        Generate image from text prompt (locally or remotely).
        """
        if self.mock_mode:
            return self._generate_mock_image(prompt, width, height, progress_callback)
            
        if self.inference_target == "remote":
            return self._generate_remote_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                guidance=guidance,
                seed=seed,
                progress_callback=progress_callback
            )
            
        # Local Mode
        if not self.generation_pipeline:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        
        if progress_callback:
            progress_callback(0, "Starting image generation...")
        
        image = self.generation_pipeline.text_to_image(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            seed=seed,
        )
        
        if progress_callback:
            progress_callback(100, "Generation complete!")
        
        return image
    
    def generate_video(
        self,
        image: Union[str, Image.Image],
        prompt: str,
        negative_prompt: str = "",
        width: int = 1280,
        height: int = 720,
        num_frames: int = 189,
        fps: float = 24.0,
        steps: int = 50,
        guidance: float = 7.0,
        seed: Optional[int] = None,
        progress_callback: Callable = None,
    ) -> Tuple[List[Image.Image], float]:
        """
        Generate video from image and text prompt (locally or remotely).
        """
        if self.mock_mode:
            # If image is path, load it
            if isinstance(image, str):
                image = Image.open(image)
            return self._generate_mock_video(image, prompt, width, height, num_frames, fps, progress_callback)
            
        if self.inference_target == "remote":
            return self._generate_remote_video(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                fps=fps,
                steps=steps,
                guidance=guidance,
                seed=seed,
                progress_callback=progress_callback
            )
            
        # Local Mode
        if not self.generation_pipeline:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        
        if progress_callback:
            progress_callback(0, "Starting video generation...")
        
        frames, actual_fps = self.generation_pipeline.image_to_video(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            num_inference_steps=steps,
            guidance_scale=guidance,
            seed=seed,
        )
        
        if progress_callback:
            progress_callback(100, "Generation complete!")
        
        return frames, actual_fps
    
    def _generate_remote_image(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        seed: Optional[int],
        progress_callback: Callable = None,
    ) -> Image.Image:
        """Helper to send text-to-image request to remote server."""
        if progress_callback:
            progress_callback(20, "Sending request to remote GPU server...")
            
        data = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance": guidance,
            "seed": seed if seed is not None else -1
        }
        
        try:
            response = requests.post(
                f"{self.remote_url}/generate/t2i",
                data=data,
                timeout=120  # Remote generation can take a bit
            )
            
            if response.status_code != 200:
                error_msg = f"Remote server error: {response.text}"
                try:
                    error_msg = response.json().get("detail", error_msg)
                except:
                    pass
                raise RuntimeError(error_msg)
                
            if progress_callback:
                progress_callback(80, "Receiving image from server...")
                
            image = Image.open(io.BytesIO(response.content))
            
            if progress_callback:
                progress_callback(100, "Generation complete!")
                
            return image
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to communicate with remote server: {e}")

    def _generate_remote_video(
        self,
        image: Union[str, Image.Image],
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_frames: int,
        fps: float,
        steps: int,
        guidance: float,
        seed: Optional[int],
        progress_callback: Callable = None,
    ) -> Tuple[List[Image.Image], float]:
        """Helper to send image-to-video request to remote server."""
        if progress_callback:
            progress_callback(20, "Preparing reference image...")
            
        # Get PIL Image
        if isinstance(image, str):
            image = Image.open(image)
            
        # Serialize PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        data = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "fps": fps,
            "steps": steps,
            "guidance": guidance,
            "seed": seed if seed is not None else -1
        }
        
        files = {
            "image": ("reference.jpg", img_byte_arr, "image/jpeg")
        }
        
        if progress_callback:
            progress_callback(40, "Sending video generation request to remote GPU...")
            
        try:
            response = requests.post(
                f"{self.remote_url}/generate/i2v",
                data=data,
                files=files,
                timeout=300  # Video generation takes longer
            )
            
            if response.status_code != 200:
                error_msg = f"Remote server error: {response.text}"
                try:
                    error_msg = response.json().get("detail", error_msg)
                except:
                    pass
                raise RuntimeError(error_msg)
                
            if progress_callback:
                progress_callback(80, "Decoding video frames...")
                
            # Write video response content to temporary file
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
                temp_video.write(response.content)
                temp_video_path = temp_video.name
                
            # Read frames using imageio
            import imageio
            reader = imageio.get_reader(temp_video_path)
            frames = []
            for frame_arr in reader:
                frames.append(Image.fromarray(frame_arr))
            reader.close()
            
            # Clean up temp file
            try:
                os.unlink(temp_video_path)
            except:
                pass
                
            if progress_callback:
                progress_callback(100, "Video generation complete!")
                
            return frames, fps
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to communicate with remote server: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to decode remote video response: {e}")

    def _generate_mock_image(self, prompt: str, width: int, height: int, progress_callback: Callable = None) -> Image.Image:
        """Generate a beautiful abstract mock image for testing."""
        if progress_callback:
            for i in range(1, 6):
                progress_callback(i * 20, f"Generating mock image... (Step {i}/5)")
                import time; time.sleep(0.1)
                
        # Create an abstract gradient image
        image = Image.new("RGB", (width, height), "#1a1a2e")
        draw = ImageDraw.Draw(image)
        
        # Draw nice gradient/abstract circles
        import random
        # Seed by prompt hash to keep consistent for same prompt
        random.seed(hash(prompt))
        
        for _ in range(8):
            r = random.randint(100, 300)
            x = random.randint(0, width)
            y = random.randint(0, height)
            color = random.choice(["#00d2ff", "#3a7bd5", "#667eea", "#764ba2", "#00ff88"])
            # Draw semi-transparent circle (emulated via blend)
            overlay = Image.new("RGB", (width, height), "#1a1a2e")
            draw_overlay = ImageDraw.Draw(overlay)
            draw_overlay.ellipse([x-r, y-r, x+r, y+r], fill=color)
            image = Image.blend(image, overlay, 0.25)
            
        # Draw text
        draw = ImageDraw.Draw(image)
        text = f"Mock T2I: {prompt}"
        # Make sure text fits on screen
        if len(text) > 60:
            text = text[:57] + "..."
            
        # Attempt to load a default font, or fall back to standard
        try:
            font = ImageFont.load_default()
        except:
            font = None
            
        draw.text((30, height - 50), text, fill="#ffffff", font=font)
        draw.text((30, 30), "COSMOS3 INFERENCE STUDIO (MOCK MODE)", fill="#00d2ff", font=font)
        
        return image
        
    def _generate_mock_video(self, ref_image: Image.Image, prompt: str, width: int, height: int, 
                             num_frames: int, fps: float, progress_callback: Callable = None) -> Tuple[List[Image.Image], float]:
        """Generate a mock animation from a reference image."""
        if progress_callback:
            for i in range(1, 6):
                progress_callback(i * 20, f"Generating mock video... (Step {i}/5)")
                import time; time.sleep(0.15)
                
        # Ensure ref_image matches height and width
        ref_image = ref_image.resize((width, height), Image.Resampling.LANCZOS)
        
        frames = []
        for f in range(num_frames):
            # Evolve image: pan it slightly and apply color shifting/zooming
            # We shift the image horizontally and vertically over time
            shift_x = int(15 * np.sin(2 * np.pi * f / 30))
            shift_y = int(10 * np.cos(2 * np.pi * f / 30))
            
            # Simple transform to simulate motion
            frame = ref_image.transform(
                (width, height),
                Image.Transform.AFFINE,
                (1, 0, shift_x, 0, 1, shift_y),
                resample=Image.Resampling.BILINEAR
            )
            
            # Draw frame index indicator
            draw = ImageDraw.Draw(frame)
            try:
                font = ImageFont.load_default()
            except:
                font = None
                
            draw.text((20, height - 40), f"Frame {f+1}/{num_frames} | Motion: {shift_x},{shift_y}", fill="#00ff88", font=font)
            draw.text((20, 20), f"Prompt: {prompt}", fill="#00d2ff", font=font)
            
            frames.append(frame)
            
        return frames, fps

    def save_output(self, data, filepath: str, fps: float = 24.0):
        """Save generated output to disk."""
        if isinstance(data, Image.Image):
            # Save image
            data.save(filepath, format="JPEG", quality=95)
            # Save prompt/seed metadata if possible
            # FileHandler.save_metadata(params, filepath) is handled at GUI layer
        elif isinstance(data, list):
            # Save video frames
            try:
                from diffusers.utils import export_to_video
                export_to_video(data, filepath, fps=fps, macro_block_size=1)
            except Exception:
                import imageio
                np_frames = [np.array(f) for f in data]
                imageio.mimsave(filepath, np_frames, fps=fps, macro_block_size=1)
    
    def shutdown(self):
        """Clean shutdown of the inference engine."""
        if self.model_manager:
            self.model_manager.unload_model()
        self.generation_pipeline = None
