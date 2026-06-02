import logging
from typing import Optional, Union, Tuple
from PIL import Image

logger = logging.getLogger(__name__)

class GenerationPipeline:
    """
    Orchestrates the generation process for both text-to-image and image-to-video.
    """
    
    def __init__(self, pipeline, config=None):
        from app.config import GenerationConfig
        self.pipeline = pipeline
        self.config = config or GenerationConfig()
    
    def text_to_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        height: int = None,
        width: int = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        seed: Optional[int] = None,
        progress_callback: Optional[callable] = None,
    ) -> Image.Image:
        """
        Generate an image from text prompt.
        """
        import torch
        height = height or self.config.DEFAULT_HEIGHT
        width = width or self.config.DEFAULT_WIDTH
        
        # Set random seed if provided
        generator = None
        if seed is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(seed)
            
        # Hook progress callback if supported
        extra_kwargs = {}
        if progress_callback:
            try:
                import inspect
                sig = inspect.signature(self.pipeline.__call__)
                if "callback_on_step_end" in sig.parameters:
                    def step_end_cb(pipe, step, timestep, callback_kwargs):
                        percent = int(10 + 80 * (step / num_inference_steps))
                        progress_callback(percent, f"Step {step}/{num_inference_steps}")
                        return callback_kwargs
                    extra_kwargs["callback_on_step_end"] = step_end_cb
                elif "callback" in sig.parameters:
                    def legacy_cb(step, timestep, latents):
                        percent = int(10 + 80 * (step / num_inference_steps))
                        progress_callback(percent, f"Step {step}/{num_inference_steps}")
                    extra_kwargs["callback"] = legacy_cb
                    extra_kwargs["callback_steps"] = 1
            except Exception as e:
                logger.warning(f"Could not hook T2I pipeline callback: {e}")
        
        # Run inference
        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=1,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            **extra_kwargs
        )
        
        # Extract first frame as image
        if hasattr(result, 'frames') and len(result.frames) > 0:
            image = result.frames[0]
        elif hasattr(result, 'video') and len(result.video) > 0:
            image = result.video[0]
        elif isinstance(result, tuple) or isinstance(result, list):
            image = result[0]
        else:
            # Check if dict
            try:
                image = result.images[0]
            except AttributeError:
                raise ValueError("No image generated in output structure")
        
        # If it's a numpy array, convert to PIL Image
        if not isinstance(image, Image.Image):
            import numpy as np
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
        
        return image
    
    def image_to_video(
        self,
        image: Union[str, Image.Image],
        prompt: str,
        negative_prompt: Optional[str] = None,
        height: int = None,
        width: int = None,
        num_frames: int = None,
        fps: float = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        seed: Optional[int] = None,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[list, float]:
        """
        Generate a video from an input image and text prompt.
        """
        import torch
        from diffusers.utils import load_image
        
        height = height or self.config.DEFAULT_HEIGHT
        width = width or self.config.DEFAULT_WIDTH
        num_frames = num_frames or self.config.DEFAULT_NUM_FRAMES_I2V
        fps = fps or self.config.DEFAULT_FPS
        
        # Load image if path provided
        if isinstance(image, str):
            image = load_image(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("image must be a file path or PIL Image")
        
        # Validate frame count
        num_frames = min(num_frames, self.config.MAX_NUM_FRAMES)
        
        # Set random seed if provided
        generator = None
        if seed is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(seed)
            
        # Hook progress callback if supported
        extra_kwargs = {}
        if progress_callback:
            try:
                import inspect
                sig = inspect.signature(self.pipeline.__call__)
                if "callback_on_step_end" in sig.parameters:
                    def step_end_cb(pipe, step, timestep, callback_kwargs):
                        percent = int(10 + 80 * (step / num_inference_steps))
                        progress_callback(percent, f"Step {step}/{num_inference_steps}")
                        return callback_kwargs
                    extra_kwargs["callback_on_step_end"] = step_end_cb
                elif "callback" in sig.parameters:
                    def legacy_cb(step, timestep, latents):
                        percent = int(10 + 80 * (step / num_inference_steps))
                        progress_callback(percent, f"Step {step}/{num_inference_steps}")
                    extra_kwargs["callback"] = legacy_cb
                    extra_kwargs["callback_steps"] = 1
            except Exception as e:
                logger.warning(f"Could not hook I2V pipeline callback: {e}")
        
        # Run inference
        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            **extra_kwargs
        )
        
        # Extract video frames
        if hasattr(result, 'video') and len(result.video) > 0:
            frames = result.video
        elif hasattr(result, 'frames') and len(result.frames) > 0:
            frames = result.frames
        elif isinstance(result, dict) and 'video' in result:
            frames = result['video']
        else:
            try:
                frames = result.images
            except AttributeError:
                raise ValueError("No video frames generated in output structure")
        
        # Convert any numpy array frames to PIL Images
        processed_frames = []
        for frame in frames:
            if not isinstance(frame, Image.Image):
                import numpy as np
                if isinstance(frame, np.ndarray):
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8)
                    frame = Image.fromarray(frame)
            processed_frames.append(frame)
            
        return processed_frames, fps
    
    def save_image(self, image: Image.Image, filepath: str, quality: int = 85):
        """Save a generated image to disk."""
        image.save(filepath, format="JPEG", quality=quality)
    
    def save_video(self, frames: list, filepath: str, fps: float = 24.0):
        """Save generated video frames to disk."""
        try:
            from diffusers.utils import export_to_video
            export_to_video(frames, filepath, fps=fps, macro_block_size=1)
        except Exception as e:
            logger.warning(f"Could not use diffusers export_to_video: {e}. Falling back to imageio.")
            import imageio
            import numpy as np
            
            # Convert PIL Images to numpy arrays
            np_frames = []
            for frame in frames:
                if isinstance(frame, Image.Image):
                    np_frames.append(np.array(frame))
                else:
                    np_frames.append(frame)
                    
            imageio.mimsave(filepath, np_frames, fps=fps, macro_block_size=1)
