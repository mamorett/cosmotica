from PyQt6.QtCore import QThread, pyqtSignal
from PIL import Image
from typing import Optional

class GenerationWorker(QThread):
    """
    Worker thread for running generation without blocking the GUI.
    """
    
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object, str, float)  # data, type, fps
    error = pyqtSignal(str)
    
    def __init__(self, engine, params: dict, mode: str, 
                 prompts: dict, image: Optional[Image.Image] = None):
        super().__init__()
        self.engine = engine
        self.params = params
        self.mode = mode
        self.prompts = prompts
        self.image = image
    
    def run(self):
        """Execute the generation in background thread."""
        try:
            if self.mode == "t2i":
                self._run_text_to_image()
            elif self.mode == "i2v":
                self._run_image_to_video()
        except Exception as e:
            self.error.emit(str(e))
    
    def _run_text_to_image(self):
        """Run text-to-image generation."""
        self.progress.emit(10, "Generating image...")
        
        image = self.engine.generate_image(
            prompt=self.prompts["positive"],
            negative_prompt=self.prompts.get("negative", ""),
            width=self.params["width"],
            height=self.params["height"],
            steps=self.params["steps"],
            guidance=self.params["guidance"],
            seed=self.params.get("seed"),
            progress_callback=self._progress_handler
        )
        
        self.finished.emit(image, "image", 0.0)
    
    def _run_image_to_video(self):
        """Run image-to-video generation."""
        if self.image is None:
            self.error.emit("No reference image provided for image-to-video")
            return
        
        self.progress.emit(10, "Generating video...")
        
        frames, fps = self.engine.generate_video(
            image=self.image,
            prompt=self.prompts["positive"],
            negative_prompt=self.prompts.get("negative", ""),
            width=self.params["width"],
            height=self.params["height"],
            num_frames=self.params["num_frames"],
            fps=self.params["fps"],
            steps=self.params["steps"],
            guidance=self.params["guidance"],
            seed=self.params.get("seed"),
            progress_callback=self._progress_handler
        )
        
        self.finished.emit(frames, "video", fps)

    def _progress_handler(self, value: int, message: str):
        """Relay progress events to the GUI."""
        self.progress.emit(value, message)
