import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

@dataclass
class ModelConfig:
    MODEL_ID: str = "nvidia/Cosmos3-Nano"
    TORCH_DTYPE: str = "bfloat16"  # Store as string for easier config serialization
    DEVICE_MAP: str = "cuda"
    ENABLE_SAFETY_CHECKER: bool = False
    USE_FLASH_ATTENTION: bool = True
    
@dataclass
class GenerationConfig:
    DEFAULT_HEIGHT: int = 720
    DEFAULT_WIDTH: int = 1280
    DEFAULT_FPS: float = 24.0
    DEFAULT_NUM_FRAMES_T2I: int = 1
    DEFAULT_NUM_FRAMES_I2V: int = 189
    MAX_NUM_FRAMES: int = 250
    DEFAULT_GUIDANCE_SCALE: float = 7.0
    
@dataclass
class AppConfig:
    APP_NAME: str = "Cosmos3 Inference Studio"
    VERSION: str = "1.0.0"
    MODELS_DIR: Path = Path("models")
    OUTPUTS_DIR: Path = Path("outputs")
    CACHE_DIR: Path = Path.home() / ".cache" / "cosmos3-studio"
    SUPPORTED_IMAGE_FORMATS: tuple = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    SUPPORTED_VIDEO_FORMATS: tuple = (".mp4", ".avi", ".mov", ".webm")
    MAX_QUEUE_SIZE: int = 5
    
    # Remote inference settings
    DEFAULT_INFERENCE_TARGET: Literal["local", "remote"] = "local"
    DEFAULT_REMOTE_URL: str = "http://localhost:8000"
    
    def __post_init__(self):
        self.MODELS_DIR.mkdir(exist_ok=True)
        self.OUTPUTS_DIR.mkdir(exist_ok=True)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
