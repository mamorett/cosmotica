import os
from pathlib import Path
from typing import Union
from PIL import Image
import json
from datetime import datetime

class FileHandler:
    """Utility class for file operations."""
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]):
        """Ensure a directory exists, creating it if necessary."""
        Path(path).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def get_timestamped_filename(base_name: str, extension: str) -> str:
        """Generate a timestamped filename."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}.{extension}"
    
    @staticmethod
    def load_image(filepath: Union[str, Path]) -> Image.Image:
        """Load an image with error handling."""
        try:
            image = Image.open(filepath)
            # Convert to RGB if necessary
            if image.mode in ('RGBA', 'P'):
                image = image.convert('RGB')
            return image
        except Exception as e:
            raise IOError(f"Failed to load image: {str(e)}")
    
    @staticmethod
    def save_image(image: Image.Image, filepath: Union[str, Path], 
                   quality: int = 95) -> bool:
        """Save an image with error handling."""
        try:
            FileHandler.ensure_directory(Path(filepath).parent)
            image.save(filepath, quality=quality)
            return True
        except Exception as e:
            raise IOError(f"Failed to save image: {str(e)}")
    
    @staticmethod
    def save_metadata(params: dict, filepath: Union[str, Path]):
        """Save generation metadata alongside output."""
        metadata_path = Path(filepath).with_suffix('.json')
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "parameters": params
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
