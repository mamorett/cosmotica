import logging
from typing import Optional

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages the lifecycle of the Cosmos3 model pipeline.
    Handles loading, unloading, and memory management of the model.
    """
    
    def __init__(self, config=None):
        from app.config import ModelConfig
        self.config = config or ModelConfig()
        self.pipeline = None
        self._is_loaded = False
        
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded and self.pipeline is not None
    
    def load_model(self, progress_callback=None):
        """
        Load the Cosmos3 model pipeline with optimized settings.
        
        Args:
            progress_callback: Optional callback for loading progress
            
        Returns:
            Loaded pipeline object
        """
        try:
            # Lazy imports to allow running client without PyTorch on CPU-only machines
            import torch
            from diffusers import Cosmos3OmniPipeline
            
            logger.info(f"Loading model: {self.config.MODEL_ID}")
            
            if progress_callback:
                progress_callback(10, "Initializing pipeline...")
            
            # Map string to actual torch dtype
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32
            }
            torch_dtype = dtype_map.get(self.config.TORCH_DTYPE, torch.bfloat16)
            
            # Load pipeline with optimizations
            self.pipeline = Cosmos3OmniPipeline.from_pretrained(
                self.config.MODEL_ID,
                torch_dtype=torch_dtype,
                device_map=self.config.DEVICE_MAP,
                enable_safety_checker=self.config.ENABLE_SAFETY_CHECKER,
            )
            
            if progress_callback:
                progress_callback(50, "Optimizing pipeline...")
            
            # Apply performance optimizations
            self._optimize_pipeline()
            
            if progress_callback:
                progress_callback(80, "Compiling CUDA operations...")
            
            # Move to CUDA if available and requested
            if torch.cuda.is_available() and self.config.DEVICE_MAP == "cuda":
                self.pipeline = self.pipeline.to("cuda")
            
            if progress_callback:
                progress_callback(100, "Model loaded successfully!")
            
            self._is_loaded = True
            logger.info("Model loaded and optimized successfully")
            
        except ImportError as e:
            logger.error(f"Missing required ML dependencies (torch, diffusers): {e}")
            self._is_loaded = False
            raise RuntimeError(
                f"ML dependencies (PyTorch/Diffusers) are missing: {e}.\n"
                "Please run client in 'Remote GPU' mode or install requirements.txt."
            )
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            self._is_loaded = False
            raise RuntimeError(f"Model loading failed: {str(e)}")
        
        return self.pipeline
    
    def _optimize_pipeline(self):
        """Apply various performance optimizations to the pipeline."""
        if self.pipeline is None:
            return
        
        # Enable memory efficient attention if available
        if hasattr(self.pipeline, 'enable_xformers_memory_efficient_attention'):
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                logger.info("XFormers memory efficient attention enabled")
            except Exception as e:
                logger.warning(f"Could not enable xformers: {e}")
        
        # Enable model CPU offload to save VRAM
        if hasattr(self.pipeline, 'enable_model_cpu_offload'):
            try:
                self.pipeline.enable_model_cpu_offload()
                logger.info("Model CPU offload enabled")
            except Exception as e:
                logger.warning(f"Could not enable CPU offload: {e}")
        
        # Enable VAE slicing for memory efficiency
        if hasattr(self.pipeline, 'enable_vae_slicing'):
            try:
                self.pipeline.enable_vae_slicing()
                logger.info("VAE slicing enabled")
            except Exception as e:
                logger.warning(f"Could not enable VAE slicing: {e}")
        
        # Enable VAE tiling for large images
        if hasattr(self.pipeline, 'enable_vae_tiling'):
            try:
                self.pipeline.enable_vae_tiling()
                logger.info("VAE tiling enabled")
            except Exception as e:
                logger.warning(f"Could not enable VAE tiling: {e}")
    
    def unload_model(self):
        """Unload the model from memory to free resources."""
        if self.pipeline is not None:
            logger.info("Unloading model from memory")
            del self.pipeline
            self.pipeline = None
            self._is_loaded = False
            
            # Clear CUDA cache
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except ImportError:
                pass
            
            logger.info("Model unloaded and memory freed")
    
    def get_memory_usage(self) -> dict:
        """Get current GPU memory usage statistics."""
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    "allocated": torch.cuda.memory_allocated() / 1024**3,
                    "reserved": torch.cuda.memory_reserved() / 1024**3,
                    "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,
                }
        except ImportError:
            pass
        return {}
