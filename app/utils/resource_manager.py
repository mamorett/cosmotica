import torch
from typing import Dict, Any

class ResourceManager:
    """Utility class for monitoring system resources."""
    
    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        """Get GPU information and memory usage."""
        gpu_info = {
            "available": torch.cuda.is_available() if 'torch' in globals() or _try_import_torch() else False,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        if gpu_info["available"]:
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_info.update({
                        "name": gpu.name,
                        "memory_total": gpu.memoryTotal,
                        "memory_used": gpu.memoryUsed,
                        "memory_free": gpu.memoryFree,
                        "memory_util": gpu.memoryUtil * 100,
                        "gpu_util": gpu.gpuUtil * 100,
                        "temperature": gpu.temperature,
                    })
            except Exception:
                pass
        
        return gpu_info
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get system resource information."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=None) # Non-blocking call
            
            return {
                "cpu_percent": cpu_percent,
                "memory_total_gb": memory.total / (1024**3),
                "memory_used_gb": memory.used / (1024**3),
                "memory_percent": memory.percent,
            }
        except ImportError:
            return {
                "cpu_percent": 0.0,
                "memory_total_gb": 0.0,
                "memory_used_gb": 0.0,
                "memory_percent": 0.0,
            }
    
    @staticmethod
    def can_allocate_gpu_memory(required_gb: float) -> bool:
        """Check if enough GPU memory is available."""
        if not torch.cuda.is_available():
            return False
        
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return (gpu.memoryFree / 1024) > required_gb
        except Exception:
            pass
        
        return True

def _try_import_torch() -> bool:
    global torch
    try:
        import torch
        return True
    except ImportError:
        return False
