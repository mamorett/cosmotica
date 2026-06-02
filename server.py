#!/usr/bin/env python3
"""
Cosmos3 Inference Studio - Remote Inference Server
Run this script on the computer hosting the GPU to allow clients to run remote inference.
Usage: python server.py [--host 0.0.0.0] [--port 8000]
"""

import argparse
import io
import os
import sys
import tempfile
import logging
from pathlib import Path
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from fastapi import FastAPI, Form, File, UploadFile, HTTPException
    from fastapi.responses import StreamingResponse
    import uvicorn
except ImportError:
    print("Error: Missing server dependencies. Please install fastapi, uvicorn, and python-multipart.")
    print("Run: pip install fastapi uvicorn python-multipart")
    sys.exit(1)

from app.engine.inference import InferenceEngine
from app.utils.resource_manager import ResourceManager

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("cosmos3-server")

app = FastAPI(
    title="Cosmos3 Remote GPU Server",
    description="Remote inference backend server for Cosmos3 Inference Studio",
    version="1.0.0"
)

# Global engine instance
engine = None

@app.on_event("startup")
def startup_event():
    """Initialize model on server startup."""
    global engine
    logger.info("Initializing Cosmos3 Inference Engine on Server...")
    engine = InferenceEngine()
    engine.set_inference_target("local", "")
    
    def progress_callback(progress, message):
        logger.info(f"Init: {progress}% - {message}")
        
    try:
        success = engine.initialize(progress_callback)
        if success:
            if engine.mock_mode:
                logger.warning("Server initialized in MOCK MODE (No local GPU/CUDA or ML packages found).")
            else:
                logger.info("Server engine initialized successfully on local GPU!")
        else:
            logger.error("Engine failed to initialize.")
    except Exception as e:
        logger.error(f"Error during engine initialization: {e}")

@app.get("/")
@app.get("/status")
def get_status():
    """Return GPU and CPU system status of the server host."""
    gpu_info = ResourceManager.get_gpu_info()
    sys_info = ResourceManager.get_system_info()
    
    # Check if real model is loaded
    model_loaded = engine.model_manager.is_loaded if engine else False
    mock_mode = engine.mock_mode if engine else True
    
    return {
        "status": "Online",
        "model_loaded": model_loaded,
        "mock_mode": mock_mode,
        "gpu_info": gpu_info,
        "sys_info": sys_info
    }

@app.post("/generate/t2i")
def generate_t2i(
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    width: int = Form(1280),
    height: int = Form(720),
    steps: int = Form(50),
    guidance: float = Form(7.0),
    seed: int = Form(-1)
):
    """Generate image from text prompt and return raw JPEG bytes."""
    if not engine:
        raise HTTPException(status_code=503, detail="Inference engine not ready")
        
    actual_seed = None if seed == -1 else seed
    logger.info(f"T2I Request - Prompt: '{prompt}', Size: {width}x{height}, Steps: {steps}, Seed: {actual_seed}")
    
    try:
        # Run generation
        image = engine.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            guidance=guidance,
            seed=actual_seed
        )
        
        # Save to buffer
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)
        
        return StreamingResponse(img_byte_arr, media_type="image/jpeg")
        
    except Exception as e:
        logger.error(f"T2I generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/generate/i2v")
async def generate_i2v(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    width: int = Form(1280),
    height: int = Form(720),
    num_frames: int = Form(189),
    fps: float = Form(24.0),
    steps: int = Form(50),
    guidance: float = Form(7.0),
    seed: int = Form(-1)
):
    """Generate video from reference image and return raw MP4 video bytes."""
    if not engine:
        raise HTTPException(status_code=503, detail="Inference engine not ready")
        
    actual_seed = None if seed == -1 else seed
    logger.info(f"I2V Request - Prompt: '{prompt}', Size: {width}x{height}, Frames: {num_frames}, FPS: {fps}, Steps: {steps}")
    
    # Save uploaded file to PIL Image
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
    except Exception as e:
        logger.error(f"Failed to read input image: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid reference image: {str(e)}")
        
    temp_mp4_path = None
    try:
        # Run video generation
        frames, actual_fps = engine.generate_video(
            image=pil_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            steps=steps,
            guidance=guidance,
            seed=actual_seed
        )
        
        # Save video to a temporary MP4 file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_mp4:
            temp_mp4_path = temp_mp4.name
            
        engine.save_output(frames, temp_mp4_path, actual_fps)
        
        # Read file bytes to return
        with open(temp_mp4_path, 'rb') as f:
            video_bytes = f.read()
            
        return StreamingResponse(io.BytesIO(video_bytes), media_type="video/mp4")
        
    except Exception as e:
        logger.error(f"I2V generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
        
    finally:
        # Always clean up server temp files
        if temp_mp4_path and os.path.exists(temp_mp4_path):
            try:
                os.unlink(temp_mp4_path)
            except Exception as ex:
                logger.warning(f"Could not delete temp file {temp_mp4_path}: {ex}")

def main():
    parser = argparse.ArgumentParser(description="Cosmos3 Remote GPU Server")
    parser.add_argument("--host", default="0.0.0.0", help="Binding address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to run server on (default: 8000)")
    args = parser.parse_args()
    
    logger.info(f"Starting Cosmos3 Remote server on {args.host}:{args.port}...")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
