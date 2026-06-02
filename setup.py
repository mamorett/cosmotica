from setuptools import setup, find_packages

setup(
    name="cosmos3-studio",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "diffusers>=0.31.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "safetensors>=0.4.0",
        "pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "PyQt6>=6.5.0",
        "numpy>=1.24.0",
        "imageio>=2.31.0",
        "imageio-ffmpeg>=0.4.9",
        "qdarkstyle>=3.2.0",
        "qt-material>=2.14",
        "requests>=2.31.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "python-multipart>=0.0.6",
        "psutil>=5.9.0",
        "gputil>=1.4.0",
    ],
    entry_points={
        "console_scripts": [
            "cosmos3-studio=app.main:main",
        ],
    },
)
