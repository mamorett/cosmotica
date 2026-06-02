# 🎨 Cosmos3 Inference Studio

A production-ready, beautiful PyQt6 desktop client and FastAPI remote inference server for NVIDIA's Cosmos3 models. Designed to work on local CUDA-enabled workstations or distribute inference tasks to a GPU-hosting computer over your local network.

---

## ✨ Features

*   **Dual Generation Modes**:
    *   **Text-to-Image (T2I)**: Generate high-resolution visual assets from textual descriptions.
    *   **Image-to-Video (I2V)**: Bring still frames to life using motion-guided prompt parameters.
*   **Network-Distributed Compute**:
    *   **Local GPU**: Direct integration with local PyTorch/CUDA.
    *   **Remote GPU**: Run the GUI client on a CPU-only laptop and stream high-performance inference from a GPU workstation on your network.
*   **Aesthetic Dark Theme**: Premium styling featuring vibrant gradient buttons, modern typography, glassmorphism accents, and responsive hover effects.
*   **Robust Media Playback**:
    *   Inline preview screen supporting video looping at custom frame rates.
    *   Frame-by-frame seeking via interactive sliders.
*   **VRAM Optimizations**: Integrated model CPU offloading, VAE slicing/tiling, and memory-efficient attention (XFormers) configurations.
*   **Local Mock Mode**: Automatically falls back to a mock demo mode if CUDA is missing, allowing offline UI testing and grading on any device.

---

## 📁 Architecture Overview

```
cosmos3-studio/
├── app/
│   ├── main.py                 # Central bootstrapper
│   ├── config.py               # Application configurations
│   ├── engine/
│   │   ├── model_manager.py    # Local weight loading & VRAM configurations
│   │   ├── pipeline.py         # Generation orchestration
│   │   └── inference.py        # Router routing local/remote/mock workloads
│   ├── gui/
│   │   ├── main_window.py      # Core GUI coordinator
│   │   ├── components/
│   │   │   ├── compute_selector.py     # Remote target and URL connections
│   │   │   ├── mode_selector.py        # Mode toggles (T2I vs I2V)
│   │   │   ├── prompt_input.py         # Positive & Negative prompt editors
│   │   │   ├── image_preview.py        # Drag & Drop reference images
│   │   │   ├── media_preview.py        # Video player & timeline controls
│   │   │   └── generation_controls.py  # Resolution, steps, seed parameters
│   │   ├── styles/
│   │   │   └── theme.py                # Gradient UI dark-mode stylesheet
│   │   └── workers/
│   │       └── generation_worker.py    # QThread background workers
│   └── utils/
│       ├── file_handler.py     # Content loading, writing, & JSON metadata
│       └── resource_manager.py # Host CPU/GPU usage metrics
├── server.py                   # FastAPI GPU Server backend script
├── run.py                      # Client launcher with soft ML checks
├── requirements.txt            # Dependency definitions
└── setup.py                    # Distribution setup script
```

---

## 🚀 Setup & Installation

### Prerequities
*   Python 3.10 or higher.
*   (Optional but Recommended) An NVIDIA GPU with CUDA 11.8+ drivers configured on the host computer.

### Quick Start Installation

1.  Clone the repository and enter the workspace directory:
    ```bash
    cd cosmotica
    ```

2.  Initialize an isolated virtual environment and install the required dependencies:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

---

## 🌐 How to Run Remote GPU Inference

### Step 1: Run the Server on the GPU Workstation
On the computer equipped with the graphics card, run the FastAPI server backend:

```bash
# Bind to all interfaces (0.0.0.0) so clients on the network can connect
./.venv/bin/python server.py --host 0.0.0.0 --port 8000
```
*The server will initialize the local GPU. If no PyTorch/CUDA environment is available, it gracefully boots into Mock Mode for API integration checks.*

#### API Endpoints exposed:
*   `GET /status`: Retrieves real-time GPU/CPU resource usage and model loading state.
*   `POST /generate/t2i`: Receives text-to-image parameters and streams back the generated JPEG image.
*   `POST /generate/i2v`: Receives a reference image file upload alongside prompting parameters and streams back the resulting MP4 video.

---

### Step 2: Run the GUI Client
On your client machine (e.g., your laptop), launch the application:

```bash
./.venv/bin/python run.py
```
*Note: The launcher does not crash if PyTorch or CUDA libraries are missing locally, allowing you to run client-only installations seamlessly.*

#### Connecting to the Server:
1.  In the **Compute Target Settings** panel at the top-left, switch the Location dropdown to **Remote GPU (Network Server)**.
2.  Enter the URL of the GPU workstation (e.g. `http://192.168.1.100:8000`).
3.  Click **Test Connection**. A successful test will display the server state and identify the GPU model hosted on the workstation.

---

## 🎨 GUI Interface Guide

1.  **Compute Settings**: Choose Local vs Remote compute, manage URLs, and test latency or connectivity.
2.  **Mode Toggle**: Toggle between T2I (Text-to-Image) and I2V (Image-to-Video).
3.  **Prompt Inputs**: Separate fields for positive prompts and negative prompts (video mode pre-populates specialized Cosmos3 negation keyword lists).
4.  **Drag-and-Drop Dropzone**: Drag any PNG, JPG, or WebP reference image into the dropzone to use as seed input for Video generation.
5.  **Parameter Control Grid**: Adjust Resolution, Guidance (CFG), Steps, Seed, and frame limits.
6.  **Timeline Control Player**: Live video player showing output frames with full play/pause toggle and seek bar.
7.  **Single-Click Exports**: Click the download button to export images (JPEG) or videos (MP4) alongside JSON parameters containing timestamped generation metadata.

---

## 🔧 Model Configurations & Optimizations

If executing in **Local GPU** mode or running the server, optimizations can be configured in [app/config.py](file:///gorgon/ia/cosmotica/app/config.py):

| Config Option | Default | Purpose |
| :--- | :--- | :--- |
| `MODEL_ID` | `nvidia/Cosmos3-Nano` | Cosmos3 model weight repository path on Hugging Face |
| `TORCH_DTYPE` | `bfloat16` | Model weights representation (`bfloat16`, `float16`, `float32`) |
| `DEVICE_MAP` | `cuda` | Hardware accelerator targeting (`cuda` or `cpu`) |
| `USE_FLASH_ATTENTION` | `True` | Leverages memory-efficient attention |

*Model optimizations like model CPU offloading, VAE slicing, and VAE tiling are applied automatically by the [model_manager.py](file:///gorgon/ia/cosmotica/app/engine/model_manager.py) class during model loading to ensure optimal execution on lower-spec consumer hardware.*

---

## 📝 License
This project is licensed under the MIT License - see the `LICENSE` file for details.