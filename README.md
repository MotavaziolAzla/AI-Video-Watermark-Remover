# ⚡ ClearFlow-AI: High-Performance GPU Video Watermark Inpainter

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-2.6%2Bcu124-EE4C2C.svg)](https://pytorch.org/)
[![Gradio UI](https://img.shields.io/badge/Gradio-Web%20App-orange.svg)](https://gradio.app/)
[![Hardware Acceleration](https://img.shields.io/badge/Encoder-NVENC%20h264-brightgreen.svg)](https://developer.nvidia.com/video-codec-sdk)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A state-of-the-art, GPU-accelerated video inpainting suite engineered to seamlessly erase static text, AI watermarks, and corner logos (such as **Google Gemini** and **Dola AI**) from video sequences. Features **zero rectangular seam borders**, **motion-compensated spatio-temporal stabilization**, **direct NVENC streaming**, and **lossless audio preservation**.

---

## 🌟 Highlights

- **🎯 1-Click Multi-Preset Masking**: Instantly switch between built-in watermark profiles (e.g. **Gemini** 4-pointed sparkle star logo or **Dola AI** text) directly from the sidebar.
- **⚡ Hybrid Inpainting Engine**: Fuses **ProPainter's** spatio-temporal transformer structure reconstruction with **RAFT** flow-guided motion-compensated stabilization—eliminating camera pan/tilt stutters and boundary flickers.
- **🚀 Smart Context-Padded ROI Processing**: Extracts only the localized watermark region (multiples of 8/16 with configurable padding) for inpainting, saving up to 90% GPU compute compared to full-frame processing. Non-masked pixels are preserved bitwise.
- **🎨 Gaussian Feathered Alpha Blending**: Dilation kernels (11–13px) and Gaussian feathering (5–7px) generate smooth continuous alpha transitions, guaranteeing **zero visible crop borders or edge seams**.
- **🎬 Direct NVENC Hardware Streaming**: Pipes raw frames directly to hardware-accelerated `h264_nvenc` via FFmpeg with zero disk frame-dumping, muxing the original audio track (`-c:a copy`) with zero quality loss.
- **✂️ Scene Cut Protection**: Dual-metric scene detector (L1 pixel delta + HSV histogram correlation) prevents temporal models from blending across shot cuts.

---

## 🚀 Supported Engines

| Engine | Type | Primary Strength | Typical Speed | Best Used For |
|---|---|---|---|---|
| **Hybrid [RECOMMENDED]** | Video Flow + SOTA Inpainting | Dual-domain ProPainter structure synthesis + Flow-guided motion compensated stabilizer | ~4–8 FPS (Laptop) / 15–30 FPS (24GB VRAM) | **Ultimate Consistency**: Coherent textures, zero flicker during camera motion, pan, and tilt |
| **ProPainter (ICCV 2023)** | Video Temporal Flow | Dual-domain flow completion + Spatio-temporal transformer | ~4–8 FPS | **SOTA Video Quality**: Complex camera motions, moving objects behind watermarks |
| **E2FGVI-HQ (CVPR 2022)** | Video Temporal Flow | Multi-scale feature propagation + Focal transformer | ~7–12 FPS | **Fast Video SOTA**: High temporal consistency at double the speed |
| **AI LaMa** | Single-Frame Neural | Fast Fourier Convolutions (Large Mask Inpainting) | ~20–35 FPS | **Lightweight AI**: Ideal for static scenes, quick renders, and high resolution textures |
| **OpenCV NS / Telea** | Classical Math | Partial differential equations (Navier-Stokes / Telea) | ~50–85 FPS | **Fastest Baseline**: CPU-friendly, ultra-fast previews |

---

## 📁 Repository Layout

```text
ClearFlow-AI/
├── app.py                      # Interactive Gradio Web UI
├── pipeline.py                 # Core video processing pipeline (Stabilizer, Scene Cuts, NVENC)
├── engine.py                   # Unified inpainter factory & wrappers
├── propainter_engine.py        # ProPainter video inpainting engine
├── e2fgvi_engine.py            # E2FGVI-HQ video inpainting engine
├── mask_generator.py           # Auto watermark detector & mask utilities
├── batch_processor.py          # Headless batch CLI processor
├── config.py                   # Central path & GPU configurations
├── download_models.py          # Automated offline model weights downloader
├── requirements.txt            # Python dependencies
├── run_app.bat                 # 1-Click UI launcher
├── setup_new_machine.bat       # Automatic 1-click environment installer
├── mask.png                    # Default watermark mask
│
├── masks/                      # Built-in watermark mask presets
│   ├── dola_ai.png             # Preset for Dola AI text watermark
│   └── gemini.png              # Preset for Google Gemini sparkle logo
├── checkpoints/                # Model weights directory (fetched via script)
├── models/                     # Deep learning architecture modules
├── input_videos/               # Input video folder
├── output_videos/              # Processed output folder
└── tests/                      # Unit, benchmark, and regression test suite
```

---

## 💻 Quick Start & Installation

### Option A: 1-Click Setup (Windows)
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/ClearFlow-AI.git
   cd ClearFlow-AI
   ```
2. Double-click:
   ```bat
   setup_new_machine.bat
   ```
   *(This automatically creates `.venv`, installs PyTorch CUDA 12.4, dependencies, and downloads model weights).*
3. Double-click:
   ```bat
   run_app.bat
   ```
   Your default web browser will automatically open at **`http://127.0.0.1:7860`**.

---

### Option B: Manual Setup
```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download AI model weights
python download_models.py

# 4. Launch Web UI
python app.py
```

---

## 🎛️ Web UI Walkthrough

Open **`http://127.0.0.1:7860`** in your browser:

### 1. Persistent Sidebar Controls (Governs All Tabs)
- **Target Mask**: 1-click toggle between **Gemini** (sparkle logo) and **Dola AI** (text).
- **Inpainting Engine**: Select **Hybrid (RECOMMENDED)**, ProPainter, E2FGVI-HQ, AI LaMa, or OpenCV.
- **Mask Dilation (px)**: Default 11–13px (ensures complete coverage of drop shadows and antialiased fringes).
- **Feather Radius (px)**: Default 7px (guarantees zero visible rectangular seams).
- **Context Padding (px)**: Default 64–96px (provides surrounding geometric context for structural synthesis).
- **NVENC HW Encoding**: Hardware acceleration toggle.

### 2. Tab 1: 🎯 Mask Inspector & Single Frame Preview
- Scrub to any timestamp in seconds.
- Zoom in on the watermark corner to inspect the original, red mask overlay contour, and clean inpainted preview before full rendering.

### 3. Tab 2: 🎬 Video Test Render
- Render a 2–5 second preview slice with synchronized audio to check temporal fluidity at 24/30 FPS.

### 4. Tab 3: 📁 Batch Folder Automation
- Select your `input_videos` and `output_videos` folders.
- Choose your target watermark preset.
- Click **Start Batch Processing** to clean an entire folder with real-time ETA and FPS metrics.

---

## ⚡ CLI Usage (Headless Batch Mode)

```powershell
# Clean videos with Google Gemini watermark preset:
python batch_processor.py --input input_videos --output output_videos --preset gemini --engine hybrid --nvenc

# Clean videos with Dola AI watermark preset:
python batch_processor.py --input input_videos --output output_videos --preset dola_ai --engine hybrid --nvenc

# High-resolution 24GB VRAM workstation (expanded 64-frame temporal chunk window):
python batch_processor.py --input input_videos --output output_videos --preset gemini --engine hybrid --chunk-size 64 --nvenc

# Fast preview mode using AI LaMa:
python batch_processor.py --input input_videos --output output_videos --preset gemini --engine lama --nvenc
```

---

## 📜 License & Acknowledgments

This project is released under the [MIT License](LICENSE).

Special thanks to the open-source research and implementations powering our engines:
- **ProPainter**: [sczhou/ProPainter](https://github.com/sczhou/ProPainter) (ICCV 2023)
- **E2FGVI-HQ**: [MCG-NKU/E2FGVI](https://github.com/MCG-NKU/E2FGVI) (CVPR 2022)
- **Large Mask Inpainting (LaMa)**: [advimman/lama](https://github.com/advimman/lama) & [simple-lama-inpainting](https://github.com/enesmsahin/simple-lama-inpainting)
- **RAFT**: [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT)
