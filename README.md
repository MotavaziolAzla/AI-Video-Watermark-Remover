# ⚡ GPU-Accelerated Video Watermark & Text Inpainter

A high-performance local video inpainting tool designed to seamlessly remove static text and corner watermarks (such as "Dola AI") from videos with zero rectangular seam borders, hardware-accelerated NVENC video encoding, and audio track preservation.

---

## 🚀 Inpainting Engines

| Engine | Type | Primary Strength | Typical Speed | Best Used For |
|---|---|---|---|---|
| **Hybrid (ProPainter + Flow Tracking)** | Video Flow + SOTA Inpainting | Dual-domain ProPainter structure synthesis + Flow-guided motion compensated stabilizer | ~4–8 FPS (RTX 3060) / 15–30 FPS (24GB VRAM) | **[RECOMMENDED] Ultimate Consistency**: Perfectly coherent textures, zero flicker during tilts/pans, and flawless scene cut handling |
| **ProPainter (ICCV 2023)** | Video Temporal Flow | Dual-domain flow completion + Spatio-temporal transformer | ~4–8 FPS (RTX 3060) / 15–30 FPS (24GB VRAM) | **SOTA Video Quality**: Complex camera motions, moving objects behind watermarks |
| **E2FGVI-HQ (CVPR 2022)** | Video Temporal Flow | Multi-scale feature propagation + Focal transformer | ~7–12 FPS (RTX 3060) / 25–45 FPS (24GB VRAM) | **Fast Video SOTA**: High temporal consistency at double the speed |
| **AI LaMa** | Single-Frame Neural | Fast Fourier Convolutions (Large Mask Inpainting) | ~20–35 FPS | **Lightweight AI**: Ideal for static scenes, quick renders, and high resolution textures |
| **OpenCV NS / Telea** | Classical Math | Partial differential equations (Navier-Stokes / Telea) | ~50–85 FPS | **Fastest Baseline**: CPU-friendly, ultra-fast previews |

---

## 📁 Project Directory Structure

The project is organized cleanly into dedicated functional directories:

```text
Watermark/
├── app.py                      # Interactive Gradio Web UI
├── pipeline.py                 # Core video processing pipeline (Stabilizer, Scene cuts, NVENC)
├── engine.py                   # Unified inpainter factory & wrappers
├── propainter_engine.py        # ProPainter video inpainting engine
├── e2fgvi_engine.py            # E2FGVI-HQ video inpainting engine
├── mask_generator.py           # Auto watermark detector & mask utilities
├── batch_processor.py          # Headless batch CLI processor
├── config.py                   # Central path & GPU configurations
├── download_models.py          # Model weights downloader & integrity checks
├── requirements.txt            # Python dependencies
├── run_app.bat                 # 1-Click UI launcher
├── setup_new_machine.bat       # Automatic 1-click environment installer
├── mask.png                    # Default watermark mask
│
├── checkpoints/                # Offline model weights (e.g., big-lama.pt)
├── models/                     # Deep learning repositories & weights (ProPainter, E2FGVI)
├── input_videos/               # Input video files for processing
├── output_videos/              # Final processed renders & comparisons
│   ├── archive_tests/          # Historical/intermediate test renders
│   └── debug_frames/           # Debug frame crops & diagnostics
├── tests/                      # Verification, unit, and regression tests
├── experiments/                # Exploratory inspection & temporary test scripts
├── dist/                       # Packaged release archives (.rar, .zip)
└── Watermark_Remover_Package/  # Isolated standalone distribution staging folder
```

> [!NOTE]
> `Watermark_Remover_Package/` is kept independent from day-to-day workspace experiments and is used strictly for staging and testing self-contained zero-dependency releases.

---

## 🛠️ Architecture Highlights

1. **Smart Context-Padded ROI (Multiples of 8/16):**
   - Automatically computes the bounding box around non-zero mask pixels plus user-defined context padding (default 96px).
   - Only processes the tiny cropped corner ROI through inpainting models instead of the entire frame, giving massive speed gains.
   - Preserves 100% of non-masked pixels bitwise.

2. **Seamless Feathered Alpha Blending:**
   - Dilation kernel (default 13px) fully covers antialiased text edges and subtle drop shadows.
   - Gaussian feathering (default 7px) produces a smooth alpha mask (`0.0` to `1.0`), guaranteeing zero visible rectangular border or crop seams.

3. **Motion-Adaptive Scene Cut Protection:**
   - Dual-metric scene cut detector (L1 pixel delta + HSV histogram correlation) prevents temporal models or temporal smoothing from blending frames across shot boundaries.

4. **Direct FFmpeg NVENC Streaming:**
   - Streams raw BGR frames directly to FFmpeg with NVIDIA hardware acceleration (`h264_nvenc`).
   - Copies the original audio track (`-c:a copy`) with zero quality loss or re-encoding artifacts.
   - Never dumps thousands of PNG frames to disk.

---

## 🖥️ How to Run the Web UI

Simply double click:
```
run_app.bat
```
Or run from terminal:
```powershell
.\.venv\Scripts\python.exe app.py
```
Open your browser at: **[http://127.0.0.1:7860](http://127.0.0.1:7860)**

---

## 🎛️ UI Tabs Overview

### Persistent Master Settings (Sidebar)
Governs all tabs and batch runs:
- **Watermark Target Mask**: 1-click toggle between **Dola AI** (bottom-right text) and **Gemini** (bottom-right sparkle logo).
- **Inpainting Engine**: Choose between Hybrid (Recommended), ProPainter, E2FGVI-HQ, AI LaMa, and OpenCV.
- **Mask Dilation (px)**: Default 11–13px.
- **Feather Radius (px)**: Default 7px.
- **Context Padding (px)**: Default 64–96px.
- **NVENC HW Encoding**: Hardware acceleration toggle.

### Tab 1: 🎯 Mask Inspector & Single Frame Preview
- **Sample Video Selector**: Load any video (`imdl5-34p1.mp4`).
- **Timestamp Scrubbing**: Scrub to any second to test frames.
- **Auto-Detect Watermark**: Samples 25 frames across the video and extracts clean `mask.png`.
- **Corner Zoom & Visual Overlay**: Red highlight overlay shows the exact dilated mask boundary and yellow contour.

### Tab 2: 🎬 Video Test Render
- Render any segment (e.g., 2–5 seconds) to inspect temporal consistency, optical flow, and audio playback.

### Tab 3: 📁 Batch Folder Automation
- Select input and output directories to process entire video folders automatically with live progress and ETA.

---

## 💻 CLI Usage (Headless Batch Mode)

```powershell
# Run with Hybrid Engine for Gemini Watermark:
.\.venv\Scripts\python.exe batch_processor.py --input input_videos --output output_videos --preset gemini --engine hybrid --nvenc

# Run with Hybrid Engine for Dola AI Watermark:
.\.venv\Scripts\python.exe batch_processor.py --input input_videos --output output_videos --preset dola_ai --engine hybrid --nvenc

# Run with ProPainter (SOTA Video Optical Flow):
.\.venv\Scripts\python.exe batch_processor.py --input input_videos --output output_videos --preset dola_ai --engine propainter --nvenc

# Run with AI LaMa (Lightweight Single-Frame AI):
.\.venv\Scripts\python.exe batch_processor.py --input input_videos --output output_videos --preset dola_ai --engine lama --nvenc
```

---

## 📦 Setting Up on a New Machine (e.g. 24GB VRAM System)

To migrate this project to a new machine:
1. Copy or clone this project folder to the target machine.
2. Double-click `setup_new_machine.bat` (or run it from cmd/powershell).
   - Automatically installs Python virtual environment (`.venv`).
   - Installs PyTorch with CUDA 12.4 and all required dependencies.
   - Automatically downloads and verifies model weights (`ProPainter.pth`, `raft-things.pth`, `recurrent_flow_completion.pth`, `E2FGVI-HQ-CVPR22.pth`, and `big-lama.pt`).
3. Double-click `run_app.bat` to launch!
