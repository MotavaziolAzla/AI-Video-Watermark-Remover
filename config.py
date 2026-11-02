import os
import shutil
import subprocess
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE_DIR / "input_videos"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output_videos"
DEFAULT_INPUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Masks & Presets
MASKS_DIR = BASE_DIR / "masks"
MASKS_DIR.mkdir(parents=True, exist_ok=True)

MASK_PRESETS = {
    "Dola AI": MASKS_DIR / "dola_ai.png",
    "Gemini": MASKS_DIR / "gemini.png"
}

# Auto-migrate legacy mask.png to masks/dola_ai.png if needed
legacy_mask = BASE_DIR / "mask.png"
if legacy_mask.exists() and not (MASKS_DIR / "dola_ai.png").exists():
    try:
        shutil.copy2(legacy_mask, MASKS_DIR / "dola_ai.png")
    except Exception:
        pass

DEFAULT_MASK_NAME = "Dola AI"
DEFAULT_MASK_PATH = MASKS_DIR / "dola_ai.png" if (MASKS_DIR / "dola_ai.png").exists() else legacy_mask

def get_mask_path(preset_name: str = "Dola AI") -> Path:
    """Return path to preset mask ('Dola AI', 'Gemini'), falling back to DEFAULT_MASK_PATH."""
    key = "Dola AI"
    norm = str(preset_name).lower().replace("_", " ").replace("-", " ").strip()
    if "gemini" in norm:
        key = "Gemini"
    elif "dola" in norm:
        key = "Dola AI"
    
    p = MASK_PRESETS.get(key)
    if p and p.exists():
        return p
    if DEFAULT_MASK_PATH.exists():
        return DEFAULT_MASK_PATH
    return p

# Locate FFmpeg
def find_ffmpeg_executable():
    # 0. Bundled inside project folder (highest priority for portable distribution)
    local_p = BASE_DIR / "bin" / "ffmpeg.exe"
    if local_p.exists():
        return str(local_p)
    # 1. System PATH
    found = shutil.which("ffmpeg")
    if found:
        return found
    # 2. Known local install path
    known_paths = [
        r"C:\Program Files\LosslessCut.3.50.0_YasDL.com\resources\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
    ]
    for p in known_paths:
        if os.path.exists(p):
            return p
    return "ffmpeg"

def find_ffprobe_executable():
    # 0. Bundled inside project folder
    local_p = BASE_DIR / "bin" / "ffprobe.exe"
    if local_p.exists():
        return str(local_p)
    # 1. System PATH
    found = shutil.which("ffprobe")
    if found:
        return found
    known_paths = [
        r"C:\Program Files\LosslessCut.3.50.0_YasDL.com\resources\ffprobe.exe",
        r"C:\ffmpeg\bin\ffprobe.exe",
    ]
    for p in known_paths:
        if os.path.exists(p):
            return p
    return "ffprobe"

FFMPEG_PATH = find_ffmpeg_executable()
FFPROBE_PATH = find_ffprobe_executable()

# Check NVENC encoder availability
def check_nvenc_available():
    try:
        cmd = [FFMPEG_PATH, "-hide_banner", "-encoders"]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
        return "h264_nvenc" in result.stdout
    except Exception:
        return False

HAS_NVENC = check_nvenc_available()
