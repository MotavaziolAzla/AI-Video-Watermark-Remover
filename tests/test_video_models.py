import time
import cv2
import numpy as np
from pathlib import Path

from config import BASE_DIR
from pipeline import process_video
from engine import get_inpainter

mask = cv2.imread(str(BASE_DIR / "mask.png"), cv2.IMREAD_GRAYSCALE)
video_path = str(BASE_DIR / "imdl5-34p1.mp4")

# Test 1: E2FGVI-HQ on 2-second clip
print("\n=== Testing E2FGVI-HQ on 2-Second Video Clip ===")
e2_out = str(BASE_DIR / "test_e2fgvi_clip.mp4")
e2_inpainter = get_inpainter("e2fgvi")
t0 = time.time()
res_e2 = process_video(
    input_path=video_path,
    output_path=e2_out,
    mask=mask,
    inpaint_fn=e2_inpainter,
    start_time=0.0,
    duration=2.0,
    context_padding=96,
    dilation_kernel=13,
    feather_radius=7
)
print(f"E2FGVI finished: {res_e2['frames_processed']} frames in {res_e2['elapsed_sec']:.2f}s ({res_e2['avg_fps']:.1f} FPS) -> {e2_out}")

# Test 2: ProPainter on 2-second clip
print("\n=== Testing ProPainter on 2-Second Video Clip ===")
pp_out = str(BASE_DIR / "test_propainter_clip.mp4")
pp_inpainter = get_inpainter("propainter")
t0 = time.time()
res_pp = process_video(
    input_path=video_path,
    output_path=pp_out,
    mask=mask,
    inpaint_fn=pp_inpainter,
    start_time=0.0,
    duration=2.0,
    context_padding=96,
    dilation_kernel=13,
    feather_radius=7
)
print(f"ProPainter finished: {res_pp['frames_processed']} frames in {res_pp['elapsed_sec']:.2f}s ({res_pp['avg_fps']:.1f} FPS) -> {pp_out}")
print("\n=== ALL VIDEO INPAINTING ENGINE TESTS PASSED! ===")
