import time
import cv2
from engine import get_inpainter
from pipeline import process_video

print("Testing 3-second render with LaMa on CUDA...")
mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)

t0 = time.time()
res = process_video(
    input_path="imdl5-34p1.mp4",
    output_path="test_3s_lama.mp4",
    mask=mask,
    inpaint_fn=get_inpainter("lama"),
    start_time=1.0,
    duration=3.0,
    use_nvenc=True,
    progress_callback=lambda d, t, fps, eta: print(f"LaMa: {d}/{t} ({fps:.1f} FPS, ETA: {eta})", flush=True)
)
print("SUCCESS!", res, f"in {time.time()-t0:.2f}s")
