import time
import cv2
from engine import get_inpainter
from pipeline import process_video

print("Starting 1-second (24 frames) test...")
mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)

t0 = time.time()
res = process_video(
    input_path="imdl5-34p1.mp4",
    output_path="test_quick.mp4",
    mask=mask,
    inpaint_fn=get_inpainter("ns"),
    start_time=0.0,
    duration=1.0,
    use_nvenc=True,
    progress_callback=lambda d, t, fps, eta: print(f"Progress: {d}/{t} ({fps:.1f} FPS)", flush=True)
)
print("SUCCESS!", res, f"in {time.time()-t0:.2f}s")
