import cv2
from engine import get_inpainter
from pipeline import process_video

mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)

print("Testing 5-second video render with LaMa and NVENC...")
res = process_video(
    input_path="imdl5-34p1.mp4",
    output_path="test_5s_lama.mp4",
    mask=mask,
    inpaint_fn=get_inpainter("lama"),
    start_time=1.0,
    duration=5.0,
    use_nvenc=True,
    dilation_kernel=9,
    feather_radius=5,
    progress_callback=lambda d, t, fps, eta: print(f"Render: {d}/{t} frames ({fps:.1f} FPS, ETA: {eta})", end="\r")
)

print(f"\nRender finished: {res['frames_processed']} frames in {res['elapsed_sec']:.2f}s ({res['avg_fps']:.1f} FPS)")
