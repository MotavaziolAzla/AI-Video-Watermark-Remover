import time
import torch
import numpy as np

print(f"CUDA Available: {torch.cuda.is_available()}")

# Create dummy sequence of 8 frames (H=128, W=256)
frames = [np.full((128, 256, 3), 120, dtype=np.uint8) for _ in range(8)]
mask = np.zeros((128, 256), dtype=np.uint8)
mask[40:80, 80:180] = 255 # center box

# 1. Test ProPainter
print("\n--- Testing ProPainter ---")
try:
    from propainter_engine import ProPainterVideoInpainter
    pp = ProPainterVideoInpainter()
    t0 = time.time()
    out_pp = pp.inpaint_sequence(frames, mask)
    print(f"ProPainter SUCCESS! Out length: {len(out_pp)}, shape: {out_pp[0].shape} in {time.time()-t0:.2f}s")
except Exception as e:
    import traceback
    print("ProPainter Error:", e)
    traceback.print_exc()

# 2. Test E2FGVI-HQ
print("\n--- Testing E2FGVI-HQ ---")
try:
    from e2fgvi_engine import E2FGVIVideoInpainter
    e2 = E2FGVIVideoInpainter()
    t0 = time.time()
    out_e2 = e2.inpaint_sequence(frames, mask)
    print(f"E2FGVI-HQ SUCCESS! Out length: {len(out_e2)}, shape: {out_e2[0].shape} in {time.time()-t0:.2f}s")
except Exception as e:
    import traceback
    print("E2FGVI-HQ Error:", e)
    traceback.print_exc()
