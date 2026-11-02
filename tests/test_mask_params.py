import cv2
import numpy as np
from engine import get_inpainter
from pipeline import SmartROICompositor

# Load full mask and frames
mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)
inpaint_lama = get_inpainter("lama")

# Test cases:
# 1. Lamp post frame (f350)
cmd_f350 = [r"C:\Program Files\LosslessCut.3.50.0_YasDL.com\resources\ffmpeg.exe", "-ss", "14.583", "-i", "imdl5-34p1.mp4", "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
import subprocess
p = subprocess.Popen(cmd_f350, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
raw = p.stdout.read(1280*720*3)
p.stdout.close(); p.wait()
f350 = np.frombuffer(raw, dtype=np.uint8).reshape((720, 1280, 3))

# Let's test combinations:
# A: Baseline (dilation=9, feather=5)
# B: Moderate expansion (dilation=13, feather=7)
# C: Generous expansion (dilation=17, feather=9)
# D: Morphological Closing (solid word clusters) + feather=7

configs = [
    ("k9_f5", 9, 5),
    ("k13_f7", 13, 7),
    ("k17_f9", 17, 9),
]

for name, dil, feat in configs:
    comp = SmartROICompositor(full_mask=mask, context_padding=96, dilation_kernel=dil, feather_radius=feat)
    roi = comp.crop(f350)
    inp = inpaint_lama(roi, comp.dilated_mask_roi)
    out = comp.composite(f350, inp)
    
    # Crop zoomed area around the lamp
    h, w = f350.shape[:2]
    crop = out[h-200:h, w-260:w-40]
    cv2.imwrite(f"temp_inspect/lamp_post/test_{name}.png", crop)
    print(f"Tested {name}")

print("All tests completed!")
