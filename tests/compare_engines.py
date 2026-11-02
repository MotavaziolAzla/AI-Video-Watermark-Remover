import cv2
import numpy as np
from engine import get_inpainter
from pipeline import SmartROICompositor

# Load full mask and sample frame
mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)
frame = cv2.imread(r"temp_inspect\frame1.png") # full frame 1280x720

compositor = SmartROICompositor(
    full_mask=mask,
    context_padding=48,
    dilation_kernel=9,
    feather_radius=5
)

roi = compositor.crop(frame)

# 1. OpenCV NS
inpaint_ns = get_inpainter("ns")
res_ns = inpaint_ns(roi, compositor.dilated_mask_roi)
comp_ns = compositor.composite(frame, res_ns)

# 2. LaMa GPU
inpaint_lama = get_inpainter("lama")
res_lama = inpaint_lama(roi, compositor.dilated_mask_roi)
comp_lama = compositor.composite(frame, res_lama)

# Save comparisons
cv2.imwrite(r"temp_inspect\roi_orig.png", roi)
cv2.imwrite(r"temp_inspect\roi_ns.png", res_ns)
cv2.imwrite(r"temp_inspect\roi_lama.png", res_lama)

cv2.imwrite(r"temp_inspect\full_ns.png", comp_ns)
cv2.imwrite(r"temp_inspect\full_lama.png", comp_lama)

print("Saved comparisons for both NS and LaMa!")
