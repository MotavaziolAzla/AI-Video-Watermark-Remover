import cv2
import numpy as np
from engine import get_inpainter
from pipeline import SmartROICompositor

mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)
frame = cv2.imread(r"temp_inspect\br_6s.png")

# Remember br_6s was cropped 350x180 from full frame (930 to 1280, 540 to 720)
# Let's extract mask patch for this 350x180 region
mask_patch = mask[540:720, 930:1280]

compositor = SmartROICompositor(
    full_mask=mask_patch,
    context_padding=48,
    dilation_kernel=9,
    feather_radius=5
)

roi = compositor.crop(frame)
inpaint_lama = get_inpainter("lama")
res_lama = inpaint_lama(roi, compositor.dilated_mask_roi)
comp_lama = compositor.composite(frame, res_lama)

cv2.imwrite(r"temp_inspect\lama_cat_tail.png", comp_lama)
print("Cat tail frame inpainted with LaMa!")
