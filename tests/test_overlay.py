import cv2
import numpy as np

# Load sample frame and mask
frame = cv2.imread(r"temp_inspect\cut_10s\crop_041.png")
mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)
mask_roi = mask[540:720, 930:1280]

# Dilate and feather
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
dilated = cv2.dilate(mask_roi, k)

# Create a red overlay: where mask > 0, blend with bright red (0, 0, 255)
overlay = frame.copy()
red_color = np.array([0, 0, 255], dtype=np.uint8) # BGR red
alpha = 0.55
mask_bool = dilated > 0
overlay[mask_bool] = (overlay[mask_bool].astype(np.float32) * (1 - alpha) + red_color * alpha).astype(np.uint8)

# Draw a thin yellow contour around the mask boundary
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)

cv2.imwrite(r"temp_inspect\mask_overlay_demo.png", overlay)
print("Saved mask_overlay_demo.png")
