import cv2
import numpy as np

# Load video
cap = cv2.VideoCapture(r"imdl5-34p1.mp4")
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {width}x{height}, {frame_count} frames, {fps} fps")

# Sample 30 frames evenly
sample_indices = np.linspace(0, frame_count - 1, 30, dtype=int)
samples = []
for idx in sample_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
        samples.append(frame)
cap.release()

print(f"Collected {len(samples)} sample frames")

# Stack and compute temporal minimum and temporal median
stack = np.array(samples) # shape: (N, H, W, 3)
# In the bottom-right corner:
# x: width - 300 to width - 20
# y: height - 120 to height - 20
h_start, h_end = height - 120, height - 10
w_start, w_end = width - 300, width - 10

br_stack = stack[:, h_start:h_end, w_start:w_end, :]

# In the bottom right corner, let's find the text.
# The text "Dola AI" is bright white across EVERY frame!
# So min across all frames will be bright ONLY where the text is!
min_frame = np.min(br_stack, axis=0) # shape (H_roi, W_roi, 3)
gray_min = cv2.cvtColor(min_frame, cv2.COLOR_BGR2GRAY)

# Threshold the temporal minimum
_, mask_tight = cv2.threshold(gray_min, 160, 255, cv2.THRESH_BINARY)

cv2.imwrite(r"temp_inspect\min_corner.png", min_frame)
cv2.imwrite(r"temp_inspect\mask_temporal_min.png", mask_tight)

# Also let's dilate slightly (e.g. 3px kernel) to cover antialiased edges
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask_dilated = cv2.dilate(mask_tight, kernel, iterations=1)
cv2.imwrite(r"temp_inspect\mask_dilated.png", mask_dilated)

print("Saved min_corner.png, mask_temporal_min.png, and mask_dilated.png")
