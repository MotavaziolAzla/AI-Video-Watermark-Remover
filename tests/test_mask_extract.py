import cv2
import numpy as np

# Load the sample frame
frame = cv2.imread(r"temp_inspect\frame1.png")
h, w, _ = frame.shape
print(f"Frame shape: {w}x{h}")

# In frame1, we know "Dola AI" is at the bottom right.
# Let's crop the bottom right 350x180 region:
# x: from w-350 to w, y: from h-180 to h
roi_full = frame[h-180:h, w-350:w]

# Let's inspect the text. "Dola AI" is bright white letters with some slight shadow/anti-aliasing.
# Let's convert to grayscale
gray = cv2.cvtColor(roi_full, cv2.COLOR_BGR2GRAY)

# Threshold to isolate the bright white text
# Usually white text has high values in grayscale (> 200 or 210)
_, thresh = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)

# Save the cropped ROI and threshold
cv2.imwrite(r"temp_inspect\test_roi.png", roi_full)
cv2.imwrite(r"temp_inspect\test_thresh.png", thresh)
print("Saved test_roi.png and test_thresh.png")
