import cv2
import numpy as np

class EnhancedSceneCutDetector:
    def __init__(self, pixel_diff_threshold: float = 12.0, hist_threshold: float = 0.80):
        self.pixel_diff_threshold = pixel_diff_threshold
        self.hist_threshold = hist_threshold
        self.prev_frame = None
        self.prev_hist = None

    def is_cut(self, roi_bgr: np.ndarray) -> bool:
        if self.prev_frame is None:
            self.prev_frame = roi_bgr.copy()
            hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
            self.prev_hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            return True
            
        pixel_diff = float(cv2.absdiff(roi_bgr, self.prev_frame).mean())
        
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        hist_sim = cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_CORREL)
        
        self.prev_frame = roi_bgr.copy()
        self.prev_hist = hist
        
        is_cut_event = (pixel_diff > self.pixel_diff_threshold) or (hist_sim < self.hist_threshold)
        print(f"Diff: {pixel_diff:.2f}, Sim: {hist_sim:.2f} -> Cut: {is_cut_event}")
        return is_cut_event

detector = EnhancedSceneCutDetector()
for i in range(38, 45):
    img = cv2.imread(f"temp_inspect/cut_10s/crop_{i:03d}.png")
    print(f"Frame {i}: ", end="")
    detector.is_cut(img)
