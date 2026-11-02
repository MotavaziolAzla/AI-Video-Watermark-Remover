import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

def extract_sample_frame(video_path: str, timestamp_sec: float = 1.0) -> Optional[np.ndarray]:
    """Extract a single frame from the video at the given timestamp."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_idx = int(timestamp_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def auto_detect_temporal_watermark(
    video_path: str,
    num_samples: int = 25,
    corner_box: Optional[Tuple[int, int, int, int]] = None,
    threshold_val: int = 150
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes a temporal minimum across frames to isolate static bright watermarks
    from moving backgrounds, producing a clean binary mask.
    
    corner_box: (x1, y1, x2, y2). If None, defaults to bottom-right 25% of frame.
    Returns: (full_frame_mask, preview_temporal_min)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if corner_box is None:
        # Default bottom-right corner: x from 75% to 99%, y from 80% to 99%
        x1 = int(width * 0.70)
        y1 = int(height * 0.75)
        x2 = width - 5
        y2 = height - 5
    else:
        x1, y1, x2, y2 = corner_box
    
    # Sample frames evenly
    sample_indices = np.linspace(0, max(0, total_frames - 1), num_samples, dtype=int)
    crops = []
    
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            crops.append(frame[y1:y2, x1:x2])
            
    cap.release()
    
    if not crops:
        raise RuntimeError("Failed to extract frames from video.")
        
    stack = np.array(crops) # (N, H_roi, W_roi, 3)
    # Temporal minimum makes moving background dark, keeping static white watermark bright
    min_roi = np.min(stack, axis=0)
    gray_min = cv2.cvtColor(min_roi, cv2.COLOR_BGR2GRAY)
    
    # Threshold the bright text
    _, mask_roi = cv2.threshold(gray_min, threshold_val, 255, cv2.THRESH_BINARY)
    
    # Filter small noise particles (keep only significant text contours)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_roi)
    cleaned_mask_roi = np.zeros_like(mask_roi)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # Text letter strokes typically have area > 10 pixels
        if area >= 12:
            cleaned_mask_roi[labels == i] = 255
            
    # Place ROI into full frame mask
    full_mask = np.zeros((height, width), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = cleaned_mask_roi
    
    # Preview image
    full_preview = np.zeros((height, width, 3), dtype=np.uint8)
    full_preview[y1:y2, x1:x2] = min_roi
    
    return full_mask, full_preview

def get_mask_bounding_box(mask: np.ndarray, padding: int = 48) -> Tuple[int, int, int, int]:
    """
    Finds bounding box of non-zero pixels in mask, adds padding,
    and ensures width and height are divisible by 8.
    Returns (x1, y1, x2, y2).
    """
    h, w = mask.shape[:2]
    pts = cv2.findNonZero(mask)
    if pts is None:
        # Fallback to bottom-right corner if mask is empty
        return (max(0, w - 256), max(0, h - 128), w, h)
        
    x, y, bw, bh = cv2.boundingRect(pts)
    
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w, x + bw + padding)
    y2 = min(h, y + bh + padding)
    
    roi_w = x2 - x1
    roi_h = y2 - y1
    
    # Make multiple of 8 (for DL models)
    rem_w = roi_w % 8
    if rem_w != 0:
        add_w = 8 - rem_w
        if x2 + add_w <= w:
            x2 += add_w
        elif x1 - add_w >= 0:
            x1 -= add_w
            
    rem_h = roi_h % 8
    if rem_h != 0:
        add_h = 8 - rem_h
        if y2 + add_h <= h:
            y2 += add_h
        elif y1 - add_h >= 0:
            y1 -= add_h
            
    return (x1, y1, x2, y2)
