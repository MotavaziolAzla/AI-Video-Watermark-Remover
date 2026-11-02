import os
import time
import subprocess
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Dict, Any, Tuple

from config import FFMPEG_PATH, FFPROBE_PATH, HAS_NVENC
from mask_generator import get_mask_bounding_box

class SmartROICompositor:
    """
    Crops the static watermark region of interest with context padding,
    ensures dimensions are multiples of 8, and provides seamless
    feathered alpha-blending back into the original frame.
    """
    def __init__(
        self,
        full_mask: np.ndarray,
        context_padding: int = 48,
        dilation_kernel: int = 9,
        feather_radius: int = 5
    ):
        self.full_mask = full_mask
        self.context_padding = context_padding
        self.dilation_kernel = dilation_kernel
        self.feather_radius = feather_radius
        
        # 1. Compute bounding box on full mask
        self.x1, self.y1, self.x2, self.y2 = get_mask_bounding_box(full_mask, padding=context_padding)
        self.roi_w = self.x2 - self.x1
        self.roi_h = self.y2 - self.y1
        
        # 2. Extract and pre-process mask ROI
        mask_roi = full_mask[self.y1:self.y2, self.x1:self.x2]
        
        # Dilate to cover antialiased text fringes & drop shadows
        if dilation_kernel > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel, dilation_kernel))
            self.dilated_mask_roi = cv2.dilate(mask_roi, k, iterations=1)
        else:
            self.dilated_mask_roi = mask_roi.copy()
            
        # Feather mask edges to create a smooth alpha transition
        if feather_radius > 0:
            ksize = feather_radius * 2 + 1
            feathered = cv2.GaussianBlur(
                self.dilated_mask_roi.astype(np.float32) / 255.0,
                (ksize, ksize),
                feather_radius / 2.0
            )
            # Expand to 3 channels for fast vector multiplication
            self.alpha = np.expand_dims(feathered, axis=2)
        else:
            self.alpha = np.expand_dims((self.dilated_mask_roi > 0).astype(np.float32), axis=2)

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y1:self.y2, self.x1:self.x2]

    def composite(self, original_frame: np.ndarray, inpainted_roi: np.ndarray) -> np.ndarray:
        """
        Seamlessly blend ONLY the masked pixels:
        final_roi = inpainted_roi * alpha + original_roi * (1 - alpha)
        Leaves 100% of non-masked pixels inside the ROI completely untouched!
        """
        orig_roi = original_frame[self.y1:self.y2, self.x1:self.x2].astype(np.float32)
        inp_roi = inpainted_roi.astype(np.float32)
        
        blended_roi = (inp_roi * self.alpha + orig_roi * (1.0 - self.alpha)).astype(np.uint8)
        out_frame = original_frame.copy()
        out_frame[self.y1:self.y2, self.x1:self.x2] = blended_roi
        return out_frame

class SceneCutDetector:
    """
    Multi-metric scene cut detector:
    Uses both Mean Absolute Pixel Difference and HSV color histogram correlation.
    Guarantees that scene cuts and hard transitions instantly trigger a reset.
    """
    def __init__(self, pixel_diff_threshold: float = 12.0, hist_threshold: float = 0.80):
        self.pixel_diff_threshold = pixel_diff_threshold
        self.hist_threshold = hist_threshold
        self.prev_frame = None
        self.prev_hist = None

    def check(self, roi_bgr: np.ndarray) -> Tuple[bool, float]:
        """
        Returns (is_cut: bool, motion_diff: float)
        """
        if self.prev_frame is None:
            self.prev_frame = roi_bgr.copy()
            hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
            self.prev_hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            return True, 0.0
            
        pixel_diff = float(cv2.absdiff(roi_bgr, self.prev_frame).mean())
        
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        hist_sim = cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_CORREL)
        
        self.prev_frame = roi_bgr.copy()
        self.prev_hist = hist
        
        is_cut = (pixel_diff > self.pixel_diff_threshold) or (hist_sim < self.hist_threshold)
        return is_cut, pixel_diff

class MotionCompensatedStabilizer:
    """
    Sub-pixel motion-compensated temporal stabilizer.
    Uses fast DIS optical flow to warp prev_roi onto curr_roi physical coordinates
    before blending, preventing dragging, ghosting, or lagging during camera motion.
    """
    def __init__(self, blend_factor: float = 0.35):
        self.blend_factor = blend_factor
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        self.prev_roi = None
        self.prev_gray = None

    def reset(self):
        self.prev_roi = None
        self.prev_gray = None

    def stabilize(self, curr_roi: np.ndarray, motion_diff: float = 0.0) -> np.ndarray:
        curr_gray = cv2.cvtColor(curr_roi, cv2.COLOR_BGR2GRAY)
        if self.prev_roi is None or motion_diff > 14.0:
            self.prev_roi = curr_roi.copy()
            self.prev_gray = curr_gray
            return curr_roi

        # Pure static scene (< 0.6px): standard EMA is optimal and rock solid
        if motion_diff < 0.6:
            blended = cv2.addWeighted(curr_roi, 1.0 - self.blend_factor, self.prev_roi, self.blend_factor, 0)
        else:
            # Moving scene: warp prev_roi along optical flow to match curr_roi physical position
            flow = self.dis.calc(curr_gray, self.prev_gray, None)
            h, w = curr_gray.shape
            gx, gy = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (gx + flow[:, :, 0]).astype(np.float32)
            map_y = (gy + flow[:, :, 1]).astype(np.float32)
            warped_prev = cv2.remap(self.prev_roi, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            
            # Blend smoothly without ghosting (effective blend scales down gracefully with fast speed)
            eff_blend = self.blend_factor * max(0.0, 1.0 - (motion_diff / 14.0))
            blended = cv2.addWeighted(curr_roi, 1.0 - eff_blend, warped_prev, eff_blend, 0)

        self.prev_roi = blended.copy()
        self.prev_gray = cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)
        return blended

def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """Inspects video metadata reliably via ffprobe / cv2."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {"width": w, "height": h, "fps": fps, "total_frames": total_frames}

def build_ffmpeg_reader(
    video_path: str,
    start_time: float = 0.0,
    duration: Optional[float] = None
) -> subprocess.Popen:
    """Frame-accurate rawvideo pipe reader using FFmpeg (bypasses Windows MSMF HEVC bugs)."""
    cmd = [FFMPEG_PATH, "-loglevel", "error"]
    if start_time > 0:
        cmd.extend(["-ss", f"{start_time:.3f}"])
    cmd.extend(["-i", video_path])
    if duration is not None:
        cmd.extend(["-t", f"{duration:.3f}"])
    cmd.extend(["-f", "rawvideo", "-pix_fmt", "bgr24", "-"])
    
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=10**7
    )

def build_ffmpeg_encoder(
    output_path: str,
    width: int,
    height: int,
    fps: float,
    source_video_for_audio: Optional[str] = None,
    start_time: float = 0.0,
    duration: Optional[float] = None,
    use_nvenc: bool = True
) -> subprocess.Popen:
    """
    Constructs an FFmpeg subprocess that accepts raw BGR24 frames via stdin pipe
    and encodes them with NVENC (or libx264) while copying the original audio stream.
    """
    cmd = [
        FFMPEG_PATH,
        "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "bgr24",
        "-r", f"{fps}",
        "-i", "-" # Read video from stdin pipe
    ]
    
    # If source video is provided, map audio stream from it
    if source_video_for_audio and os.path.exists(source_video_for_audio):
        if start_time > 0:
            cmd.extend(["-ss", f"{start_time:.3f}"])
        if duration is not None:
            cmd.extend(["-t", f"{duration:.3f}"])
        cmd.extend(["-i", source_video_for_audio])
        cmd.extend(["-map", "0:v:0", "-map", "1:a:0?"])
        cmd.extend(["-c:a", "copy"])
    else:
        cmd.extend(["-map", "0:v:0"])
        
    # Video encoder settings
    if use_nvenc and HAS_NVENC:
        cmd.extend([
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-rc", "vbr",
            "-cq", "18",
            "-b:v", "0"
        ])
    else:
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18"
        ])
        
    cmd.extend([
        "-pix_fmt", "yuv420p",
        "-shortest",
        output_path
    ])
    
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        bufsize=10**7
    )
    return proc

def process_video(
    input_path: str,
    output_path: str,
    mask: np.ndarray,
    inpaint_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    start_time: float = 0.0,
    duration: Optional[float] = None,
    use_nvenc: bool = True,
    dilation_kernel: int = 9,
    feather_radius: int = 5,
    context_padding: int = 48,
    enable_temporal_smoothing: bool = False,
    temporal_blend_factor: float = 0.4,
    temporal_chunk_size: int = 24,
    progress_callback: Optional[Callable[[int, int, float, str], None]] = None
) -> Dict[str, Any]:
    """
    Main high-performance video inpainting pipeline.
    Streams frames frame-accurately via FFmpeg, applies smart ROI inpainting and
    compositing with motion-adaptive scene cut protection, and writes directly
    to FFmpeg NVENC pipe without dumping frames to disk.
    """
    meta = get_video_metadata(input_path)
    width = meta["width"]
    height = meta["height"]
    fps = meta["fps"]
    total_video_frames = meta["total_frames"]
    
    frame_size_bytes = width * height * 3
    
    start_frame = int(start_time * fps)
    if duration is not None:
        frames_to_process = min(int(duration * fps), max(1, total_video_frames - start_frame))
    else:
        frames_to_process = max(1, total_video_frames - start_frame)

    # Ensure mask matches video resolution if needed
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    # Prepare Smart Compositor & Scene Detector
    compositor = SmartROICompositor(
        full_mask=mask,
        context_padding=context_padding,
        dilation_kernel=dilation_kernel,
        feather_radius=feather_radius
    )
    scene_detector = SceneCutDetector()

    # Launch FFmpeg streaming reader (frame-accurate)
    reader = build_ffmpeg_reader(input_path, start_time=start_time, duration=duration)

    # Launch FFmpeg streaming writer
    encoder = build_ffmpeg_encoder(
        output_path=output_path,
        width=width,
        height=height,
        fps=fps,
        source_video_for_audio=input_path,
        start_time=start_time,
        duration=duration,
        use_nvenc=use_nvenc
    )

    t0 = time.time()
    frames_done = 0
    frames_read = 0
    
    is_temporal = getattr(inpaint_fn, "is_temporal", False)
    chunk_size = max(8, int(temporal_chunk_size)) if is_temporal else 1
    overlap = 4 if is_temporal else 0
    step_size = chunk_size - overlap if is_temporal else 1
    
    batch_frames = []
    batch_rois = []
    prev_overlap_tail = None
    
    stabilizer = MotionCompensatedStabilizer(blend_factor=temporal_blend_factor) if enable_temporal_smoothing else None

    def flush_chunk(final_flush: bool = False):
        nonlocal frames_done, batch_frames, batch_rois, prev_overlap_tail
        if not batch_rois:
            return
            
        n_frames = len(batch_rois)
        if n_frames == 1:
            inpainted_seq = inpaint_fn.inpaint_sequence([batch_rois[0], batch_rois[0]], compositor.dilated_mask_roi)[:1]
        else:
            inpainted_seq = inpaint_fn.inpaint_sequence(batch_rois, compositor.dilated_mask_roi)
            
        # 1. Overlap cross-fade with previous chunk (seamless transition at matching timestamps)
        if prev_overlap_tail is not None:
            blend_n = min(len(prev_overlap_tail), len(inpainted_seq), overlap)
            for k in range(blend_n):
                w = (k + 1) / float(blend_n + 1)
                inpainted_seq[k] = cv2.addWeighted(inpainted_seq[k], w, prev_overlap_tail[k], 1.0 - w, 0)
                
        # 2. Flow-guided temporal stabilization
        if stabilizer is not None:
            for k in range(len(inpainted_seq)):
                m_d = float(cv2.absdiff(batch_rois[k], batch_rois[max(0, k-1)]).mean())
                inpainted_seq[k] = stabilizer.stabilize(inpainted_seq[k], motion_diff=m_d)
                
        # 3. Determine how many frames to write
        if not final_flush and n_frames >= chunk_size:
            frames_to_write = n_frames - overlap
            prev_overlap_tail = [r.copy() for r in inpainted_seq[-overlap:]]
        else:
            frames_to_write = n_frames
            prev_overlap_tail = None
            
        # 4. Composite & write
        for f, r in zip(batch_frames[:frames_to_write], inpainted_seq[:frames_to_write]):
            final_frame = compositor.composite(f, r)
            encoder.stdin.write(final_frame.tobytes())
            frames_done += 1
            if progress_callback and (frames_done % 4 == 0 or frames_done == frames_to_process):
                elapsed = max(0.001, time.time() - t0)
                cur_fps = frames_done / elapsed
                rem_frames = frames_to_process - frames_done
                eta_sec = rem_frames / cur_fps if cur_fps > 0 else 0
                eta_str = f"{int(eta_sec // 60):02d}:{int(eta_sec % 60):02d}"
                progress_callback(frames_done, frames_to_process, cur_fps, eta_str)
                
        # 5. Slide window forward
        if not final_flush and n_frames >= chunk_size:
            batch_frames = batch_frames[frames_to_write:]
            batch_rois = batch_rois[frames_to_write:]
        else:
            batch_frames.clear()
            batch_rois.clear()

    try:
        while frames_read < frames_to_process:
            raw_frame = reader.stdout.read(frame_size_bytes)
            if not raw_frame or len(raw_frame) < frame_size_bytes:
                break
            frames_read += 1
                
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
            roi = compositor.crop(frame)
            
            is_cut, motion_diff = scene_detector.check(roi)
            
            if is_temporal:
                # If a hard scene cut is detected, flush existing frames immediately without overlap across the cut
                if is_cut and len(batch_rois) > 0:
                    flush_chunk(final_flush=True)
                    if stabilizer is not None:
                        stabilizer.reset()
                batch_frames.append(frame)
                batch_rois.append(roi)
                if len(batch_rois) >= chunk_size:
                    flush_chunk(final_flush=False)
            else:
                # Single-frame inpainter (LaMa / OpenCV)
                inpainted_roi = inpaint_fn(roi, compositor.dilated_mask_roi)
                if stabilizer is not None:
                    if is_cut:
                        stabilizer.reset()
                    inpainted_roi = stabilizer.stabilize(inpainted_roi, motion_diff=motion_diff)
                
                # Composite seamlessly into frame
                final_frame = compositor.composite(frame, inpainted_roi)
                
                # Write directly to FFmpeg pipe
                encoder.stdin.write(final_frame.tobytes())
                frames_done += 1
                
                if progress_callback and (frames_done % 6 == 0 or frames_done == frames_to_process):
                    elapsed = max(0.001, time.time() - t0)
                    cur_fps = frames_done / elapsed
                    rem_frames = frames_to_process - frames_done
                    eta_sec = rem_frames / cur_fps if cur_fps > 0 else 0
                    eta_str = f"{int(eta_sec // 60):02d}:{int(eta_sec % 60):02d}"
                    progress_callback(frames_done, frames_to_process, cur_fps, eta_str)
                    
        # Flush any remaining frames in temporal queue
        if is_temporal and len(batch_rois) > 0:
            flush_chunk(final_flush=True)
                
    finally:
        if reader.stdout:
            reader.stdout.close()
        reader.wait()
        
        if encoder.stdin:
            try:
                encoder.stdin.close()
            except Exception:
                pass
        _, stderr = encoder.communicate()
        if encoder.returncode != 0:
            err_msg = stderr.decode(errors='replace') if stderr else "Unknown error"
            raise RuntimeError(f"FFmpeg encoding error (code {encoder.returncode}): {err_msg}")

    total_time = max(0.001, time.time() - t0)
    avg_fps = frames_done / total_time
    
    return {
        "frames_processed": frames_done,
        "elapsed_sec": total_time,
        "avg_fps": avg_fps,
        "output_path": output_path
    }
