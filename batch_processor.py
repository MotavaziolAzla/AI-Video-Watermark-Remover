import os
import sys
import glob
import time
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any

from config import DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_MASK_PATH, HAS_NVENC, get_mask_path
from engine import get_inpainter
from pipeline import process_video

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

def find_video_files(folder_path: str) -> List[str]:
    """Find all video files in the specified directory."""
    folder = Path(folder_path)
    if not folder.is_dir():
        return []
    
    files = []
    for f in folder.iterdir():
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS:
            files.append(str(f))
    return sorted(files)

def run_batch_processing(
    input_dir: str,
    output_dir: str,
    mask_path: str,
    engine_name: str = "ns",
    use_nvenc: bool = True,
    dilation_kernel: int = 9,
    feather_radius: int = 5,
    context_padding: int = 48,
    enable_temporal_smoothing: bool = False,
    temporal_chunk_size: int = 24,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
) -> List[Dict[str, Any]]:
    """
    Processes all video files in input_dir and saves cleaned versions to output_dir.
    """
    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask file not found or invalid: {mask_path}")
        
    videos = find_video_files(input_dir)
    if not videos:
        # Check if input_dir itself is a single video file
        if Path(input_dir).is_file() and Path(input_dir).suffix.lower() in VIDEO_EXTENSIONS:
            videos = [input_dir]
        else:
            raise ValueError(f"No video files found in: {input_dir}")
            
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    
    inpaint_fn = get_inpainter(engine_name)
    
    results = []
    total_videos = len(videos)
    
    for v_idx, video_path in enumerate(videos, 1):
        v_name = Path(video_path).name
        out_name = f"cleaned_{v_name}"
        out_path = str(out_dir_path / out_name)
        
        status_info = {
            "current_video": v_name,
            "video_index": v_idx,
            "total_videos": total_videos,
            "status": "Processing",
            "frames_done": 0,
            "total_frames": 0,
            "fps": 0.0,
            "eta": "--:--"
        }
        
        if progress_callback:
            progress_callback(status_info)
            
        def per_frame_cb(done, total, fps, eta):
            if progress_callback:
                status_info["frames_done"] = done
                status_info["total_frames"] = total
                status_info["fps"] = fps
                status_info["eta"] = eta
                progress_callback(status_info)
                
        try:
            stat = process_video(
                input_path=video_path,
                output_path=out_path,
                mask=mask,
                inpaint_fn=inpaint_fn,
                use_nvenc=use_nvenc,
                dilation_kernel=dilation_kernel,
                feather_radius=feather_radius,
                context_padding=context_padding,
                enable_temporal_smoothing=enable_temporal_smoothing,
                temporal_chunk_size=temporal_chunk_size,
                progress_callback=per_frame_cb
            )
            stat["filename"] = v_name
            stat["status"] = "Success"
            results.append(stat)
        except Exception as e:
            results.append({
                "filename": v_name,
                "status": f"Failed: {str(e)}",
                "frames_processed": 0,
                "elapsed_sec": 0,
                "avg_fps": 0,
                "output_path": out_path
            })
            
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Watermark Inpainter CLI")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_DIR), help="Input directory containing videos")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--mask", default=None, help="Path to custom mask image or mask preset name ('dola_ai' or 'gemini')")
    parser.add_argument("--preset", "--mask-preset", default="dola_ai", choices=["dola_ai", "gemini", "dola", "Dola AI", "Gemini"], help="Watermark mask preset ('dola_ai' or 'gemini')")
    parser.add_argument("--engine", default="hybrid", choices=["hybrid", "propainter", "e2fgvi", "lama", "ns", "telea"], help="Inpainting engine")
    parser.add_argument("--nvenc", action="store_true", default=HAS_NVENC, help="Use NVENC hardware encoding")
    parser.add_argument("--dilation", type=int, default=13, help="Mask dilation kernel size")
    parser.add_argument("--feather", type=int, default=7, help="Mask feather radius")
    parser.add_argument("--padding", type=int, default=96, help="Context ROI padding")
    parser.add_argument("--temporal", action="store_true", default=False, help="Enable temporal smoothing (for single-frame engines)")
    parser.add_argument("--chunk-size", type=int, default=24, help="Temporal chunk size for video inpainters (e.g. 24 for 6GB, 48-64 for 24GB)")
    
    args = parser.parse_args()
    
    # Resolve mask: custom file path or preset
    if args.mask and os.path.exists(args.mask):
        resolved_mask = args.mask
    elif args.mask and args.mask.lower() in ["gemini", "dola", "dola_ai", "dola ai"]:
        resolved_mask = str(get_mask_path(args.mask))
    else:
        resolved_mask = str(get_mask_path(args.preset))
    
    print(f"Starting batch inpainting: {args.input} -> {args.output}")
    print(f"Engine: {args.engine.upper()} | NVENC: {args.nvenc} | Mask: {resolved_mask} | Chunk Size: {args.chunk_size}")
    
    res = run_batch_processing(
        input_dir=args.input,
        output_dir=args.output,
        mask_path=resolved_mask,
        engine_name=args.engine,
        use_nvenc=args.nvenc,
        dilation_kernel=args.dilation,
        feather_radius=args.feather,
        context_padding=args.padding,
        enable_temporal_smoothing=args.temporal,
        temporal_chunk_size=args.chunk_size,
        progress_callback=lambda s: print(f"[{s['video_index']}/{s['total_videos']}] {s['current_video']} - {s['frames_done']}/{s['total_frames']} frames ({s['fps']:.1f} FPS, ETA: {s['eta']})", end="\r")
    )
    print("\nBatch processing completed.")
    for r in res:
        print(f"- {r['filename']}: {r['status']} ({r.get('avg_fps', 0):.1f} avg FPS, {r.get('elapsed_sec', 0):.1f}s)")
