import os
import time
import shutil
import cv2
import numpy as np
import gradio as gr
from pathlib import Path
from typing import Tuple, Optional

from config import (
    BASE_DIR,
    DEFAULT_MASK_PATH,
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    HAS_NVENC,
    FFMPEG_PATH,
    MASK_PRESETS,
    get_mask_path
)
from mask_generator import (
    extract_sample_frame,
    auto_detect_temporal_watermark,
    get_mask_bounding_box
)
from engine import get_inpainter
from pipeline import SmartROICompositor, process_video
from batch_processor import run_batch_processing, find_video_files

# Global state
def get_default_sample_video() -> str:
    candidates = sorted(list(DEFAULT_INPUT_DIR.glob("*.mp4")) + list(DEFAULT_INPUT_DIR.glob("*.mkv")))
    if candidates:
        # If any video contains Fox or Gemini, prioritize it if present, otherwise first
        for c in candidates:
            if "fox" in c.name.lower() or "gemini" in c.name.lower():
                return str(c)
        return str(candidates[0])
    sample_candidate = BASE_DIR / "input_videos" / "sample_test.mp4"
    if sample_candidate.exists():
        return str(sample_candidate)
    return ""

sample_video_default = get_default_sample_video()

# Default preset based on default video
default_preset = "Gemini" if ("fox" in sample_video_default.lower() or "gemini" in sample_video_default.lower()) else "Dola AI"
current_mask_preset = default_preset

# Load default mask into cache
cached_mask = None
initial_mask_path = get_mask_path(current_mask_preset)
if initial_mask_path.exists():
    cached_mask = cv2.imread(str(initial_mask_path), cv2.IMREAD_GRAYSCALE)

def get_system_badge_html(active_preset: str = default_preset):
    import torch
    cuda_ok = torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0) if cuda_ok else "CPU Only"
    nvenc_badge = "<span style='background:#10b981;color:white;padding:3px 10px;border-radius:6px;font-weight:600;'>🟢 NVENC Hardware Accelerated</span>" if HAS_NVENC else "<span style='background:#f59e0b;color:white;padding:3px 10px;border-radius:6px;'>CPU x264 Fallback</span>"
    cuda_badge = f"<span style='background:#3b82f6;color:white;padding:3px 10px;border-radius:6px;font-weight:600;'>⚡ GPU CUDA: {dev_name}</span>" if cuda_ok else "<span style='background:#ef4444;color:white;padding:3px 10px;border-radius:6px;'>CUDA Unavailable</span>"
    mask_badge = f"<span style='background:#8b5cf6;color:white;padding:3px 10px;border-radius:6px;font-weight:600;'>🎯 Active Mask: {active_preset} ({active_preset.lower().replace(' ', '_')}.png)</span>" if cached_mask is not None else "<span style='background:#ef4444;color:white;padding:3px 10px;border-radius:6px;'>❌ No mask loaded</span>"
    
    return f"""
    <div style='display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-bottom:14px;font-size:13px;'>
      {cuda_badge}
      {nvenc_badge}
      {mask_badge}
      <span style='color:#6b7280;background:#f3f4f6;padding:3px 10px;border-radius:6px;'>FFmpeg: {Path(FFMPEG_PATH).name}</span>
    </div>
    """

def get_config_summary_html(engine, mask_preset, dilation, feather, padding, nvenc, temporal, chunk_size=24):
    return f"""
    <div style='background:#eef2ff;border-left:4px solid #6366f1;padding:10px 14px;border-radius:6px;margin:8px 0 16px 0;font-size:13px;color:#1e1b4b;'>
      <strong>📌 Active Master Configuration (Applies to all Tabs & Batch):</strong><br/>
      • <strong>Target Mask:</strong> <span style='background:#e0e7ff;padding:2px 8px;border-radius:4px;font-weight:bold;color:#4338ca;'>{mask_preset}</span> &nbsp;|&nbsp;
      • <strong>Engine:</strong> {engine} &nbsp;|&nbsp;
      • <strong>Dilation:</strong> {int(dilation)}px &nbsp;|&nbsp;
      • <strong>Feather:</strong> {int(feather)}px &nbsp;|&nbsp;
      • <strong>Context Padding:</strong> {int(padding)}px &nbsp;|&nbsp;
      • <strong>Chunk Window:</strong> {int(chunk_size)} frames &nbsp;|&nbsp;
      • <strong>NVENC HW:</strong> {'Enabled' if nvenc else 'Disabled'} &nbsp;|&nbsp;
      • <strong>Temporal Smoothing:</strong> {'Active (Tripod only)' if temporal else 'Off (Clean motion)'}
    </div>
    """

def switch_mask_preset_action(preset_name: str):
    """Switch active mask between 'Dola AI' and 'Gemini' presets."""
    global cached_mask, current_mask_preset
    current_mask_preset = preset_name
    mask_path = get_mask_path(preset_name)
    if mask_path.exists():
        cached_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        # Also mirror to root mask.png for backward compatibility
        try:
            cv2.imwrite(str(DEFAULT_MASK_PATH), cached_mask)
        except Exception:
            pass
        status = f"✅ Switched active mask to **{preset_name}** (`{mask_path.name}`)."
    else:
        status = f"⚠️ Preset file for **{preset_name}** not found at `{mask_path}`."
        
    badge_html = get_system_badge_html(preset_name)
    return badge_html, status

def create_mask_overlay(crop_bgr: np.ndarray, mask_roi: np.ndarray) -> np.ndarray:
    """Creates a transparent red highlight with a crisp yellow contour line over the watermark."""
    overlay = crop_bgr.copy()
    red = np.array([0, 0, 255], dtype=np.uint8) # BGR
    mask_bool = mask_roi > 0
    alpha = 0.52
    overlay[mask_bool] = (overlay[mask_bool].astype(np.float32) * (1 - alpha) + red * alpha).astype(np.uint8)
    
    contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)
    return overlay

def auto_detect_mask_action(video_path: str, num_samples: int, threshold_val: int):
    if not video_path or not os.path.exists(video_path):
        return None, "❌ Video file not found."
    try:
        mask, preview = auto_detect_temporal_watermark(
            video_path,
            num_samples=int(num_samples),
            threshold_val=int(threshold_val)
        )
        global cached_mask
        cached_mask = mask.copy()
        cv2.imwrite(str(DEFAULT_MASK_PATH), mask)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        return mask_rgb, f"✅ Watermark mask detected across {num_samples} frames and saved to mask.png!"
    except Exception as e:
        return None, f"❌ Error during auto-detection: {str(e)}"

def save_active_mask_action():
    global cached_mask, current_mask_preset
    if cached_mask is None:
        return "❌ No mask loaded in memory to save."
    target_preset_path = get_mask_path(current_mask_preset)
    cv2.imwrite(str(target_preset_path), cached_mask)
    cv2.imwrite(str(DEFAULT_MASK_PATH), cached_mask)
    return f"✅ Successfully saved current mask to **{current_mask_preset}** (`{target_preset_path.name}`) and `mask.png`!"

def update_visual_inspector(
    video_path: str,
    timestamp: float,
    mask_preset: str,
    engine_name: str,
    dilation_kernel: int,
    feather_radius: int,
    context_padding: int
):
    if not video_path or not os.path.exists(video_path):
        return None, None, None, None, None, "❌ Video file not found."
        
    global cached_mask, current_mask_preset
    if current_mask_preset != mask_preset or cached_mask is None:
        current_mask_preset = mask_preset
        m_path = get_mask_path(mask_preset)
        if m_path.exists():
            cached_mask = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
            
    if cached_mask is None:
        return None, None, None, None, None, "⚠️ Please select a valid mask preset or run Auto-Detect."
        
    frame = extract_sample_frame(video_path, timestamp)
    if frame is None:
        return None, None, None, None, None, "❌ Could not extract frame at timestamp."
    
    # Scale mask if video resolution differs
    fh, fw = frame.shape[:2]
    active_m = cached_mask
    if active_m.shape[:2] != (fh, fw):
        active_m = cv2.resize(active_m, (fw, fh), interpolation=cv2.INTER_NEAREST)
        
    compositor = SmartROICompositor(
        full_mask=active_m,
        context_padding=int(context_padding),
        dilation_kernel=int(dilation_kernel),
        feather_radius=int(feather_radius)
    )
    
    inpaint_fn = get_inpainter(engine_name)
    roi = compositor.crop(frame)
    
    if getattr(inpaint_fn, "is_temporal", False):
        # Extract a small 3-frame local temporal window for video optical flow
        fps = 24.0
        t_prev = max(0.0, timestamp - 1.0 / fps)
        t_next = timestamp + 1.0 / fps
        f_prev = extract_sample_frame(video_path, t_prev)
        if f_prev is None: f_prev = frame
        f_next = extract_sample_frame(video_path, t_next)
        if f_next is None: f_next = frame
        seq_rois = [compositor.crop(f_prev), roi, compositor.crop(f_next)]
        inpainted_seq = inpaint_fn.inpaint_sequence(seq_rois, compositor.dilated_mask_roi)
        inpainted_roi = inpainted_seq[1]
    else:
        inpainted_roi = inpaint_fn(roi, compositor.dilated_mask_roi)
        
    full_cleaned = compositor.composite(frame, inpainted_roi)
    
    # 1. Original Corner
    corner_orig_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    # 2. Red Highlight Overlay
    overlay_bgr = create_mask_overlay(roi, compositor.dilated_mask_roi)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    
    # 3. Clean Corner
    corner_clean_rgb = cv2.cvtColor(inpainted_roi, cv2.COLOR_BGR2RGB)
    
    # Binary Mask ROI
    mask_roi_rgb = cv2.cvtColor(compositor.dilated_mask_roi, cv2.COLOR_GRAY2RGB)
    
    # Full Frame Result
    full_clean_rgb = cv2.cvtColor(full_cleaned, cv2.COLOR_BGR2RGB)
    
    status_msg = f"✅ Inspected frame at {timestamp:.2f}s | Target: **{mask_preset}** | Inpainted with **{engine_name.split()[0]}** ({compositor.roi_w}x{compositor.roi_h}px ROI)."
    return corner_orig_rgb, overlay_rgb, corner_clean_rgb, mask_roi_rgb, full_clean_rgb, status_msg

def run_test_video_action(
    video_path: str,
    start_time: float,
    duration: float,
    mask_preset: str,
    engine_name: str,
    use_nvenc: bool,
    dilation_kernel: int,
    feather_radius: int,
    context_padding: int,
    enable_temporal: bool,
    chunk_size: int = 24,
    progress=gr.Progress(track_tqdm=True)
):
    if not video_path or not os.path.exists(video_path):
        return None, "❌ Video file not found."
        
    global cached_mask
    mask_path = get_mask_path(mask_preset)
    if not mask_path.exists() and cached_mask is None:
        return None, f"❌ Mask for {mask_preset} not found."
    
    active_m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else cached_mask
        
    out_dir = BASE_DIR / "temp_test_renders"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video_path = str(out_dir / f"test_{Path(video_path).stem}_{mask_preset.lower().replace(' ', '_')}_{int(time.time())}.mp4")
    
    inpaint_fn = get_inpainter(engine_name)
    
    def on_progress(done, total, fps, eta):
        progress(done / max(1, total), desc=f"Rendering {done}/{total} frames ({fps:.1f} FPS, ETA: {eta})")
        
    stats = process_video(
        input_path=video_path,
        output_path=out_video_path,
        mask=active_m,
        inpaint_fn=inpaint_fn,
        start_time=start_time,
        duration=duration,
        use_nvenc=use_nvenc,
        dilation_kernel=int(dilation_kernel),
        feather_radius=int(feather_radius),
        context_padding=int(context_padding),
        enable_temporal_smoothing=enable_temporal,
        temporal_chunk_size=int(chunk_size),
        progress_callback=on_progress
    )
    
    info_text = f"✅ **Render Complete!**\n• **Target Mask:** `{mask_preset}`\n• **Frames Rendered:** {stats['frames_processed']}\n• **Render Time:** {stats['elapsed_sec']:.2f}s\n• **Speed:** {stats['avg_fps']:.1f} FPS\n• **File:** `{Path(out_video_path).name}`"
    return out_video_path, info_text

def run_batch_ui_action(
    input_folder: str,
    output_folder: str,
    mask_preset: str,
    engine_name: str,
    use_nvenc: bool,
    dilation_kernel: int,
    feather_radius: int,
    context_padding: int,
    enable_temporal: bool,
    chunk_size: int = 24,
    progress=gr.Progress(track_tqdm=True)
):
    mask_path = get_mask_path(mask_preset)
    if not mask_path.exists():
        return f"❌ Error: Mask file for '{mask_preset}' not found at `{mask_path}`."
        
    videos = find_video_files(input_folder)
    if not videos:
        return f"❌ No video files (.mp4, .mov, etc.) found in `{input_folder}`."
        
    def batch_cb(info):
        cur_v = info["video_index"]
        tot_v = info["total_videos"]
        v_name = info["current_video"]
        f_done = info["frames_done"]
        f_tot = max(1, info["total_frames"])
        fps = info["fps"]
        eta = info["eta"]
        overall_frac = ((cur_v - 1) + (f_done / f_tot)) / max(1, tot_v)
        progress(overall_frac, desc=f"[{cur_v}/{tot_v}] {v_name} ({f_done}/{f_tot} frames, {fps:.1f} FPS, ETA: {eta})")

    results = run_batch_processing(
        input_dir=input_folder,
        output_dir=output_folder,
        mask_path=str(mask_path),
        engine_name=engine_name,
        use_nvenc=use_nvenc,
        dilation_kernel=int(dilation_kernel),
        feather_radius=int(feather_radius),
        context_padding=int(context_padding),
        enable_temporal_smoothing=enable_temporal,
        temporal_chunk_size=int(chunk_size),
        progress_callback=batch_cb
    )
    
    summary = ["### 📊 Batch Processing Results", f"*Active Mask Profile: **{mask_preset}***", "| Video File | Status | Frames | Time | Avg FPS |", "|---|---|---|---|---|"]
    for r in results:
        summary.append(f"| **{r['filename']}** | {r['status']} | {r.get('frames_processed', 0)} | {r.get('elapsed_sec', 0):.1f}s | {r.get('avg_fps', 0):.1f} |")
        
    return "\n".join(summary)


# -------------------------------------------------------------
# Gradio UI Definition
# -------------------------------------------------------------
with gr.Blocks(title="AI Video Watermark Remover") as demo:
    gr.Markdown("# ⚡ AI Video Watermark Remover")
    system_badge_display = gr.HTML(get_system_badge_html(current_mask_preset))
    
    with gr.Row():
        # LEFT COLUMN: Persistent Master Controls (Applied to ALL Tabs)
        with gr.Column(scale=4, min_width=320):
            gr.Markdown("### ⚙️ Master Inpainting Settings")
            gr.Markdown("<small style='color:#6b7280;'>These parameters govern the mask, blending, and inpainter across <strong>ALL</strong> tabs (Preview, Test Video, and Batch Automation).</small>")
            
            # Watermark Preset Selector
            master_mask_preset = gr.Radio(
                ["Dola AI", "Gemini"],
                value=default_preset,
                label="🎯 Watermark Target Mask",
                info="Switch profile: 'Dola AI' (bottom-right text) or 'Gemini' (bottom-right sparkle logo)."
            )
            
            master_engine = gr.Radio(
                [
                    "Hybrid (ProPainter Structure + AI LaMa Tracking) [RECOMMENDED]",
                    "ProPainter (Video SOTA - Temporal Flow)",
                    "AI LaMa (GPU Deep Learning)",
                    "E2FGVI-HQ (Fast Video Inpainter)",
                    "OpenCV Navier-Stokes (NS)",
                    "OpenCV Telea"
                ],
                value="Hybrid (ProPainter Structure + AI LaMa Tracking) [RECOMMENDED]",
                label="Inpainting Engine",
                info="Hybrid: Combines ProPainter's intelligent content reconstruction with LaMa's rock-solid tracking stability."
            )
            
            master_dilation = gr.Slider(
                1, 25, value=11, step=2,
                label="Mask Dilation Kernel (px)",
                info="Expands mask outward. 11-13px covers drop-shadows & font fringes cleanly."
            )
            
            master_feather = gr.Slider(
                0, 16, value=7, step=1,
                label="Feather Radius (px)",
                info="Gaussian edge softening. 7px eliminates rectangular seams."
            )
            
            master_padding = gr.Slider(
                32, 160, value=64, step=8,
                label="Context ROI Padding (px)",
                info="64-96px context provides full geometry for poles, rails, and facades."
            )
            
            with gr.Accordion("Advanced Encoding & Temporal Settings", open=False):
                master_nvenc = gr.Checkbox(
                    value=HAS_NVENC,
                    label="NVENC Hardware Encoding (h264_nvenc)",
                    info="Uses RTX GPU hardware encoder for maximum speed."
                )
                master_temporal = gr.Checkbox(
                    value=True,
                    label="Temporal Motion-Adaptive Smoothing",
                    info="Eliminates frame-to-frame micro-flicker and stabilizes chunk boundaries while preserving motion."
                )
                master_chunk_size = gr.Slider(
                    8, 64, value=24, step=8,
                    label="Temporal Window Chunk Size (frames)",
                    info="Sequence length for ProPainter/E2FGVI. 24 frames on 6GB laptop; 48-64 frames on 24GB VRAM GPU."
                )

        # RIGHT COLUMN: The 3 Main Workflow Tabs
        with gr.Column(scale=7):
            with gr.Tabs():
                
                # TAB 1: Visual Mask & Single Frame Inspector
                with gr.Tab("🎯 1. Mask Inspector & Single Frame Preview"):
                    gr.Markdown("Inspect how your mask covers the watermark and see live before/after results.")
                    
                    with gr.Row():
                        tab1_video = gr.Textbox(
                            label="Video Path",
                            value=sample_video_default,
                            scale=3
                        )
                        tab1_time = gr.Slider(
                            0.0, 15.0, value=1.0, step=0.25,
                            label="Timestamp (sec)",
                            scale=3
                        )
                        btn_inspect = gr.Button("🔄 Inspect Frame", variant="primary", scale=2)

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### 🔍 Corner Zoom: Original vs. Mask Highlight")
                            with gr.Row():
                                img_orig_corner = gr.Image(label="1. Original Corner", interactive=False)
                                img_overlay_corner = gr.Image(label="2. Mask Coverage (Red = Mask, Yellow = Border)", interactive=False)
                                
                        with gr.Column():
                            gr.Markdown("#### ✨ Inpainted Clean Corner")
                            img_clean_corner = gr.Image(label="3. Clean Inpainted Result", interactive=False)

                    with gr.Accordion("Full Frame View & Binary Mask", open=False):
                        with gr.Row():
                            img_mask_binary = gr.Image(label="Binary Dilated Mask ROI", interactive=False)
                            img_full_frame = gr.Image(label="Full Frame Clean Result", interactive=False)

                    with gr.Row():
                        btn_detect = gr.Button("✨ Auto-Detect Watermark from Video", variant="secondary")
                        btn_save_mask = gr.Button("💾 Save Active Mask", variant="secondary")
                        
                    tab1_status = gr.Markdown("Ready to inspect.")

                # TAB 2: 5-Second Video Test
                with gr.Tab("🎬 2. Video Test Render"):
                    gr.Markdown("Render a test clip with full audio to verify temporal consistency and playback.")
                    
                    # Live status banner showing master settings
                    banner_tab2 = gr.HTML(get_config_summary_html(
                        "Hybrid", default_preset, 11, 7, 64, HAS_NVENC, True
                    ))
                    
                    with gr.Row():
                        tab2_video = gr.Textbox(label="Video Path", value=sample_video_default, scale=3)
                        tab2_start = gr.Slider(0.0, 15.0, value=0.0, step=0.5, label="Start Time (sec)", scale=2)
                        tab2_dur = gr.Slider(1.0, 10.0, value=4.0, step=0.5, label="Duration (sec)", scale=2)
                        
                    btn_test_video = gr.Button("🚀 Render Test Video Clip", variant="primary")
                    
                    with gr.Row():
                        test_video_player = gr.Video(label="Rendered Video Player (with Audio)")
                        test_video_log = gr.Markdown("Ready.")

                # TAB 3: Batch Folder Automation
                with gr.Tab("📁 3. Batch Folder Automation"):
                    gr.Markdown("Process entire folders of videos automatically using the current mask and master settings.")
                    
                    # Live status banner showing master settings
                    banner_tab3 = gr.HTML(get_config_summary_html(
                        "Hybrid", default_preset, 11, 7, 64, HAS_NVENC, True
                    ))
                    
                    with gr.Row():
                        batch_input = gr.Textbox(label="Input Videos Folder", value=str(DEFAULT_INPUT_DIR))
                        batch_output = gr.Textbox(label="Output Videos Folder", value=str(DEFAULT_OUTPUT_DIR))
                        
                    btn_run_batch = gr.Button("⚡ Start Batch Folder Processing", variant="primary")
                    batch_report_md = gr.Markdown("Batch report will be displayed here.")

    # ---------------------------------------------------------
    # Event Handlers
    # ---------------------------------------------------------
    
    # Update status banners when master parameters change
    def update_banners(engine, mask_preset, dilation, feather, padding, nvenc, temporal, chunk_size):
        html = get_config_summary_html(engine, mask_preset, dilation, feather, padding, nvenc, temporal, chunk_size)
        return html, html

    for ctrl in [master_engine, master_mask_preset, master_dilation, master_feather, master_padding, master_nvenc, master_temporal, master_chunk_size]:
        ctrl.change(
            update_banners,
            inputs=[master_engine, master_mask_preset, master_dilation, master_feather, master_padding, master_nvenc, master_temporal, master_chunk_size],
            outputs=[banner_tab2, banner_tab3]
        )

    # Switching Preset updates the top system badge and cached mask
    master_mask_preset.change(
        switch_mask_preset_action,
        inputs=[master_mask_preset],
        outputs=[system_badge_display, tab1_status]
    )

    # Tab 1: Inspect frame with red overlay
    btn_inspect.click(
        update_visual_inspector,
        inputs=[tab1_video, tab1_time, master_mask_preset, master_engine, master_dilation, master_feather, master_padding],
        outputs=[img_orig_corner, img_overlay_corner, img_clean_corner, img_mask_binary, img_full_frame, tab1_status]
    )
    
    # Tab 1: Auto-detect
    btn_detect.click(
        auto_detect_mask_action,
        inputs=[tab1_video, gr.Number(value=25, visible=False), gr.Number(value=150, visible=False)],
        outputs=[img_mask_binary, tab1_status]
    )
    
    # Tab 1: Save mask
    btn_save_mask.click(
        save_active_mask_action,
        outputs=[tab1_status]
    )

    # Tab 2: Test video render
    btn_test_video.click(
        run_test_video_action,
        inputs=[
            tab2_video,
            tab2_start,
            tab2_dur,
            master_mask_preset,
            master_engine,
            master_nvenc,
            master_dilation,
            master_feather,
            master_padding,
            master_temporal,
            master_chunk_size
        ],
        outputs=[test_video_player, test_video_log]
    )

    # Tab 3: Batch folder run
    btn_run_batch.click(
        run_batch_ui_action,
        inputs=[
            batch_input,
            batch_output,
            master_mask_preset,
            master_engine,
            master_nvenc,
            master_dilation,
            master_feather,
            master_padding,
            master_temporal,
            master_chunk_size
        ],
        outputs=[batch_report_md]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)
