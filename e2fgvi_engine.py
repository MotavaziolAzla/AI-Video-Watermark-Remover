import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import List

BASE_DIR = Path(__file__).resolve().parent
E2FGVI_DIR = BASE_DIR / "models" / "E2FGVI"

# Ensure E2FGVI is loaded in its own namespace
if str(E2FGVI_DIR) not in sys.path:
    sys.path.insert(0, str(E2FGVI_DIR))

from e2fgvi_model.e2fgvi_hq import InpaintGenerator

class E2FGVIVideoInpainter:
    """
    Video-level temporal inpainter using E2FGVI-HQ:
    End-to-End Flow-Guided Video Inpainting with multi-scale feature propagation.
    """
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_path = E2FGVI_DIR / "release_model" / "E2FGVI-HQ-CVPR22.pth"
        self.model = None
        self._load_model()

    def _load_model(self):
        print(f"[E2FGVI-HQ] Initializing on {self.device}...")
        self.model = InpaintGenerator().to(self.device)
        state_dict = torch.load(str(self.ckpt_path), map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("[E2FGVI-HQ] Model loaded successfully!")

    @torch.no_grad()
    def inpaint_sequence(self, rois_bgr: List[np.ndarray], mask_gray: np.ndarray) -> List[np.ndarray]:
        """
        Takes a sequence of BGR ROI frames and a 2D binary mask.
        Returns the sequence of inpainted BGR ROI frames.
        """
        if not rois_bgr:
            return []
            
        T = len(rois_bgr)
        H, W = rois_bgr[0].shape[:2]
        
        # Format frames: (1, T, 3, H, W) normalized to [-1, 1]
        frames_rgb = [np.ascontiguousarray(f[:, :, ::-1]) for f in rois_bgr]
        tensors = [torch.from_numpy(f.transpose(2, 0, 1)).float() / 127.5 - 1.0 for f in frames_rgb]
        imgs = torch.stack(tensors, dim=0).unsqueeze(0).to(self.device)
        
        # Mask: (1, T, 1, H, W) binary [0, 1]
        m_tensor = (torch.from_numpy(np.ascontiguousarray(mask_gray)).float() / 255.0).unsqueeze(0).unsqueeze(0)
        masks = m_tensor.repeat(1, T, 1, 1, 1).to(self.device)
        
        # E2FGVI expects masked image: imgs * (1 - masks)
        masked_imgs = imgs * (1.0 - masks)
        
        # Window size multiples: mod_h=60, mod_w=108 (matches focal transformer window requirements)
        mod_h = 60
        mod_w = 108
        pad_h = (mod_h - H % mod_h) % mod_h
        pad_w = (mod_w - W % mod_w) % mod_w
        
        if pad_h > 0:
            masked_imgs = torch.cat([masked_imgs, torch.flip(masked_imgs, [3])], 3)[:, :, :, :H + pad_h, :]
        if pad_w > 0:
            masked_imgs = torch.cat([masked_imgs, torch.flip(masked_imgs, [4])], 4)[:, :, :, :, :W + pad_w]
            
        # Run forward pass (num_local_frames = T)
        pred_imgs, _ = self.model(masked_imgs, T)
        
        # Reshape to (1, T, 3, H_padded, W_padded)
        pred_imgs = pred_imgs.view(1, T, 3, H + pad_h, W + pad_w)
        
        # Crop back to original (H, W)
        if pad_h > 0 or pad_w > 0:
            pred_imgs = pred_imgs[:, :, :, :H, :W]
            
        # Composite completed areas: masks * pred + (1 - masks) * imgs
        comp_imgs = masks * pred_imgs + (1.0 - masks) * imgs
        
        # Convert back to list of BGR uint8
        comp_imgs = (comp_imgs.squeeze(0).clamp(-1.0, 1.0) + 1.0) * 127.5
        out_rgb = comp_imgs.permute(0, 2, 3, 1).byte().cpu().numpy()
        return [np.ascontiguousarray(f[:, :, ::-1]) for f in out_rgb]
