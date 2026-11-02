import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional

BASE_DIR = Path(__file__).resolve().parent
PROPAINTER_DIR = BASE_DIR / "models" / "ProPainter"

# Ensure ProPainter is loaded in its own namespace
if str(PROPAINTER_DIR) not in sys.path:
    sys.path.insert(0, str(PROPAINTER_DIR))

from model.propainter import InpaintGenerator
from model.recurrent_flow_completion import RecurrentFlowCompleteNet
from model.modules.flow_comp_raft import RAFT_bi

class ProPainterVideoInpainter:
    """
    Video-level temporal inpainter using ProPainter:
    Dual-domain recurrent flow completion + sparse spatio-temporal attention.
    Enhanced with sub-pixel bilinear flow warping, minimum dimensional safety padding,
    and optional clean-prior flow guidance.
    """
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_dir = PROPAINTER_DIR / "weights"
        
        self.raft = None
        self.flow_comp = None
        self.model = None
        self._load_models()

    def _load_models(self):
        print(f"[ProPainter] Initializing on {self.device}...")
        raft_path = str(self.weights_dir / "raft-things.pth")
        flow_path = str(self.weights_dir / "recurrent_flow_completion.pth")
        model_path = str(self.weights_dir / "ProPainter.pth")
        
        self.raft = RAFT_bi(raft_path, device=self.device)
        
        self.flow_comp = RecurrentFlowCompleteNet(flow_path)
        for p in self.flow_comp.parameters():
            p.requires_grad = False
        self.flow_comp.to(self.device).eval()
        
        self.model = InpaintGenerator(model_path=model_path).to(self.device).eval()
        print("[ProPainter] All models loaded successfully!")

    @torch.no_grad()
    def inpaint_sequence(
        self,
        rois_bgr: List[np.ndarray],
        mask_gray: np.ndarray,
        flow_guidance_rois: Optional[List[np.ndarray]] = None
    ) -> List[np.ndarray]:
        """
        Takes a sequence of BGR ROI frames and a 2D binary mask.
        flow_guidance_rois: optional sequence of clean, watermark-free frames (e.g. from LaMa)
                            used to compute true background optical flow without stationary text drag.
        Returns the sequence of inpainted BGR ROI frames.
        """
        if not rois_bgr:
            return []
            
        T = len(rois_bgr)
        H, W = rois_bgr[0].shape[:2]
        
        # Ensure minimum dimension of at least 128 for 4-level optical flow pyramid
        min_dim = 128
        target_h = max(min_dim, H)
        target_w = max(min_dim, W)
        pad_h = (8 - target_h % 8) % 8 + (target_h - H)
        pad_w = (8 - target_w % 8) % 8 + (target_w - W)
        
        # Format frames: (1, T, 3, H, W) normalized to [-1, 1]
        frames_rgb = [np.ascontiguousarray(cv2_frame[:, :, ::-1]) for cv2_frame in rois_bgr]
        tensors = [torch.from_numpy(f.transpose(2, 0, 1)).float() / 127.5 - 1.0 for f in frames_rgb]
        frames = torch.stack(tensors, dim=0).unsqueeze(0).to(self.device)
        
        # Format masks: (1, T, 1, H, W) binary [0, 1]
        m_tensor = (torch.from_numpy(np.ascontiguousarray(mask_gray)).float() / 255.0).unsqueeze(0).unsqueeze(0)
        masks = m_tensor.repeat(1, T, 1, 1, 1).to(self.device)
        
        import torch.nn.functional as F
        def pad_5d(t5d, pw, ph):
            if pw == 0 and ph == 0:
                return t5d
            b_sz, t_sz, c_sz, h_sz, w_sz = t5d.shape
            t4d = t5d.view(b_sz * t_sz, c_sz, h_sz, w_sz)
            padded = F.pad(t4d, (0, pw, 0, ph), mode="reflect")
            return padded.view(b_sz, t_sz, c_sz, h_sz + ph, w_sz + pw)

        frames = pad_5d(frames, pad_w, pad_h)
        masks = pad_5d(masks, pad_w, pad_h)
            
        b, t, _, cur_h, cur_w = masks.size()
        
        # Prepare flow estimation frames: if clean guidance prior is provided, use it!
        if flow_guidance_rois is not None and len(flow_guidance_rois) == T:
            guidance_rgb = [np.ascontiguousarray(g[:, :, ::-1]) for g in flow_guidance_rois]
            g_tensors = [torch.from_numpy(g.transpose(2, 0, 1)).float() / 127.5 - 1.0 for g in guidance_rgb]
            flow_frames = torch.stack(g_tensors, dim=0).unsqueeze(0).to(self.device)
            flow_frames = pad_5d(flow_frames, pad_w, pad_h)
        else:
            flow_frames = frames
            
        # 1. Complete bidirectional flow using RAFT and Recurrent Flow Completion
        forward_flows, backward_flows = self.raft(flow_frames)
        gt_flows_bi = (forward_flows, backward_flows)
        pred_flows_bi, _ = self.flow_comp.forward_bidirect_flow(gt_flows_bi, masks)
        pred_flows_bi = self.flow_comp.combine_flow(gt_flows_bi, pred_flows_bi, masks)
        
        # 2. Image propagation along completed optical flows using smooth bilinear warping
        masked_frames = frames * (1.0 - masks)
        prop_imgs, updated_local_masks = self.model.img_propagation(masked_frames, pred_flows_bi, masks, 'bilinear')
        updated_frames = frames * (1.0 - masks) + prop_imgs.view(b, t, 3, cur_h, cur_w) * masks
        updated_masks = updated_local_masks.view(b, t, 1, cur_h, cur_w)
        
        # 3. Spatio-temporal sparse transformer inpainting
        t_dil = 2 if t > 2 else 1
        out_tensor = self.model(updated_frames, pred_flows_bi, masks, updated_masks, t, t_dilation=t_dil)
        out_tensor = out_tensor.view(b, t, 3, cur_h, cur_w)
        
        # Crop back if padded
        if pad_h > 0 or pad_w > 0:
            out_tensor = out_tensor[:, :, :, :H, :W]
            
        # Convert back to list of BGR uint8 arrays
        out_tensor = (out_tensor.squeeze(0).clamp(-1.0, 1.0) + 1.0) * 127.5
        out_frames_rgb = out_tensor.permute(0, 2, 3, 1).byte().cpu().numpy()
        
        out_bgr = [np.ascontiguousarray(f[:, :, ::-1]) for f in out_frames_rgb]
        return out_bgr
