import os
import cv2
import numpy as np
from PIL import Image
from typing import Optional, Callable, List
import torch

class OpenCVInpainter:
    """Fast, lightweight CPU inpainter using OpenCV Navier-Stokes or Telea."""
    def __init__(self, method: str = "ns", radius: int = 4):
        self.method = method.lower()
        self.radius = radius
        self.is_temporal = False

    def __call__(self, roi: np.ndarray, mask: np.ndarray) -> np.ndarray:
        flag = cv2.INPAINT_NS if self.method == "ns" else cv2.INPAINT_TELEA
        return cv2.inpaint(roi, mask, inpaintRadius=self.radius, flags=flag)

    def inpaint_sequence(self, rois: List[np.ndarray], mask: np.ndarray) -> List[np.ndarray]:
        flag = cv2.INPAINT_NS if self.method == "ns" else cv2.INPAINT_TELEA
        return [cv2.inpaint(r, mask, inpaintRadius=self.radius, flags=flag) for r in rois]


class LamaInpainter:
    """Deep learning AI inpainter using LaMa (Large Mask Inpainting) on GPU."""
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.is_temporal = False
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            from config import BASE_DIR
            local_lama = BASE_DIR / "checkpoints" / "big-lama.pt"
            if local_lama.exists():
                os.environ["LAMA_MODEL"] = str(local_lama)
            from simple_lama_inpainting import SimpleLama
            print(f"[Engine] Loading LaMa model on {self.device}...")
            self.model = SimpleLama(device=self.device)
            print("[Engine] LaMa model loaded successfully.")
        except Exception as e:
            print(f"[Engine] Warning: Could not load SimpleLama: {e}")
            self.model = None

    def __call__(self, roi: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self.model is None:
            return cv2.inpaint(roi, mask, inpaintRadius=4, flags=cv2.INPAINT_NS)
            
        try:
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(roi_rgb)
            mask_pil = Image.fromarray(mask)
            
            result_pil = self.model(img_pil, mask_pil)
            result_rgb = np.array(result_pil)
            return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[Engine] LaMa inference error: {e}. Falling back to OpenCV NS.")
            return cv2.inpaint(roi, mask, inpaintRadius=4, flags=cv2.INPAINT_NS)

    def inpaint_sequence(self, rois: List[np.ndarray], mask: np.ndarray) -> List[np.ndarray]:
        return [self(r, mask) for r in rois]


class ProPainterInpainterWrapper:
    """Wrapper around ProPainter video inpainter exposing both sequence and single-frame APIs."""
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.is_temporal = True
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            from propainter_engine import ProPainterVideoInpainter
            self._engine = ProPainterVideoInpainter(device=self.device)
        return self._engine

    def inpaint_sequence(self, rois: List[np.ndarray], mask: np.ndarray) -> List[np.ndarray]:
        if not rois:
            return []
        engine = self._get_engine()
        if len(rois) == 1:
            res = engine.inpaint_sequence([rois[0], rois[0]], mask)
            return [res[0]]
        return engine.inpaint_sequence(rois, mask)

    def __call__(self, roi: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.inpaint_sequence([roi], mask)[0]


class HybridInpainterWrapper:
    """
    SOTA Hybrid Inpainter:
    Fuses AI LaMa's rock-solid frame tracking and boundary anchoring with
    ProPainter's intelligent spatio-temporal structure and detail completion.
    
    1. Computes LaMa prior: removes watermark to anchor background motion without stationary text drag.
    2. RAFT computes pristine optical flow on the clean background.
    3. Bilinear flow propagation warps ground-truth details across time without discrete pixel hopping.
    4. ProPainter spatio-temporal transformer reconstructs fine geometric structures.
    5. Multi-scale frequency fusion anchors the low-frequency ambient illumination and camera cadence
       to LaMa while adopting ProPainter's intelligent high-frequency structural details.
    """
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.is_temporal = True
        self._lama = None
        self._propainter = None

    def _get_lama(self):
        if self._lama is None:
            self._lama = LamaInpainter(device=self.device)
        return self._lama

    def _get_propainter(self):
        if self._propainter is None:
            from propainter_engine import ProPainterVideoInpainter
            self._propainter = ProPainterVideoInpainter(device=self.device)
        return self._propainter

    def inpaint_sequence(self, rois: List[np.ndarray], mask: np.ndarray) -> List[np.ndarray]:
        if not rois:
            return []
        if len(rois) == 1:
            return self._get_lama().inpaint_sequence(rois, mask)
            
        lama = self._get_lama()
        pp = self._get_propainter()
        
        # 1. Generate clean LaMa base prior
        lama_rois = lama.inpaint_sequence(rois, mask)
        
        # 2. Run ProPainter with clean LaMa-prior flow guidance
        pp_rois = pp.inpaint_sequence(rois, mask, flow_guidance_rois=lama_rois)
        
        # 3. Speed-adaptive multi-scale frequency fusion
        fused_rois = []
        for i in range(len(rois)):
            l_r = lama_rois[i]
            p_r = pp_rois[i]
            
            # Local inter-frame motion speed
            if i > 0:
                speed = float(np.mean(np.abs(rois[i].astype(float) - rois[i-1].astype(float))))
            elif len(rois) > 1:
                speed = float(np.mean(np.abs(rois[1].astype(float) - rois[0].astype(float))))
            else:
                speed = 0.0
                
            # If motion is slow/static (speed < 2.0px): 100% LaMa low-frequency base anchors lighting & eliminates flicker
            # If camera is moving fast (> 6.0px, like upward tilt in imdl5): ProPainter's flow-aligned base takes over
            w_lama = max(0.20, min(1.0, 1.0 - (speed - 2.0) / 6.0))
            
            l_low = cv2.GaussianBlur(l_r.astype(np.float32), (15, 15), 0)
            p_low = cv2.GaussianBlur(p_r.astype(np.float32), (15, 15), 0)
            p_high = p_r.astype(np.float32) - p_low
            
            fused_low = l_low * w_lama + p_low * (1.0 - w_lama)
            fused = np.clip(fused_low + p_high, 0, 255).astype(np.uint8)
            fused_rois.append(fused)
            
        return fused_rois

    def __call__(self, roi: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.inpaint_sequence([roi], mask)[0]


class E2FGVIInpainterWrapper:
    """Wrapper around E2FGVI-HQ video inpainter exposing both sequence and single-frame APIs."""
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.is_temporal = True
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            from e2fgvi_engine import E2FGVIVideoInpainter
            self._engine = E2FGVIVideoInpainter(device=self.device)
        return self._engine

    def inpaint_sequence(self, rois: List[np.ndarray], mask: np.ndarray) -> List[np.ndarray]:
        if not rois:
            return []
        engine = self._get_engine()
        if len(rois) == 1:
            res = engine.inpaint_sequence([rois[0], rois[0]], mask)
            return [res[0]]
        return engine.inpaint_sequence(rois, mask)

    def __call__(self, roi: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.inpaint_sequence([roi], mask)[0]


# Global cached singletons to avoid re-allocating model weights
_CACHED_ENGINES = {}

def get_inpainter(engine_name: str, radius: int = 4):
    """
    Factory function returning a unified inpainter instance.
    Supports:
      - 'hybrid'      -> Hybrid (ProPainter Structure + AI LaMa Tracking Stability) [RECOMMENDED]
      - 'propainter'  -> ProPainter Video SOTA Optical Flow Inpainter
      - 'lama' / 'ai' -> LaMa PyTorch GPU Inpainter
      - 'e2fgvi'      -> E2FGVI-HQ High-Quality Video Inpainter
      - 'ns'          -> OpenCV Navier-Stokes (CPU)
      - 'telea'       -> OpenCV Telea (CPU)
    """
    global _CACHED_ENGINES
    name = engine_name.lower()

    if "hybrid" in name:
        if "hybrid" not in _CACHED_ENGINES:
            _CACHED_ENGINES["hybrid"] = HybridInpainterWrapper()
        return _CACHED_ENGINES["hybrid"]
    elif "propainter" in name:
        if "propainter" not in _CACHED_ENGINES:
            _CACHED_ENGINES["propainter"] = ProPainterInpainterWrapper()
        return _CACHED_ENGINES["propainter"]
    elif "e2fgvi" in name or "e2" in name:
        if "e2fgvi" not in _CACHED_ENGINES:
            _CACHED_ENGINES["e2fgvi"] = E2FGVIInpainterWrapper()
        return _CACHED_ENGINES["e2fgvi"]
    elif "lama" in name or "ai lama" in name or "ai" in name:
        if "lama" not in _CACHED_ENGINES:
            _CACHED_ENGINES["lama"] = LamaInpainter()
        return _CACHED_ENGINES["lama"]
    elif "telea" in name:
        return OpenCVInpainter(method="telea", radius=radius)
    else:
        return OpenCVInpainter(method="ns", radius=radius)
