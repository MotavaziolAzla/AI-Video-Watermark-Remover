import torch
from simple_lama_inpainting import SimpleLama
from PIL import Image
import numpy as np

print("Testing SimpleLama...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
lama = SimpleLama(device=device)

# Test dummy inference on small 64x64 image
img = Image.new("RGB", (64, 64), color="blue")
mask = Image.new("L", (64, 64), color=0)
res = lama(img, mask)
print("SimpleLama test passed successfully! Output size:", res.size)
