import cv2
import numpy as np

# Load the cropped mask
mask_dilated = cv2.imread(r"temp_inspect\mask_dilated.png", cv2.IMREAD_GRAYSCALE)
h_roi, w_roi = mask_dilated.shape

# Create full frame mask (1280x720)
mask_full = np.zeros((720, 1280), dtype=np.uint8)
h_start, h_end = 720 - 120, 720 - 10
w_start, w_end = 1280 - 300, 1280 - 10
mask_full[h_start:h_end, w_start:w_end] = mask_dilated
cv2.imwrite("mask.png", mask_full)
print("Saved full mask.png")

# Now let's test inpainting on the bottom right of br_2s (pavement) and br_6s (cat tail)
for name in ["br_2s", "br_6s", "br_11s"]:
    img = cv2.imread(f"temp_inspect\\{name}.png")
    # Note that br_*.png was cropped with 350x180:
    # in full frame: x: 1280-350=930 to 1280, y: 720-180=540 to 720
    # Our text ROI inside this 350x180 crop is:
    # y: (720-120) - 540 = 60 to (720-10) - 540 = 170
    # x: (1280-300) - 930 = 50 to (1280-10) - 930 = 340
    # Let's align exact coordinates
    mask_patch = np.zeros((180, 350), dtype=np.uint8)
    mask_patch[60:170, 50:340] = mask_dilated
    
    # 1. OpenCV Navier-Stokes
    inpainted_ns = cv2.inpaint(img, mask_patch, inpaintRadius=4, flags=cv2.INPAINT_NS)
    
    # 2. Seamless alpha compositing
    # Feather mask slightly
    alpha = cv2.GaussianBlur(mask_patch.astype(np.float32) / 255.0, (5, 5), 1.5)
    alpha = np.expand_dims(alpha, axis=2)
    composite = (inpainted_ns.astype(np.float32) * alpha + img.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    
    # Save comparison side-by-side
    comparison = np.hstack([img, composite])
    cv2.imwrite(f"temp_inspect\\comp_{name}.png", comparison)
    print(f"Generated comparison for {name}")
