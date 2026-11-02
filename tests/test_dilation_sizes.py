import cv2
import numpy as np

# Let's test different dilation sizes on mask_temporal_min
mask_tight = cv2.imread(r"temp_inspect\mask_temporal_min.png", cv2.IMREAD_GRAYSCALE)
img = cv2.imread(r"temp_inspect\br_6s.png")

for k in [5, 9, 13, 17]:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated = cv2.dilate(mask_tight, kernel, iterations=1)
    
    mask_patch = np.zeros((180, 350), dtype=np.uint8)
    mask_patch[60:170, 50:340] = dilated
    
    inpaint_ns = cv2.inpaint(img, mask_patch, inpaintRadius=k, flags=cv2.INPAINT_NS)
    inpaint_telea = cv2.inpaint(img, mask_patch, inpaintRadius=k, flags=cv2.INPAINT_TELEA)
    
    # Save results
    cv2.imwrite(f"temp_inspect\\test_k{k}_ns.png", inpaint_ns)
    cv2.imwrite(f"temp_inspect\\test_k{k}_telea.png", inpaint_telea)
    print(f"k={k} done")
