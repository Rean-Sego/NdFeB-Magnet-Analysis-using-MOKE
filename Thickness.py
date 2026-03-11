import os
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# ==========================================
# USER SETTINGS
# ==========================================

SC_FOLDER = r"C:\Users\lavab\Desktop\Stuff\UCF\Research\MOKE Program\MOKE_REPO\sc_results"
RAST_FOLDER = r"C:\Users\lavab\Desktop\Stuff\UCF\Research\MOKE Program\MOKE_REPO\rast_results"

BLUR_SIGMA = 1.5

THRESHOLD_MODE = "valley"   # "valley" or "otsu"
THRESHOLD_OFFSET = 0        # bias threshold (can be negative or positive)

MASK_ERODE_PX = 0           # 0 = off

# ==========================================

os.makedirs(RAST_FOLDER, exist_ok=True)

grain_files = [f for f in os.listdir(SC_FOLDER) if f.endswith(".png") and "_mask" not in f]

for grain_file in grain_files:

    base = grain_file.replace(".png", "")
    mask_file = base + "_mask.png"

    grain_path = os.path.join(SC_FOLDER, grain_file)
    mask_path = os.path.join(SC_FOLDER, mask_file)

    if not os.path.exists(mask_path):
        continue

    grain = cv2.imread(grain_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    gray = cv2.cvtColor(grain, cv2.COLOR_BGR2GRAY)

    # ==========================================
    # OPTIONAL MASK EROSION
    # ==========================================

    if MASK_ERODE_PX > 0:
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=MASK_ERODE_PX)

    mask_bool = mask > 0

    # ==========================================
    # CONTRAST ENHANCEMENT (inside mask only)
    # ==========================================

    pixels = gray[mask_bool]

    p_low = np.percentile(pixels, 2)
    p_high = np.percentile(pixels, 98)

    contrast = np.clip((gray - p_low) * 255 / (p_high - p_low + 1e-6), 0, 255).astype(np.uint8)

    # ==========================================
    # GAUSSIAN BLUR
    # ==========================================

    blurred = gaussian_filter(contrast, sigma=BLUR_SIGMA)

    # ==========================================
    # THRESHOLD SELECTION
    # ==========================================

    valid_pixels = blurred[mask_bool]

    if THRESHOLD_MODE == "valley":

        hist, bins = np.histogram(valid_pixels, bins=256, range=(0,255))

        peak1 = np.argmax(hist[:128])
        peak2 = np.argmax(hist[128:]) + 128

        valley_region = hist[min(peak1,peak2):max(peak1,peak2)]
        valley_index = np.argmin(valley_region)

        threshold = min(peak1,peak2) + valley_index

    else:
        threshold, _ = cv2.threshold(valid_pixels.astype(np.uint8), 0, 255, cv2.THRESH_OTSU)

    threshold = int(threshold + THRESHOLD_OFFSET)

    # ==========================================
    # APPLY THRESHOLD
    # ==========================================

    binary = np.zeros_like(gray, dtype=np.uint8)
    binary[(blurred > threshold) & mask_bool] = 255

    # ==========================================
    # SAVE RESULT
    # ==========================================

    binary_name = base + "_binary.png"
    cv2.imwrite(os.path.join(RAST_FOLDER, binary_name), binary)

    # ==========================================
    # QC PANEL (optional – comment out if needed)
    # ==========================================

    fig, ax = plt.subplots(1,4, figsize=(12,3))

    ax[0].imshow(gray, cmap='gray')
    ax[0].set_title("original")

    ax[1].imshow(contrast, cmap='gray')
    ax[1].set_title("contrast")

    ax[2].imshow(blurred, cmap='gray')
    ax[2].set_title("blur")

    ax[3].imshow(binary, cmap='gray')
    ax[3].set_title("binary")

    for a in ax:
        a.axis("off")

    qc_name = base + "_qc.png"
    plt.tight_layout()
    plt.savefig(os.path.join(RAST_FOLDER, qc_name))
    plt.close()