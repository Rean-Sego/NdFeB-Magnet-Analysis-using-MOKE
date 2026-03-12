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
THRESHOLD_OFFSET = 0        # bias threshold (can be negative or positive). I Want to change this to threshold at half of total pixels

MASK_ERODE_PX = 0           # 0 = off  Edges of image may be weird, so cutting the mask could remove anomalies

RASTER_DIRECTIONS = ["horizontal", "vertical", "diag_down", "diag_up"]
RASTER_SPACING_PX = 1
RASTER_LINE_WIDTH_PX = 1
RASTER_LEN_PX_RATIO = 41.56

RASTER_SAVE_OVERLAY = True
RASTER_SAVE_COORDS = True

    #===========================================
    # Definitions Creation
    #===========================================
def build_raster_mask(mask_u8, directions, spacing=1, line_width=1):
    h, w = mask_u8.shape
    raster = np.zeros((h, w), dtype=np.uint8)

    if "horizontal" in directions:
        for y in range(0, h, spacing):
            cv2.line(raster, (0, y), (w - 1, y), 255, line_width)

    if "vertical" in directions:
        for x in range(0, w, spacing):
            cv2.line(raster, (x, 0), (x, h - 1), 255, line_width)

    if "diag_down" in directions:
        for x0 in range(0, w, spacing):
            travel = min(w - 1 - x0, h - 1)
            cv2.line(raster, (x0, 0), (x0 + travel, travel), 255, line_width)

        for y0 in range(spacing, h, spacing):
            travel = min(w - 1, h - 1 - y0)
            cv2.line(raster, (0, y0), (travel, y0 + travel), 255, line_width)

    if "diag_up" in directions:
        for x0 in range(0, w, spacing):
            travel = min(w - 1 - x0, h - 1)
            cv2.line(raster, (x0, h - 1), (x0 + travel, h - 1 - travel), 255, line_width)

        for y0 in range(0, h - spacing, spacing):
            travel = min(w - 1, y0)
            cv2.line(raster, (0, y0), (travel, y0 - travel), 255, line_width)

    raster = cv2.bitwise_and(raster, mask_u8)
    return raster


def save_raster_coords(path, raster_binary):
    ys, xs = np.where(raster_binary > 0)
    coords = np.column_stack((xs, ys))
    np.savetxt(path, coords, fmt="%d", delimiter=",", header="x,y", comments="")

    # ==========================================
    # Selecting Images
    # ==========================================

os.makedirs(RAST_FOLDER, exist_ok=True)

grain_files = [f for f in os.listdir(SC_FOLDER) if f.endswith(".png") and "_mask" not in f]
grain_domain_widths = [] #Used for total domain thickness across all domains

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
    # RASTERIZATION
    # ==========================================

    mask_u8 = (mask_bool.astype(np.uint8) * 255)

    raster_mask = build_raster_mask(
        mask_u8,
        directions=RASTER_DIRECTIONS,
        spacing=RASTER_SPACING_PX,
        line_width=RASTER_LINE_WIDTH_PX
    )

    raster_binary = cv2.bitwise_and(binary, raster_mask)

    # ------------------------------------------
    # DOMAIN THICKNESS ANALYSIS
    # ------------------------------------------

    ys, xs = np.where(raster_binary > 0)

    coords = []
    transition_count = 0
    line_length_px = 0

    for y, x in zip(ys, xs):
        coords.append((x, y))

    # count transitions along raster mask. This block converts the individual lines and transitions into one big one.
    for y in range(raster_binary.shape[0]):

        line = raster_binary[y, :]
        mask_line = raster_mask[y, :] > 0

        if np.sum(mask_line) < 2:
            continue

        values = line[mask_line]

        transitions = np.sum(values[1:] != values[:-1])

        transition_count += transitions
        line_length_px += len(values)

    # avoid divide by zero
    if transition_count > 0:
        domain_width_px = line_length_px / transition_count
    else:
        domain_width_px = 0

    domain_width_len = domain_width_px/RASTER_LEN_PX_RATIO
    grain_domain_widths.append((base, domain_width_len))

    # ------------------------------------------
    # SAVE RASTER IMAGE
    # ------------------------------------------

    raster_name = base + "_raster.png"
    cv2.imwrite(os.path.join(RAST_FOLDER, raster_name), raster_binary)

    # ------------------------------------------
    # SAVE OVERLAY (optional)
    # ------------------------------------------

    if RASTER_SAVE_OVERLAY:

        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        overlay[(raster_mask > 0) & (mask_bool)] = (0, 255, 0)

        overlay_name = base + "_raster_overlay.png"
        cv2.imwrite(os.path.join(RAST_FOLDER, overlay_name), overlay)

    # ------------------------------------------
    # SAVE COORDS + ANALYSIS
    # ------------------------------------------

    coords_name = base + "_raster_coords.csv"
    csv_path = os.path.join(RAST_FOLDER, coords_name)

    with open(csv_path, "w") as f:

        f.write("x,y,analysis,value\n")

        max_len = max(len(coords), 3)

        for i in range(max_len):

            if i < len(coords):
                x, y = coords[i]
                coord_text = f"{x},{y}"
            else:
                coord_text = ","

            if i == 0:
                analysis_text = f"total_line_length_px,{line_length_px}"
            elif i == 1:
                analysis_text = f"total_transitions,{transition_count}"
            elif i == 2:
                analysis_text = f"domain_width_px,{domain_width_px}"
            elif i == 3:
                analysis_text = f"domain_width_um,{domain_width_len}"
            else:
                analysis_text = ","

            f.write(f"{coord_text},{analysis_text}\n")



    # ==========================================
    # QC PANEL (optional – comment out if needed)
    # ==========================================

    fig, ax = plt.subplots(1,4, figsize=(12,3))

    ax[0].imshow(gray, cmap='gray', vmin=0, vmax=255)
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


# ==========================================
# SAVE OVERALL DOMAIN AVERAGE
# ==========================================

summary_path = os.path.join(RAST_FOLDER, "overall_domain_summary.csv")

values = [v for _, v in grain_domain_widths if v > 0]

if len(values) > 0:
    avg_domain_width_um = np.mean(values)
else:
    avg_domain_width_um = 0

with open(summary_path, "w") as f:

    f.write("grain_file,domain_width_um\n")

    for name, value in grain_domain_widths:
        f.write(f"{name},{value}\n")

    f.write(f"\nAVERAGE,{avg_domain_width_um}\n")