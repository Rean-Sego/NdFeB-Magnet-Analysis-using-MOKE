import os
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# ==========================================
# USER SETTINGS
# ==========================================

DATA_FOLDER = r"processed_data"
SC_FOLDER = r"sc_results"
RAST_FOLDER = r"rast_results"

PATCH_ROWS = 4
PATCH_COLS = 4

BLUR_SIGMA = 1.5

THRESHOLD_MODE = "median"   # valley, median, or peak_midpoint
THRESHOLD_OFFSET = 6
MASK_ERODE_PX = 0

RASTER_DIRECTIONS = ["horizontal", "vertical", "diag_down", "diag_up"]
RASTER_SPACING_PX = 1
RASTER_LINE_WIDTH_PX = 1
RASTER_PX_LEN_RATIO = 41.56   # um / px

RASTER_SAVE_OVERLAY = True
RASTER_SAVE_COORDS = True

# ==========================================
# HELPERS
# ==========================================

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


def patch_sort_key(grain, patch_h, patch_w):
    ys, xs = np.where(grain["mask"] > 0)
    cy = np.mean(ys)
    cx = np.mean(xs)

    patch_row = int(cy // patch_h)
    patch_col = int(cx // patch_w)

    patch_row = min(patch_row, PATCH_ROWS - 1)
    patch_col = min(patch_col, PATCH_COLS - 1)

    return (patch_row, patch_col, cx)


def compute_threshold(valid_pixels):                                        ##Right now, I have a few options. When we find what works, we can delete the others
    hist, _ = np.histogram(valid_pixels, bins=256, range=(0, 255))

    peak1 = np.argmax(hist[:128])
    peak2 = np.argmax(hist[128:]) + 128

    if THRESHOLD_MODE == "valley":
        lo = min(peak1, peak2)
        hi = max(peak1, peak2)

        valley_region = hist[lo:hi]
        if len(valley_region) == 0:
            threshold = int(np.mean(valid_pixels))
        else:
            valley_index = np.argmin(valley_region)
            threshold = lo + valley_index

    elif THRESHOLD_MODE == "median":
        threshold = int(np.median(valid_pixels))

    elif THRESHOLD_MODE == "peak_midpoint":
        threshold = int((peak1 + peak2) / 2)

    else:
        threshold, _ = cv2.threshold(valid_pixels.astype(np.uint8), 0, 255, cv2.THRESH_OTSU)

    threshold = int(threshold + THRESHOLD_OFFSET)
    threshold = max(0, min(255, threshold))
    return threshold

def measure_direction(binary, mask_u8_small, direction, spacing, line_width):
    direction_mask = build_raster_mask(
        mask_u8_small,
        directions=[direction],
        spacing=spacing,
        line_width=line_width
    )

    direction_binary = cv2.bitwise_and(binary, direction_mask)

    transition_count = 0
    sampled_px = 0

    if direction == "horizontal":
        for row in range(direction_binary.shape[0]):
            line = direction_binary[row, :]
            mask_line = direction_mask[row, :] > 0
            if np.sum(mask_line) < 2:
                continue
            values = line[mask_line]
            transition_count += np.sum(values[1:] != values[:-1])
            sampled_px += len(values)

    elif direction == "vertical":
        for col in range(direction_binary.shape[1]):
            line = direction_binary[:, col]
            mask_line = direction_mask[:, col] > 0
            if np.sum(mask_line) < 2:
                continue
            values = line[mask_line]
            transition_count += np.sum(values[1:] != values[:-1])
            sampled_px += len(values)

    elif direction == "diag_down":
        h, w = direction_binary.shape

        for x0 in range(0, w, spacing):
            travel = min(w - 1 - x0, h - 1)
            if travel < 1:
                continue
            xs = np.arange(x0, x0 + travel + 1)
            ys = np.arange(0, travel + 1)
            values = direction_binary[ys, xs]
            transition_count += np.sum(values[1:] != values[:-1])
            sampled_px += len(values)

        for y0 in range(spacing, h, spacing):
            travel = min(w - 1, h - 1 - y0)
            if travel < 1:
                continue
            xs = np.arange(0, travel + 1)
            ys = np.arange(y0, y0 + travel + 1)
            values = direction_binary[ys, xs]
            transition_count += np.sum(values[1:] != values[:-1])
            sampled_px += len(values)

    elif direction == "diag_up":
        h, w = direction_binary.shape

        for x0 in range(0, w, spacing):
            travel = min(w - 1 - x0, h - 1)
            if travel < 1:
                continue
            xs = np.arange(x0, x0 + travel + 1)
            ys = np.arange(h - 1, h - 1 - travel - 1, -1)
            values = direction_binary[ys, xs]
            transition_count += np.sum(values[1:] != values[:-1])
            sampled_px += len(values)

        for y0 in range(0, h - spacing, spacing):
            travel = min(w - 1, y0)
            if travel < 1:
                continue
            xs = np.arange(0, travel + 1)
            ys = np.arange(y0, y0 - travel - 1, -1)
            values = direction_binary[ys, xs]
            transition_count += np.sum(values[1:] != values[:-1])
            sampled_px += len(values)

    return direction_mask, direction_binary, sampled_px, transition_count

# ==========================================
# FOLDERS
# ==========================================

os.makedirs(SC_FOLDER, exist_ok=True)
os.makedirs(RAST_FOLDER, exist_ok=True)

# ==========================================
# FIND PARENT SAMPLES
# ==========================================

npy_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith("_struct.npy")]
grain_domain_widths = []

for npy_file in npy_files:
    sample_base = npy_file.replace("_struct.npy", "")

    npy_path = os.path.join(DATA_FOLDER, npy_file)
    struct_img_path = os.path.join(DATA_FOLDER, f"{sample_base}_struct.png")
    polar_img_path = os.path.join(DATA_FOLDER, f"{sample_base}_polar.png")

    if not os.path.exists(struct_img_path):
        print(f"Skipping {sample_base}: missing struct image")
        continue

    if not os.path.exists(polar_img_path):
        print(f"Skipping {sample_base}: missing polar image")
        continue

    print(f"Processing {sample_base}...")

    data = np.load(npy_path, allow_pickle=True)
    struct_img = cv2.imread(struct_img_path)
    polar_img = cv2.imread(polar_img_path)

    if struct_img is None or polar_img is None:
        print(f"Skipping {sample_base}: could not read images")
        continue

    if len(data) == 0:
        print(f"Skipping {sample_base}: no grains")
        continue

    # ==========================================
    # 4x4 PATCH SORTING
    # ==========================================

    h, w = polar_img.shape[:2]
    patch_w = w / PATCH_COLS
    patch_h = h / PATCH_ROWS

    data = sorted(data, key=lambda g: patch_sort_key(g, patch_h, patch_w))

    # ==========================================
    # PATCH QC IMAGE
    # ==========================================

    qc_img = polar_img.copy()
    line_color = (255, 255, 255)
    line_thickness = 2

    for i in range(1, PATCH_COLS):
        x = int(i * patch_w)
        cv2.line(qc_img, (x, 0), (x, h), line_color, line_thickness)

    for i in range(1, PATCH_ROWS):
        y = int(i * patch_h)
        cv2.line(qc_img, (0, y), (w, y), line_color, line_thickness)

    qc_filename = f"{sample_base}_patch_qc.png"
    cv2.imwrite(os.path.join(SC_FOLDER, qc_filename), qc_img)

    # ==========================================
    # GRAIN LOOP
    # ==========================================

    for i, g in enumerate(data):
        mask_u8 = (g["mask"].astype(np.uint8) * 255)

        coords = cv2.findNonZero(mask_u8)
        if coords is None:
            continue

        x, y, bw, bh = cv2.boundingRect(coords)

        # --------------------------------------
        # SAVE SINGLE GRAIN CUTOUT + MASK
        # --------------------------------------

        grain_cutout = np.zeros((bh, bw, 3), dtype=np.uint8)
        cropped_img = polar_img[y:y+bh, x:x+bw]
        cropped_mask = mask_u8[y:y+bh, x:x+bw]

        grain_cutout[cropped_mask > 0] = cropped_img[cropped_mask > 0]

        grain_file_base = f"{sample_base}_grain_{i:04d}"
        grain_filename = f"{grain_file_base}.png"
        mask_filename = f"{grain_file_base}_mask.png"

        cv2.imwrite(os.path.join(SC_FOLDER, grain_filename), grain_cutout)
        cv2.imwrite(os.path.join(SC_FOLDER, mask_filename), cropped_mask)

        # --------------------------------------
        # PREP FOR THICKNESS / RASTER
        # --------------------------------------

        grain = grain_cutout.copy()
        mask = cropped_mask.copy()

        gray = cv2.cvtColor(grain, cv2.COLOR_BGR2GRAY)

        if MASK_ERODE_PX > 0:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=MASK_ERODE_PX)

        mask_bool = mask > 0
        if np.count_nonzero(mask_bool) == 0:
            continue

        # --------------------------------------
        # CONTRAST
        # --------------------------------------

        pixels = gray[mask_bool]
        p_low = np.percentile(pixels, 2)
        p_high = np.percentile(pixels, 98)

        contrast = np.clip((gray - p_low) * 255 / (p_high - p_low + 1e-6), 0, 255).astype(np.uint8)

        # --------------------------------------
        # GAUSSIAN BLUR
        # --------------------------------------

        blurred = gaussian_filter(contrast, sigma=BLUR_SIGMA)

        # --------------------------------------
        # THRESHOLD
        # --------------------------------------

        valid_pixels = blurred[mask_bool]

        original_pixels = gray[mask_bool]
        contrast_pixels = contrast[mask_bool]
        blur_pixels = blurred[mask_bool]

        threshold = compute_threshold(valid_pixels)

        binary = np.zeros_like(gray, dtype=np.uint8)
        binary[(blurred > threshold) & mask_bool] = 255

        binary_name = f"{grain_file_base}_binary.png"
        cv2.imwrite(os.path.join(RAST_FOLDER, binary_name), binary)

        # --------------------------------------
        # RASTERIZATION
        # --------------------------------------

        mask_u8_small = (mask_bool.astype(np.uint8) * 255)

        raster_mask = np.zeros_like(mask_u8_small)
        raster_binary = np.zeros_like(binary)

        total_hv_px = 0
        total_diag_px = 0
        total_transitions = 0

        for direction in RASTER_DIRECTIONS:
            dir_mask, dir_binary, sampled_px, transitions = measure_direction(
                binary,
                mask_u8_small,
                direction,
                RASTER_SPACING_PX,
                RASTER_LINE_WIDTH_PX
            )

            raster_mask = cv2.bitwise_or(raster_mask, dir_mask)
            raster_binary = cv2.bitwise_or(raster_binary, dir_binary)

            total_transitions += transitions

            if direction in ["horizontal", "vertical"]:
                total_hv_px += sampled_px
            else:
                total_diag_px += sampled_px

        raster_binary = cv2.bitwise_and(binary, raster_mask)

        # --------------------------------------
        # DOMAIN THICKNESS ANALYSIS
        # --------------------------------------

        ys, xs = np.where(raster_binary > 0)
        coords_list = [(xv, yv) for yv, xv in zip(ys, xs)]

        total_length_um = (
            total_hv_px / RASTER_PX_LEN_RATIO +
            (total_diag_px * np.sqrt(2)) / RASTER_PX_LEN_RATIO
        )

        if total_transitions > 0:
            domain_width_um = total_length_um / total_transitions
            domain_width_px = domain_width_um * RASTER_PX_LEN_RATIO
        else:
            domain_width_um = 0
            domain_width_px = 0

        if domain_width_um > 0:
            grain_domain_widths.append((grain_file_base, domain_width_um))
            
        # --------------------------------------
        # SAVE RASTER IMAGE
        # --------------------------------------

        raster_name = f"{grain_file_base}_raster.png"
        cv2.imwrite(os.path.join(RAST_FOLDER, raster_name), raster_binary)

        # --------------------------------------
        # SAVE OVERLAY
        # --------------------------------------

        if RASTER_SAVE_OVERLAY:
            overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            overlay[(raster_mask > 0) & mask_bool] = (0, 255, 0)

            overlay_name = f"{grain_file_base}_raster_overlay.png"
            cv2.imwrite(os.path.join(RAST_FOLDER, overlay_name), overlay)

        # --------------------------------------
        # SAVE COORDS + ANALYSIS CSV
        # --------------------------------------

        if RASTER_SAVE_COORDS:
            csv_name = f"{grain_file_base}_raster_coords.csv"
            csv_path = os.path.join(RAST_FOLDER, csv_name)

            with open(csv_path, "w") as f:
                f.write("x,y,analysis,value\n")

                max_len = max(len(coords_list), 5)

                for j in range(max_len):
                    if j < len(coords_list):
                        cx, cy = coords_list[j]
                        coord_text = f"{cx},{cy}"
                    else:
                        coord_text = ","

                    if j == 0:
                        analysis_text = f"total_hv_px,{total_hv_px}"
                    elif j == 1:
                        analysis_text = f"total_diag_px,{total_diag_px}"
                    elif j == 2:
                        analysis_text = f"total_transitions,{total_transitions}"
                    elif j == 3:
                        analysis_text = f"domain_width_px,{domain_width_px}"
                    elif j == 4:
                        analysis_text = f"domain_width_um,{domain_width_um}"
                    else:
                        analysis_text = ","

                    f.write(f"{coord_text},{analysis_text}\n")

        # --------------------------------------
        # QC PANEL
        # --------------------------------------

        fig, ax = plt.subplots(1, 4, figsize=(12, 3))

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

        qc_name = f"{grain_file_base}_qc.png"
        plt.tight_layout()
        plt.savefig(os.path.join(RAST_FOLDER, qc_name))
        plt.close()


        # --------------------------------------
        # HISTOGRAM QC PANEL
        # --------------------------------------

        fig, ax = plt.subplots(1, 3, figsize=(12, 3))

        ax[0].hist(original_pixels, bins=256, range=(0, 255))
        ax[0].set_title("original hist")

        ax[1].hist(contrast_pixels, bins=256, range=(0, 255))
        ax[1].set_title("contrast hist")

        ax[2].hist(blur_pixels, bins=256, range=(0, 255))
        ax[2].set_title("blur hist")

        hist_qc_name = f"{grain_file_base}_hist_qc.png"
        plt.tight_layout()
        plt.savefig(os.path.join(RAST_FOLDER, hist_qc_name))
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