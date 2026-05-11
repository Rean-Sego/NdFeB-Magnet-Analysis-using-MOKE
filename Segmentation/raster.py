import os
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# ==========================================
# USER SETTINGS
# ==========================================

OUTPUTS_FOLDER = r"outputs"

DATA_FOLDER = os.path.join(OUTPUTS_FOLDER, "images")
OBJECT_MASK_FOLDER = os.path.join(OUTPUTS_FOLDER, "object_mask")

SC_FOLDER = os.path.join(OUTPUTS_FOLDER, "sc_results")
RAST_FOLDER = os.path.join(OUTPUTS_FOLDER, "rast_results")

PATCH_ROWS = 4
PATCH_COLS = 4

BLUR_SIGMA = 1.5 

THRESHOLD_MODE = "median"   # valley, median, or peak_midpoint
THRESHOLD_OFFSET = 0
MASK_ERODE_PX = 0

RASTER_DIRECTIONS = ["horizontal", "vertical", "diag_down", "diag_up"]
RASTER_SPACING_PX = 1
RASTER_LINE_WIDTH_PX = 1
RASTER_PX_LEN_RATIO = 41.56   # px / um

RASTER_SAVE_OVERLAY = True
RASTER_SAVE_COORDS = True

DWCLOSURE = 0.57
DWSTRIPE = 2.16
ALIGNMENT_THRESHOLD = 95  # Percentage alignment threshold for "aligned" classification

 #---Individual line removal thresholds.
MIN_LINE_LENGTH_PX = 10
MIN_LINE_TRANSITIONS = 2

 #---Reject poorly thresholded domains
REJECT_EXTREME_BINARY = True
WHITE_REJECT_THRESHOLD = 0.6
BLACK_REJECT_THRESHOLD = 0.6
MIN_MAX_LINE_TRANSITIONS = 5  #Grain Removal; If no lines meet the threshold, remove the whole grain.


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
    max_line_transitions = 0

    if direction == "horizontal":
        for row in range(direction_binary.shape[0]):
            line = direction_binary[row, :]
            mask_line = direction_mask[row, :] > 0
            if np.sum(mask_line) < 2:
                continue
            values = line[mask_line]
            ##This line below takes two pixels next to each other and adds 1 to transition count if they are NOT equal. so, if a 0 and 255 are next to each other, its +1 to transition

            line_length_px = len(values)
            line_transitions = np.sum(values[1:] != values[:-1])

            if line_length_px < MIN_LINE_LENGTH_PX:
                continue
            if line_transitions < MIN_LINE_TRANSITIONS:
                continue

            transition_count += line_transitions
            max_line_transitions = max(max_line_transitions, line_transitions)
            sampled_px += line_length_px

    elif direction == "vertical":
        for col in range(direction_binary.shape[1]):
            line = direction_binary[:, col]
            mask_line = direction_mask[:, col] > 0
            if np.sum(mask_line) < 2:
                continue
            values = line[mask_line]
            ##This line below takes two pixels next to each other and adds 1 to transition count if they are NOT equal. so, if a 0 and 255 are next to each other, its +1 to transition

            line_length_px = len(values)
            line_transitions = np.sum(values[1:] != values[:-1])

            if line_length_px < MIN_LINE_LENGTH_PX:
                continue
            if line_transitions < MIN_LINE_TRANSITIONS:
                continue

            transition_count += line_transitions
            max_line_transitions = max(max_line_transitions, line_transitions)
            sampled_px += line_length_px

    elif direction == "diag_down":
        h, w = direction_binary.shape

        for x0 in range(0, w, spacing):
            travel = min(w - 1 - x0, h - 1)
            if travel < 1:
                continue
            xs = np.arange(x0, x0 + travel + 1)
            ys = np.arange(0, travel + 1)
            values = direction_binary[ys, xs]
            ##This line below takes two pixels next to each other and adds 1 to transition count if they are NOT equal. so, if a 0 and 255 are next to each other, its +1 to transition

            line_length_px = len(values)
            line_transitions = np.sum(values[1:] != values[:-1])

            if line_length_px < MIN_LINE_LENGTH_PX:
                continue
            if line_transitions < MIN_LINE_TRANSITIONS:
                continue

            transition_count += line_transitions
            max_line_transitions = max(max_line_transitions, line_transitions)
            sampled_px += line_length_px

        for y0 in range(spacing, h, spacing):
            travel = min(w - 1, h - 1 - y0)
            if travel < 1:
                continue
            xs = np.arange(0, travel + 1)
            ys = np.arange(y0, y0 + travel + 1)
            values = direction_binary[ys, xs]
            ##This line below takes two pixels next to each other and adds 1 to transition count if they are NOT equal. so, if a 0 and 255 are next to each other, its +1 to transition

            line_length_px = len(values)
            line_transitions = np.sum(values[1:] != values[:-1])

            if line_length_px < MIN_LINE_LENGTH_PX:
                continue
            if line_transitions < MIN_LINE_TRANSITIONS:
                continue

            transition_count += line_transitions
            max_line_transitions = max(max_line_transitions, line_transitions)
            sampled_px += line_length_px

    elif direction == "diag_up":
        h, w = direction_binary.shape

        for x0 in range(0, w, spacing):
            travel = min(w - 1 - x0, h - 1)
            if travel < 1:
                continue
            xs = np.arange(x0, x0 + travel + 1)
            ys = np.arange(h - 1, h - 1 - travel - 1, -1)
            values = direction_binary[ys, xs]
            ##This line below takes two pixels next to each other and adds 1 to transition count if they are NOT equal. so, if a 0 and 255 are next to each other, its +1 to transition

            line_length_px = len(values)
            line_transitions = np.sum(values[1:] != values[:-1])

            if line_length_px < MIN_LINE_LENGTH_PX:
                continue
            if line_transitions < MIN_LINE_TRANSITIONS:
                continue

            transition_count += line_transitions
            max_line_transitions = max(max_line_transitions, line_transitions)
            sampled_px += line_length_px

        for y0 in range(0, h - spacing, spacing):
            travel = min(w - 1, y0)
            if travel < 1:
                continue
            xs = np.arange(0, travel + 1)
            ys = np.arange(y0, y0 - travel - 1, -1)
            values = direction_binary[ys, xs]
            ##This line below takes two pixels next to each other and adds 1 to transition count if they are NOT equal. so, if a 0 and 255 are next to each other, its +1 to transition

            line_length_px = len(values)
            line_transitions = np.sum(values[1:] != values[:-1])

            if line_length_px < MIN_LINE_LENGTH_PX:
                continue
            if line_transitions < MIN_LINE_TRANSITIONS:
                continue

            transition_count += line_transitions
            max_line_transitions = max(max_line_transitions, line_transitions)
            sampled_px += line_length_px

    return direction_mask, direction_binary, sampled_px, transition_count, max_line_transitions

# ==========================================
# FOLDERS
# ==========================================

os.makedirs(SC_FOLDER, exist_ok=True)
os.makedirs(RAST_FOLDER, exist_ok=True)

# ==========================================
# FIND PARENT SAMPLES
# ==========================================

npy_files = [f for f in os.listdir(OBJECT_MASK_FOLDER) if f.endswith("_P.npy")]

#--For final domain width and alignment
all_avg_domain_widths = []
all_alignment_pcts = []
#--End
batch_rows = []


for npy_file in npy_files:
    sample_base = npy_file.replace("_P.npy", "")
    type, grain_id = sample_base.split("_")

    grain_domain_widths = []
    grain_alignments = []
    sample_area_data = []
    grain_area_data = []
    aligned_grain_area_data = []
    heatmap_data = []

    #--make folder for each sample in single crystal and raster results
    sample_sc_folder = os.path.join(SC_FOLDER, sample_base)
    os.makedirs(sample_sc_folder, exist_ok=True)
    sample_rast_folder = os.path.join(RAST_FOLDER, sample_base)
    os.makedirs(sample_rast_folder, exist_ok=True)
    #--End

    npy_path = os.path.join(OBJECT_MASK_FOLDER, npy_file)

    polar_img_path = os.path.join(DATA_FOLDER, f"{sample_base}_P.png")

    if not os.path.exists(polar_img_path):
        print(f"Skipping {sample_base}: missing polar image")
        continue

    print(f"Processing {sample_base}...")

    data = np.load(npy_path, allow_pickle=True)

    polar_img = cv2.imread(polar_img_path)

    if  polar_img is None:
        print(f"Skipping {sample_base}: could not read images")
        continue

    if len(data) == 0:
        print(f"Skipping {sample_base}: no grains")
        continue

    # ==========================================
    # 4x4 PATCH SORTING
    # ==========================================

    h, w = polar_img.shape[:2]
    total_image_area_um2 = h * w * (1 / RASTER_PX_LEN_RATIO) ** 2
    sample_area_data.append((sample_base, total_image_area_um2))
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
    cv2.imwrite(os.path.join(sample_sc_folder, qc_filename), qc_img)

    # ==========================================
    # GRAIN LOOP
    # ==========================================
    zero_domain_count = 0
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

        cv2.imwrite(os.path.join(sample_sc_folder, grain_filename), grain_cutout)
        cv2.imwrite(os.path.join(sample_sc_folder, mask_filename), cropped_mask)

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

        #- Grain Area Calculation of mask
        grain_area_px = np.count_nonzero(mask_bool)
        grain_area_um2 = grain_area_px * (1 / RASTER_PX_LEN_RATIO) ** 2

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
        cv2.imwrite(os.path.join(sample_rast_folder, binary_name), binary)

        white_fraction = np.mean(binary[mask_bool] == 255)
        black_fraction = np.mean(binary[mask_bool] == 0)

        reject_binary = (
            white_fraction >= WHITE_REJECT_THRESHOLD or
            black_fraction >= BLACK_REJECT_THRESHOLD
        )

        if reject_binary:
            print(
                f"REJECTED BINARY: {grain_file_base} | "
                f"white={white_fraction:.3f} | black={black_fraction:.3f}"
            )
            continue


        # --------------------------------------
        # RASTERIZATION
        # --------------------------------------

        mask_u8_small = (mask_bool.astype(np.uint8) * 255)

        raster_mask = np.zeros_like(mask_u8_small)
        raster_binary = np.zeros_like(binary)

        total_hv_px = 0
        total_diag_px = 0
        total_transitions = 0
        max_transitions_any_line = 0

        for direction in RASTER_DIRECTIONS:
            dir_mask, dir_binary, sampled_px, transitions, max_line_transitions = measure_direction(
                binary,
                mask_u8_small,
                direction,
                RASTER_SPACING_PX,
                RASTER_LINE_WIDTH_PX
            )

            raster_mask = cv2.bitwise_or(raster_mask, dir_mask)
            raster_binary = cv2.bitwise_or(raster_binary, dir_binary)

            total_transitions += transitions
            max_transitions_any_line = max(max_transitions_any_line, max_line_transitions)

            if direction in ["horizontal", "vertical"]:
                total_hv_px += sampled_px
            else:
                total_diag_px += sampled_px


        if max_transitions_any_line < MIN_MAX_LINE_TRANSITIONS:
            print(f"REJECTED LOW TRANSITIONS: {grain_file_base} | max_line_transitions={max_transitions_any_line}")
            continue
        raster_binary = cv2.bitwise_and(binary, raster_mask)

        # overlay raster on binary image
        binary_overlay = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        binary_overlay[raster_binary > 0] = (255, 0, 0)

        binary_overlay_name = f"{grain_file_base}_binary_raster_overlay.png"
        cv2.imwrite(os.path.join(sample_rast_folder, binary_overlay_name), binary_overlay)

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

        if domain_width_um == 0:                                #Check for 0um domain errors
            print(f"ZERO DOMAIN: {grain_file_base} | transitions={total_transitions} | hv_px={total_hv_px} | diag_px={total_diag_px}")
            zero_domain_count += 1
        alignment = (1 - (domain_width_um - DWCLOSURE) / (DWSTRIPE - DWCLOSURE)) * 100  ### Domain alignment calculation
            
        if domain_width_um > 0:                                                         #Add domain width & alignment to list
            grain_domain_widths.append((grain_file_base, domain_width_um))
            grain_alignments.append((grain_file_base, alignment))
            grain_area_data.append((grain_file_base, grain_area_um2))
            heatmap_data.append({
                "mask": g["mask"],
                "domain_width": domain_width_um,
                "alignment": alignment
            })
            if alignment >= ALIGNMENT_THRESHOLD:
                aligned_grain_area_data.append((grain_file_base, grain_area_um2))
        
        


        # --------------------------------------
        # SAVE RASTER IMAGE
        # --------------------------------------

        overlay = cv2.cvtColor(blurred.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        overlay[raster_binary > 0] = (255, 0, 0)

        raster_name = f"{grain_file_base}_raster.png"
        cv2.imwrite(os.path.join(sample_rast_folder, raster_name), overlay)

        # --------------------------------------
        # SAVE OVERLAY
        # --------------------------------------

        if RASTER_SAVE_OVERLAY:
            overlay = cv2.cvtColor(blurred.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            overlay[(raster_mask > 0) & mask_bool] = (255, 0, 0)

            overlay_name = f"{grain_file_base}_raster_overlay.png"
            cv2.imwrite(os.path.join(sample_rast_folder, overlay_name), overlay)

        # --------------------------------------
        # SAVE COORDS + ANALYSIS CSV
        # --------------------------------------

        if RASTER_SAVE_COORDS:
            csv_name = f"{grain_file_base}_raster_coords.csv"
            csv_path = os.path.join(sample_rast_folder, csv_name)

            with open(csv_path, "w") as f:
                f.write("x,y,analysis,value\n")

                max_len = max(len(coords_list), 7)

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
                    elif j == 5:
                        analysis_text = f"alignment,{alignment}"
                    elif j == 6:
                        analysis_text = f"grain_area_um2,{grain_area_um2}"
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
        plt.savefig(os.path.join(sample_rast_folder, qc_name))
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
        plt.savefig(os.path.join(sample_rast_folder, hist_qc_name))
        plt.close()
    
    # ==========================================
    # HEAT MAPS
    # ==========================================

    if len(heatmap_data) > 0:
        width_vals = [d["domain_width"] for d in heatmap_data]
        align_vals = [d["alignment"] for d in heatmap_data]

        w_min, w_max = min(width_vals), max(width_vals)
        a_min, a_max = min(align_vals), max(align_vals)

        heatmap_width = np.zeros((h, w), dtype=np.float32)
        heatmap_align = np.zeros((h, w), dtype=np.float32)

        for d in heatmap_data:
            mask = d["mask"] > 0

            heatmap_width[mask] = d["domain_width"]
            heatmap_align[mask] = d["alignment"]

        width_color = cv2.applyColorMap((heatmap_width * 255).astype(np.uint8), cv2.COLORMAP_JET)
        align_color = cv2.applyColorMap((heatmap_align * 255).astype(np.uint8), cv2.COLORMAP_JET)

        alpha = 0.5
        overlay_width = cv2.addWeighted(polar_img, 1 - alpha, width_color, alpha, 0)
        overlay_align = cv2.addWeighted(polar_img, 1 - alpha, align_color, alpha, 0)

        #- Plot width heatmap
        img_h, img_w = polar_img.shape[:2]
        fig_w = 10
        fig_h = fig_w * img_h / img_w
        fig, ax = plt.subplots(figsize=(14, 14))

        ax.imshow(cv2.cvtColor(polar_img, cv2.COLOR_BGR2RGB))
        im = ax.imshow(heatmap_width, cmap="jet", alpha=0.5, vmin=w_min, vmax=w_max)
        ax.set_facecolor("black") #Remove if you want to overlay original image
        ax.axis("off")
        cbar = plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.02)
        cbar.set_label("Domain Width (um)")
        plt.subplots_adjust(left=0, right=1, top=0.5, bottom=0.2)
        plt.savefig(os.path.join(sample_rast_folder, f"{sample_base}_domain_width_heatmap.png"), bbox_inches="tight", pad_inches=0)
        plt.close()

        #- Plot alignment heatmap
        img_h, img_w = polar_img.shape[:2]
        fig_w = 10
        fig_h = fig_w * img_h / img_w
        fig, ax = plt.subplots(figsize=(14, 14))

        ax.imshow(cv2.cvtColor(polar_img, cv2.COLOR_BGR2RGB))
        im = ax.imshow(heatmap_align, cmap="coolwarm", alpha=0.5, vmin=a_min, vmax=a_max)
        ax.set_facecolor("black") #Remove if you want to overlay original image
        ax.axis("off")
        cbar = plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.02)
        cbar.set_label("Alignment (%)")
        plt.subplots_adjust(left=0, right=1, top=.6, bottom=0.2)
        plt.savefig(os.path.join(sample_rast_folder, f"{sample_base}_alignment_heatmap.png"), bbox_inches="tight", pad_inches=0)
        plt.close()




    # ==========================================
    # SAVE OVERALL DOMAIN AVERAGE OF PARENT
    # ==========================================

    summary_path = os.path.join(sample_rast_folder, f"{sample_base}_metadata.csv")

    domain_width_values = [v for _, v in grain_domain_widths if v > 0]
    grain_area_values = [v for _, v in grain_area_data]
    alignment_values = [a for _, a in grain_alignments]

    # --- Domain width stats ---
    if len(domain_width_values) > 0:
        sample_avg_domain_width_um = np.mean(domain_width_values)
        sample_median_domain_width_um = np.median(domain_width_values)
        sample_std_domain_width_um = np.std(domain_width_values)
        sample_min_domain_width_um = np.min(domain_width_values)
        sample_max_domain_width_um = np.max(domain_width_values)
    else:
        sample_avg_domain_width_um = 0
        sample_median_domain_width_um = 0
        sample_std_domain_width_um = 0
        sample_min_domain_width_um = 0
        sample_max_domain_width_um = 0

    # --- Grain area stats ---
    if len(grain_area_values) > 0:
        sample_avg_grain_area_um2 = np.mean(grain_area_values)
        sample_median_grain_area_um2 = np.median(grain_area_values)
        sample_std_grain_area_um2 = np.std(grain_area_values)
    else:
        sample_avg_grain_area_um2 = 0
        sample_median_grain_area_um2 = 0
        sample_std_grain_area_um2 = 0

    # --- Alignment stats ---
    if len(alignment_values) > 0:
        sample_avg_alignment_pct = np.mean(alignment_values)
        sample_median_alignment_pct = np.median(alignment_values)
        sample_std_alignment_pct = np.std(alignment_values)
    else:
        sample_avg_alignment_pct = 0
        sample_median_alignment_pct = 0
        sample_std_alignment_pct = 0

    # --- Count and area totals ---
    sample_grain_count = len(domain_width_values)
    if total_image_area_um2 > 0:
        sample_grain_density_grains_per_um2 = sample_grain_count / total_image_area_um2
    else:
        sample_grain_density_grains_per_um2 = 0


    all_grain_area_um2 = sum(grain_area_values)
    aligned_grain_area_um2 = sum(v for _, v in aligned_grain_area_data)

    if all_grain_area_um2 > 0:
        sample_grain_area_alignment_pct = 100 * aligned_grain_area_um2 / all_grain_area_um2
    else:
        sample_grain_area_alignment_pct = 0

    

    if len(alignment_values) > 0:
        aligned_count = sum(a >= ALIGNMENT_THRESHOLD for a in alignment_values)
        sample_grain_count_alignment_pct = 100 * aligned_count / len(alignment_values)
    else:
        aligned_count = 0
        sample_grain_count_alignment_pct = 0



    with open(summary_path, "w") as f:
        f.write(
            "uid,low_contrast_img,high_contrast_img,moke_img,grain_dw_um,grain_alignment_pct,grain_grain_area_um2,"
            "sample_avg_domain_width_um,sample_image_area_um2,"
            "sample_total_grain_area_um2,sample_aligned_grain_area_um2,"
            "sample_grain_count_alignment_pct,sample_grain_area_alignment_pct\n"
        )

        alignment_dict = dict(grain_alignments)
        area_dict = dict(grain_area_data)

        for name, value in grain_domain_widths:
            alignment_value = alignment_dict.get(name, "")
            area_value = area_dict.get(name, "")

            f.write(
                f"{sample_base},{sample_base}_LC.png,{sample_base}_HC.png,{sample_base}_P.png,{value},{alignment_value},{area_value},"
                f"{sample_avg_domain_width_um},{total_image_area_um2},"
                f"{all_grain_area_um2},{aligned_grain_area_um2},"
                f"{sample_grain_count_alignment_pct},{sample_grain_area_alignment_pct}\n"
        )
        print(f"\nTOTAL ZERO DOMAIN GRAINS: {zero_domain_count}")   ##See if any 0 um domains are found. error check

        batch_rows.append({
        "uid": sample_base,
        "low_contrast_img": f"{sample_base}_LC.png",
        "high_contrast_img": f"{sample_base}_HC.png",
        "moke_img": f"{sample_base}_P.png",

        "sample_avg_domain_width_um": sample_avg_domain_width_um,
        "sample_median_domain_width_um": sample_median_domain_width_um,
        "sample_std_domain_width_um": sample_std_domain_width_um,
        "sample_min_domain_width_um": sample_min_domain_width_um,
        "sample_max_domain_width_um": sample_max_domain_width_um,

        "sample_avg_grain_area_um2": sample_avg_grain_area_um2,
        "sample_median_grain_area_um2": sample_median_grain_area_um2,
        "sample_std_grain_area_um2": sample_std_grain_area_um2,

        "sample_avg_alignment_pct": sample_avg_alignment_pct,
        "sample_median_alignment_pct": sample_median_alignment_pct,
        "sample_std_alignment_pct": sample_std_alignment_pct,

        "sample_image_area_um2": total_image_area_um2,
        "sample_total_grain_area_um2": all_grain_area_um2,
        "sample_aligned_grain_area_um2": aligned_grain_area_um2,

        "sample_grain_count": sample_grain_count,
        "sample_grain_density_grains_per_um2": sample_grain_density_grains_per_um2,
        "sample_grain_count_alignment_pct": sample_grain_count_alignment_pct,
        "sample_grain_area_alignment_pct": sample_grain_area_alignment_pct,

        "target_grade": type
        })




#-Get final averages for domain width and grain alignment
master_path = os.path.join(RAST_FOLDER, "batch_summary.csv")

batch_columns = [
    "uid",
    "low_contrast_img",
    "high_contrast_img",
    "moke_img",

    "sample_avg_domain_width_um",
    "sample_median_domain_width_um",
    "sample_std_domain_width_um",
    "sample_min_domain_width_um",
    "sample_max_domain_width_um",

    "sample_avg_grain_area_um2",
    "sample_median_grain_area_um2",
    "sample_std_grain_area_um2",

    "sample_avg_alignment_pct",
    "sample_median_alignment_pct",
    "sample_std_alignment_pct",

    "sample_image_area_um2",
    "sample_total_grain_area_um2",
    "sample_aligned_grain_area_um2",

    "sample_grain_count",
    "sample_grain_density_grains_per_um2",
    "sample_grain_count_alignment_pct",
    "sample_grain_area_alignment_pct",

    "target_grade"
]

with open(master_path, "w") as f:
    f.write(",".join(batch_columns) + "\n")

    for row in batch_rows:
        f.write(",".join(str(row.get(col, "")) for col in batch_columns) + "\n")