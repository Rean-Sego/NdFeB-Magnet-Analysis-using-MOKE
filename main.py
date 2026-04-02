import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from skimage.color import rgb2gray
from skimage.feature import structure_tensor
from skimage.measure import regionprops
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage import exposure

# ==========================================
# --- CONFIGURATION ---
# ==========================================
MICRONS_PER_PIXEL = 0.066  
ALIGNMENT_THRESHOLD_DEG = 15.0 
SCALE_BAR_HEIGHT = 180  #How much of the bottom of the image is cropped

# MODEL SETTINGS
MODEL_TYPE = "vit_b"
CHECKPOINT_PATH = "segment-anything/assets/models/sam_vit_b_01ec64.pth"

# FOLDERS
OUTPUT_FOLDER = "processed_results"
DATA_FOLDER = "processed_data"
TEST_FOLDER = "data/test_samples"
SC_FOLDER = "sc_results"
MAX_GRAIN_DIAMETER = 50.0 

# ==========================================
# --- CORE LOGIC ---
# ==========================================

class GrainSegmenter:
    def __init__(self, checkpoint_path=CHECKPOINT_PATH):
        print("--- INITIALIZING AI (V32.0 DUAL-STREAM FUSION) ---")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            print("Using GPU (CUDA)")
        else:
            print("WARNING: Using CPU")        
            
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError("Model missing")

        self.sam = sam_model_registry[MODEL_TYPE](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)
        
        self.mask_generator = SamAutomaticMaskGenerator(  #Variables to how the prompt points and boxes are made
            model=self.sam,
            points_per_side=64,             
            crop_n_layers=1,                
            crop_n_points_downscale_factor=2,
            pred_iou_thresh=0.90,           
            stability_score_thresh=0.92,
            min_mask_region_area=50  
        )

    def grain_detection(self, object_list, struct_img, percentile=10):
        """
        Filters objects based on brightness in struct image.
        Returns only likely grains.
        """
        gray = cv2.cvtColor(struct_img, cv2.COLOR_BGR2GRAY)
        
        #  --- build histogram from full struct image ---
        threshold = np.percentile(gray,percentile) #Get the 88th percentile of brightness in the struct image, to use as a cutoff for what is a grain and what is a pore/phase

        grain_list = []

        for g in object_list:
            mask =g["mask"]

            pixels = gray[mask > 0]
            if len(pixels) == 0: continue

            mean_val = np.mean(pixels)

            if mean_val > threshold:
                grain_list.append(g)
        return grain_list

    def process_image_common(self, image):
        """Helper to Clean and Enhance images."""
        h, w = image.shape[:2]
        if w > 2000 or h > 2000:            #Downsize large images
            scale = 2000 / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h))

        clean = image.copy()                #Crop out scale bar
        h, w = clean.shape[:2] 
        clean = clean[:h-SCALE_BAR_HEIGHT, :] # Remove Scale Bar
        
        # CLAHE Contrast Boost
        lab = cv2.cvtColor(clean, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)) 
        cl = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        
        return clean, enhanced

    def align_images_safe(self, img_ref, img_target): #This is aligning the images? I am not sure why? I am not sure how this helps. Internsity of MOKE and Non MOKE wont be similar
        """Translation-Only Alignment."""             #struct is being passed as img_ref. polar is being passed as img_target accorind to analyze pairs
        gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
        gray_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)
        warp_mode = cv2.MOTION_TRANSLATION
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)
        try:
            (cc, warp_matrix) = cv2.findTransformECC(gray_ref, gray_target, warp_matrix, warp_mode, criteria)
            aligned_image = cv2.warpAffine(img_target, warp_matrix, (img_ref.shape[1], img_ref.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            return aligned_image
        except cv2.error:
            return img_target                                          #Returning the polar image

    def get_defect_mask_from_structure(self, struct_img): #defunct. May delete after removing downstream influence
        """
        STREAM A: Detects Pores and Phases using the clean Structure Image.
        Returns a mask where 255 = Defect (Bad), 0 = Grain (Good).
        """
        gray = cv2.cvtColor(struct_img, cv2.COLOR_BGR2GRAY)
        
        # 1. Pores are PITCH BLACK (< 30)
        _, pores = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)  #Below 30 becomes white, makes pores white
        
        # 2. Phases are BRIGHT WHITE (> 230)
        _, phases = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY) #Above 230 becomes White, makes crystals white
        
        # 3. Combine
        defects = cv2.bitwise_or(pores, phases)
        
        # 4. Dilate slightly to ensure we cover the edges     (Will increase the outer edges to be white by 2 pxiels, to help clean up super sharp changes)
        kernel = np.ones((2,2), np.uint8)
        defects = cv2.dilate(defects, kernel, iterations=1)
        
        return defects

    def get_boundary_barrier(self, struct_img):         #Needed for watersheding. You want the grain edges to be detected, this keeps the segmenting from cutting the images down the middle.
        """
        Creates a 'Barrier Map' from grain boundaries in the structure image.
        Used to stop Watershed from cutting through grains.
        """
        gray = cv2.cvtColor(struct_img, cv2.COLOR_BGR2GRAY)
        # Invert: Boundaries are dark, so inverted they become bright "walls"
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        return edges

    def apply_guided_watershed(self, mask, boundary_barrier):
        """
        V32.0: GUIDED WATERSHED.
        Uses the Structure Image boundaries to guide the cuts.
        """
        # 1. Distance Transform on the Mask
        dist = distance_transform_edt(mask)
        
        # 2. V32 KEY: Subtract the physical boundaries from the distance map
        # This creates "valleys" where the real grain boundaries are.
        # We normalize the barrier to be 0-1 and subtract it.
        barrier_norm = boundary_barrier.astype(float) / 255.0
        dist = dist - (barrier_norm * 5.0) # Weight the barrier heavily
        
        dist = gaussian_filter(dist, sigma=1.0) # Smooth out noise
        
        # 3. Find Peaks
        coords = peak_local_max(dist, min_distance=10, labels=mask)
        
        mask_bool = mask.astype(bool)
        markers = np.zeros(dist.shape, dtype=int)
        for i, coord in enumerate(coords):
            markers[tuple(coord)] = i + 1
            
        if len(coords) <= 1: return [mask]

        labels = watershed(-dist, markers, mask=mask_bool)
        new_masks = []
        for i in range(1, labels.max() + 1):
            split_mask = (labels == i).astype(np.uint8)
            if np.sum(split_mask) > 50: 
                new_masks.append(split_mask)
        return new_masks

    def remove_overlapping_masks(self, raw_masks):
        """Strict IoU NMS to remove duplicates."""
        sorted_masks = sorted(raw_masks, key=lambda x: x['stability_score'], reverse=True)
        keep_masks = []
        
        for i, current in enumerate(sorted_masks):
            curr_mask = current['segmentation']
            curr_area = np.sum(curr_mask)
            if curr_area == 0: continue

            is_duplicate = False
            for kept in keep_masks:
                kept_mask = kept['segmentation']
                intersection = np.logical_and(curr_mask, kept_mask)
                inter_area = np.sum(intersection)
                if inter_area == 0: continue
                
                overlap_ratio = inter_area / curr_area
                if overlap_ratio > 0.40: 
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                keep_masks.append(current)
        return keep_masks

    def classify_grains_hybrid(self, grain_data_list):
        if not grain_data_list: return []

        textures = [g['texture'] for g in grain_data_list]
        median_texture = np.median(textures)
        texture_threshold = max(2.5, median_texture * 0.35)
        
        print(f"    -> Hybrid Logic: Median Tex={median_texture:.2f} | Cutoff={texture_threshold:.2f}")

        magnetic_count = 0
        for g in grain_data_list:
            is_textured = g['texture'] >= texture_threshold
            is_gray = 40 < g['brightness'] < 230
            
            if is_textured or is_gray:
                g['type'] = 'magnetic'
                magnetic_count += 1
            else:
                g['type'] = 'non_magnetic'
        
        print(f"    -> Result: {magnetic_count} Magnets | {len(grain_data_list) - magnetic_count} Non-Magnets")
        return grain_data_list

    def analyze_pair(self, struct_path, polar_path):    ## 1) Cleans & aligns the image up; 2) outputs sam as "mask"
        print(f"  [1/4] Loading & Preprocessing...")
        img_struct = cv2.imread(struct_path)
        img_polar = cv2.imread(polar_path)
        if img_struct is None or img_polar is None: return None, None, None

        clean_struct, enhanced_struct = self.process_image_common(img_struct)
        clean_polar, _ = self.process_image_common(img_polar)

        print(f"  [2/4] Aligning...")
        clean_polar = self.align_images_safe(clean_struct, clean_polar)

        # --- STREAM A: DEFECT DETECTION (Structure) ---
        print(f"  [3/4] Building Defect Map from Structure...")
        defect_mask = self.get_defect_mask_from_structure(clean_struct)
        boundary_barrier = self.get_boundary_barrier(clean_struct)

        # --- STREAM B: GRAIN DETECTION (Polar/High-Contrast) ---
        print(f"  [4/4] Running AI on Polar Image (Guided)...")
        # Use Polar for segmentation as it shows domains best
        masks = self.mask_generator.generate(cv2.cvtColor(clean_struct, cv2.COLOR_BGR2RGB))           ###THIS RUNS SAM ON MOKE IMAGE
        
        print(f"        -> Found {len(masks)} candidates. De-duplicating...")
        masks = self.remove_overlapping_masks(masks)
        
        gray_polar = cv2.cvtColor(clean_polar, cv2.COLOR_BGR2GRAY)
        temp_data_list = []

        total_image_area = clean_struct.shape[0] * clean_struct.shape[1]

        for mask_data in masks:
            raw_mask = mask_data['segmentation'].astype(np.uint8)
            mask_area = np.sum(raw_mask)
            
            # 1. Filter Massive Background Masks
            if mask_area > (total_image_area * 0.05): continue

            # 2. Subtract Defects (Pores/Phases)
            clean_mask = cv2.bitwise_and(raw_mask, cv2.bitwise_not(defect_mask))
            
            # 3. Guided Watershed Split
            # Use the structure boundaries to force cuts in the right places
            split_masks = self.apply_guided_watershed(clean_mask, boundary_barrier)
            
            for sub_mask in split_masks:
                if np.count_nonzero(sub_mask) < 30: continue 

                masked_polar_pixels = gray_polar[sub_mask > 0]
                if len(masked_polar_pixels) == 0: continue

                texture = np.std(masked_polar_pixels)
                brightness = np.mean(masked_polar_pixels)
                
                area_microns = np.count_nonzero(sub_mask) * (MICRONS_PER_PIXEL**2)
                diameter = 2 * np.sqrt(area_microns / np.pi)
                
                grain_item = {
                    "mask": sub_mask,
                    "texture": texture,
                    "brightness": brightness,
                    "diameter": diameter,
                    "misalignment": 0, 
                    "type": "unknown" 
                }
                temp_data_list.append(grain_item)

        classified_data = self.classify_grains_hybrid(temp_data_list)
        
        object_list = []
        for g in classified_data:
            if g['type'] == 'magnetic':
                moke = cv2.bitwise_and(clean_polar, clean_polar, mask=g['mask'])
                st = structure_tensor(cv2.cvtColor(moke, cv2.COLOR_BGR2GRAY), sigma=1)
                angles = 0.5 * np.arctan2(2 * st[1], st[0] - st[2])
                valid = angles[g['mask'] > 0]
                
                if len(valid) > 0:
                    stripe_angle = np.degrees(np.median(valid))
                    grain_angle = np.degrees(regionprops(g['mask'])[0].orientation)
                    misalign = abs(grain_angle - stripe_angle) % 90
                    if misalign > 45: misalign = 90 - misalign
                    g['misalignment'] = misalign
            
            object_list.append(g)

        return clean_struct, clean_polar, object_list



class GrainInspector:
    def __init__(self, image_path, data_path, filename):
        self.grain_data = np.load(data_path, allow_pickle=True)
        self.image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        
        h, w = self.image.shape[:2]
        if SCALE_BAR_LOCATION == "bottom-right": self.image[h-SCALE_BAR_HEIGHT:h, w-SCALE_BAR_WIDTH:w] = 0
        elif SCALE_BAR_LOCATION == "bottom-left": self.image[h-SCALE_BAR_HEIGHT:h, 0:SCALE_BAR_WIDTH] = 0
        
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.redraw()
        plt.show()

    def redraw(self):
        self.ax.clear()
        self.ax.imshow(self.image)
        self.ax.axis('off')
        overlay = np.zeros_like(self.image)
        for g in self.grain_data:
            if g['type'] == 'non_magnetic':
                color = (255, 255, 0) # YELLOW
            else:
                color = (0, 255, 0) if g['misalignment'] < ALIGNMENT_THRESHOLD_DEG else (255, 0, 0)
            
            cv2.drawContours(overlay, [cv2.findContours(g['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]], -1, color, 2)
        self.ax.imshow(overlay, alpha=0.6)

    def on_click(self, event):
        if not event.xdata: return
        x, y = int(event.xdata), int(event.ydata)
        for g in self.grain_data:
            if g['mask'][y, x]:
                if g['type'] == 'non_magnetic':
                    self.ax.set_title(f"Non-Magnetic (Tex: {g['texture']:.1f})")
                else:
                    d, a = g['diameter'], g['misalignment']
                    status = "ALIGNED" if a < ALIGNMENT_THRESHOLD_DEG else "MISALIGNED"
                    self.ax.set_title(f"Dia: {d:.2f}µm | Misalign: {a:.1f}° | {status}")
                self.fig.canvas.draw()
                return

# --- Beginning of Main Exectution ---
if __name__ == "__main__":
    print("\n!!! RUNNING VERSION 32.0 (DUAL-STREAM FUSION) !!!")
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(DATA_FOLDER, exist_ok=True)
    os.makedirs(SC_FOLDER, exist_ok=True)


    seg = GrainSegmenter()
    
    struct_files = [f for f in os.listdir(TEST_FOLDER) if "_struct" in f]                                       ### Look for all the struct files
    print(f"Found {len(struct_files)} Non Moke Images.")
    
    # --- Begin Loop for the searching for pairs
    for f in struct_files:                                                                                     
        f_polar = f.replace("_struct", "_polar")                                                               ###Three lines save the struct and polar as path_s and path_p
        path_s = os.path.join(TEST_FOLDER, f)                                                                   
        path_p = os.path.join(TEST_FOLDER, f_polar)
        
        if not os.path.exists(path_p): continue
        
        print(f"Analyzing {f}...")
        img, processed_polar, data = seg.analyze_pair(path_s, path_p)                                          ### Run anaylze pairs and SAM in all files
        data = seg.grain_detection(data, img, percentile=22)

        base = os.path.splitext(f)[0].replace("_struct", "")

        cv2.imwrite(os.path.join(DATA_FOLDER, f"{base}_struct.png"), img)
        cv2.imwrite(os.path.join(DATA_FOLDER, f"{base}_polar.png"), processed_polar)
        
        if data:                                                                                               ###data is a list of the outputted grains. data = object_list

            # --- More Main Code Stuff --
            np.save(os.path.join(DATA_FOLDER, os.path.splitext(f)[0] + ".npy"), data)
            
            vis = processed_polar.copy()
            for g in data:
                if g['type'] == 'non_magnetic':
                    color = (0, 255, 255) # Yellow
                else:
                    color = (0, 255, 0) if g['misalignment'] < ALIGNMENT_THRESHOLD_DEG else (0, 0, 255)
                
                cnt = cv2.findContours(g['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                cv2.drawContours(vis, cnt, -1, color, 2)
                
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"Result_{f}.png"), vis)
            print(f"  -> Saved Result to {OUTPUT_FOLDER}/Result_{f}.png")




