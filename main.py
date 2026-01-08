import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from scipy.stats import wasserstein_distance

class GrainSegmenter:
    def __init__(self, model_type="vit_h", checkpoint_path="assets/models/sam_vit_h_4b8939.pth"):
        print("--- INITIALIZING AI ---")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model not found at {checkpoint_path}. Did you download it?")
            
        print("Loading SAM Model... (this may take 10-20 seconds)")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)
        
        # TUNE THESE KNOBS: Adjusting sensitivity for magnet grains
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,             # Higher = more detail
            pred_iou_thresh=0.86,           # Confidence threshold (86%)
            stability_score_thresh=0.92,    # Stability threshold
            crop_n_layers=1,                # Zoom levels
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100        # Ignore dust/noise < 100 pixels
        )

    def analyze_image(self, image_path, microns_per_pixel=1.0):
        """Returns a list of grain areas in square microns."""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read {image_path}")
            return []
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(image_rgb)
        
        grain_areas = []
        for mask_data in masks:
            area_pixels = mask_data['area']
            # Convert pixel area to physical area (microns^2)
            area_microns = area_pixels * (microns_per_pixel ** 2)
            grain_areas.append(area_microns)
            
        return grain_areas

    def save_visualization(self, image_path, output_path):
        """Generates masks and saves an image with colorful outlines overlaid."""
        print(f"Generating visual overlay for {os.path.basename(image_path)}...")
        
        image = cv2.imread(image_path)
        if image is None: 
            print("Image not found for visualization.")
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(image)
        
        # Setup plot
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        
        # Sort masks by area (largest first) so small ones aren't hidden
        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        
        ax = plt.gca()
        ax.set_autoscale_on(False)

        # Draw each mask with a random color
        for mask_data in sorted_masks:
            m = mask_data['segmentation']
            img = np.ones((m.shape[0], m.shape[1], 3))
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                img[:,:,i] = color_mask[i]
            
            # Overlay with transparency (alpha = 0.35)
            ax.imshow(np.dstack((img, m*0.35)))
        
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close() # Free up memory
        print(f"Saved visualization to: {output_path}")

class MagnetGrader:
    def __init__(self, segmenter):
        self.segmenter = segmenter
        self.reference_profiles = {} # Stores avg grain size for A, B, C, D

    def learn_grade(self, grade_name, folder_path):
        """Ingests all images in a folder to build a master profile for that grade."""
        print(f"--- Learning Grade {grade_name} from {folder_path} ---")
        
        if not os.path.exists(folder_path):
            print(f"Skipping Grade {grade_name}: Folder not found.")
            return

        all_areas = []
        valid_extensions = {".jpg", ".png", ".tif", ".tiff", ".jpeg"}
        
        files = [f for f in os.listdir(folder_path) if any(f.lower().endswith(ext) for ext in valid_extensions)]
        
        if len(files) == 0:
            print(f"Warning: No images found in {folder_path}")
            return

        for filename in files:
            filepath = os.path.join(folder_path, filename)
            # print(f"  Processing: {filename}") # Uncomment to see every file
            areas = self.segmenter.analyze_image(filepath)
            all_areas.extend(areas)
        
        if not all_areas:
            print(f"Error: No grains detected in Grade {grade_name}.")
            return

        # Calculate Median Equivalent Circle Diameter (ECD)
        # Formula: d = 2 * sqrt(Area / Pi)
        ecds = [2 * np.sqrt(a / np.pi) for a in all_areas]
        avg_ecd = np.median(ecds)
        
        self.reference_profiles[grade_name] = {
            "avg_ecd": avg_ecd,
            "raw_data": ecds
        }
        print(f"SUCCESS: Grade {grade_name} Learned. Median Grain Size: {avg_ecd:.2f} microns (based on {len(all_areas)} grains)\n")

    def grade_sample(self, image_path):
        """Compares a new image against the learned grades."""
        print(f"--- Grading Sample: {os.path.basename(image_path)} ---")
        if not os.path.exists(image_path):
            print("Error: Test image not found.")
            return

        areas = self.segmenter.analyze_image(image_path)
        if not areas:
            print("Error: No grains found in test image.")
            return

        sample_ecds = [2 * np.sqrt(a / np.pi) for a in areas]
        sample_avg = np.median(sample_ecds)
        
        print(f"Sample Median Grain Size: {sample_avg:.2f} microns")
        
        # Sort grades by size to determine ranking
        sorted_grades = sorted(self.reference_profiles.items(), key=lambda x: x[1]['avg_ecd'])
        
        lower_than = []
        higher_than = []
        
        print("\n--- Detailed Comparison ---")
        for name, data in sorted_grades:
            ref_size = data['avg_ecd']
            # Wasserstein Distance: 0 = Identical, Higher = More different
            dist = wasserstein_distance(sample_ecds, data['raw_data'])
            
            print(f"Vs Grade {name} ({ref_size:.2f} µm): Dist Score = {dist:.2f}")
            
            if sample_avg > ref_size:
                higher_than.append(name)
            else:
                lower_than.append(name)
                
        # Final Output Logic
        print("\n--- FINAL VERDICT ---")
        if not higher_than:
            print(f"RESULT: Sample is finer (smaller) than all reference grades.")
        elif not lower_than:
            print(f"RESULT: Sample is coarser (larger) than all reference grades.")
        else:
            print(f"RESULT: Sample is larger than {', '.join(higher_than)} but smaller than {', '.join(lower_than)}.")

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # 1. Initialize System
    segmenter = GrainSegmenter(checkpoint_path="assets/models/sam_vit_h_4b8939.pth")
    grader = MagnetGrader(segmenter)
    
    # 2. Learn from your Reference Folders
    # The grader learns what A, B, C, and D look like
    grader.learn_grade("A", "data/grade_N35")
    grader.learn_grade("B", "data/grade_N42")
    grader.learn_grade("C", "data/grade_N48")
    grader.learn_grade("D", "data/grade_N52")
    
    # 3. Process ALL images in the test_samples folder
    test_folder = "data/test_samples"
    valid_extensions = {".jpg", ".png", ".tif", ".tiff", ".jpeg"}
    
    # Get a list of all valid images in the folder
    test_files = [f for f in os.listdir(test_folder) if any(f.lower().endswith(ext) for ext in valid_extensions)]
    
    if not test_files:
        print(f"No test images found in {test_folder}")
    else:
        print(f"\nFound {len(test_files)} test images. Starting analysis...\n")
        
        for filename in test_files:
            # Construct the full path to the image
            image_path = os.path.join(test_folder, filename)
            
            # Run the grading logic
            print(f"--- STARTING NEW TEST: {filename} ---")
            grader.grade_sample(image_path)
            
            # Generate the visual proof (saved with the original filename prefix)
            output_name = f"visual_result_{filename}"
            segmenter.save_visualization(image_path, output_name)
            print(f"--------------------------------------\n")