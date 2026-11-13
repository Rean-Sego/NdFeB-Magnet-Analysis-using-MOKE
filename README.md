# MOKE-Based Magnetic Domain Analysis and Grain Segmentation

## 1. Overview

This repository contains a Python-based image processing pipeline developed to analyze the microstructure and magnetic domains of sintered NdFeB permanent magnets.

The pipeline combines the **Segment Anything Model (SAM)** with conventional image processing techniques to identify individual grains, extract magnetic domain patterns, and calculate quantitative microstructural and magnetic features.

The workflow consists of three primary stages:

1. **Grain segmentation:** Identify individual grains using SAM and structural microscopy images.
2. **Domain analysis:** Process magneto-optical Kerr effect (MOKE) images to measure magnetic domain widths within segmented grains.
3. **Feature extraction:** Calculate grain areas, domain widths, magnetic alignment estimates, and sample-level statistics.

The resulting measurements can be used as inputs to a separate machine learning pipeline for magnet grade classification.

---

## 2. Requirements

### Software

- Python 3.12
- Git
- CUDA-compatible NVIDIA GPU (recommended)
- PyTorch with CUDA support (recommended)

The segmentation pipeline uses the following Python libraries:

- `torch`
- `torchvision`
- `segment-anything`
- `opencv-python`
- `numpy`
- `scipy`
- `scikit-image`
- `matplotlib`

Additional dependencies may be required depending on the installed versions of these packages.

### Hardware

SAM can run on a CPU, but a CUDA-compatible GPU is strongly recommended for processing microscopy images efficiently.

GPU memory requirements depend on image resolution, model architecture, and segmentation parameters.

---

## 3. Installation

### Step 1: Clone the repository

Open a terminal and run:

```bash
git clone <REPOSITORY_URL>
cd MOKE_REPO
```

Replace `<REPOSITORY_URL>` with the GitHub repository URL.

### Step 2: Create a Python environment

Creating a dedicated virtual environment is recommended to avoid dependency conflicts.

```bash
python -m venv .venv
```

Activate the environment.

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
source .venv/bin/activate
```

### Step 3: Install PyTorch

Install a compatible version of PyTorch using the instructions provided by the official PyTorch installation guide.

For GPU acceleration, select the CUDA build compatible with your system.

Step 4: Install Python dependencies

Install the required Python packages using the provided requirements.txt file.

From the directory containing requirements.txt, run:

pip install -r requirements.txt

This command installs the dependencies specified in the file, including the image processing libraries and any listed PyTorch and Segment Anything dependencies.

---

## 4. Segment Anything Model (SAM)

### Model architecture

The segmentation pipeline uses the original **Segment Anything Model (SAM 1)** developed by Meta.

The configured architecture is:

**SAM ViT-B (Vision Transformer Base)**

The model checkpoint is:

```text
sam_vit_b_01ec64.pth
```

### Downloading the checkpoint

Download the pretrained SAM ViT-B checkpoint from the official Segment Anything repository.

The program expects the checkpoint at:

```text
segment-anything/assets/models/sam_vit_b_01ec64.pth
```

If the checkpoint is stored elsewhere, update `CHECKPOINT_PATH` in `main.py`.

### Segmentation configuration

The current SAM configuration uses:

| Parameter | Value |
|---|---|
| Model architecture | ViT-B |
| Points per side | 32 |
| Crop layers | 1 |
| Crop point downscale factor | 2 |
| Predicted IoU threshold | 0.50 |
| Stability score threshold | 0.92 |
| Minimum mask region area | 50 pixels |

These parameters control the generation and initial filtering of candidate segmentation masks.

Users may adjust these settings depending on image resolution, contrast, and microstructural characteristics.

---

## 5. Input Images

### Image requirements

The pipeline was developed for three corresponding microscopy images of each magnet sample:

| Image | Description |
|---|---|
| LC | Low-contrast structural image |
| HC | High-contrast structural image |
| P | Polarized MOKE image |

The structural images are used for grain identification, while the polarized MOKE image provides magnetic domain information.

The images should represent the same region of the sample with approximately matching magnification and orientation.

### File naming convention

Images must follow the naming convention:

```text
N48_001_LC.png
N48_001_HC.png
N48_001_P.png
```

Here:

- `N48` identifies the magnet grade.
- `001` identifies the sample.
- `LC`, `HC`, and `P` identify the image type.

The same naming convention applies to other grades and samples.

### Important considerations

For accurate analysis:

- Images should be properly focused.
- Structural and MOKE images should correspond to the same sample region.
- Pixel calibration must be determined from the microscope magnification or scale bar.
- Scale bars and other annotations should not interfere with the analyzed region.
- Image dimensions should be consistent wherever possible.

---

## 6. Grain Segmentation

The primary segmentation program is:

```text
Segmentation/main.py
```

### Configuration

Before running the program, review the user settings in `main.py`.

Important settings include:

**Structural image selection**

```python
STRUCT_MODE = "HC"
```

Options:

- `"HC"` — High-contrast structural images
- `"LC"` — Low-contrast structural images

**Pixel calibration**

```python
MICRONS_PER_PIXEL = 0.066
```

This parameter converts pixel-based grain dimensions into physical units.

It must be updated when processing images acquired at different magnifications or resolutions.

**SAM model**

```python
MODEL_TYPE = "vit_b"
```

The model type must match the downloaded checkpoint.

### Running segmentation

Navigate to the segmentation directory:

```bash
cd Segmentation
```

Run:

```bash
python main.py
```

The program processes the available sample images and saves the segmentation results.

### Segmentation procedure

The segmentation pipeline performs the following operations:

**1. Image preprocessing**

Structural and MOKE images are loaded and prepared for analysis.

Preprocessing includes image resizing when necessary, cropping, and contrast enhancement.

The current implementation also removes a fixed region from the bottom of each image to exclude the scale bar. Users should verify that this setting is appropriate for their images.

**2. Image alignment**

The structural and polarized images are aligned to compensate for translational differences between acquisitions.

**3. SAM segmentation**

SAM generates candidate masks corresponding to distinct image features.

The structural image is used for segmentation.

**4. Overlap removal**

Overlapping masks are compared to reduce duplicate detections.

The current implementation rejects a candidate mask when more than 40% of its area overlaps a previously retained mask.

**5. Defect identification**

Intensity-based thresholding identifies regions associated with pores and other non-grain features.

These regions are excluded from the candidate grain masks.

**6. Grain separation**

A boundary-guided watershed algorithm attempts to separate adjacent grains that were identified as a single object.

**7. Grain classification**

The pipeline evaluates image brightness and texture to distinguish magnetic grains from nonmagnetic regions.

A final structural-image brightness filter is applied to the detected objects.

### Segmentation limitations

Segmentation quality depends on structural contrast, grain boundaries, and imaging conditions.

SAM is a general-purpose segmentation model rather than a model trained exclusively on NdFeB microstructures.

Consequently, some grains may be missed, incorrectly separated, or merged with neighboring grains.

Users should visually inspect segmentation overlays before relying on the extracted measurements.

---

## 7. Segmentation Outputs

The segmentation program saves results within:

```text
Segmentation/outputs/
```

Important output directories include:

### `images/`

Contains the processed microscopy images used in subsequent analysis.

### `object_mask/`

Contains the segmented grain masks saved as NumPy files (`.npy`).

Each file contains the segmentation data for an individual sample.

Example:

```text
N48_001_P.npy
```

### `visual_overlay/`

Contains images displaying detected grain boundaries over the microscopy image.

These overlays provide a visual method for evaluating segmentation quality.

The overlay colors distinguish different grain classifications and estimated magnetic alignment states.

---

## 8. Magnetic Domain Analysis

The domain analysis program is:

```text
Segmentation/raster.py
```

This program uses the grain masks generated by `main.py` to analyze magnetic domains within individual grains.

### Running the program

After completing segmentation, run:

```bash
python raster.py
```

The program loads the previously generated grain masks and corresponding processed MOKE images.

### Domain processing

For each grain, the program performs the following operations:

**1. Grain extraction**

The individual grain is isolated using its segmentation mask.

**2. Contrast enhancement**

Pixel intensities within the grain are adjusted to improve magnetic domain contrast.

**3. Gaussian filtering**

A Gaussian filter reduces image noise before thresholding.

**4. Binary thresholding**

The processed MOKE image is converted into a binary representation of the magnetic domain pattern.

The available thresholding methods include:

- Median
- Valley
- Peak midpoint

The thresholding method is selected using `THRESHOLD_MODE`.

**5. Rasterization**

The binary domain image is sampled along multiple directions:

- Horizontal
- Vertical
- Diagonal down
- Diagonal up

Raster lines are restricted to the segmented grain boundary.

**6. Domain transition analysis**

The program counts transitions between black and white pixels along the sampled lines.

The measured line lengths and transition counts are used to estimate the characteristic magnetic domain width.

**7. Grain area calculation**

The area of each segmented grain is calculated from its mask pixel count and the configured pixel calibration.

**8. Magnetic alignment estimation**

An alignment percentage is calculated by comparing the measured domain width with reference closure-domain and stripe-domain widths.

This metric is an estimate based on the selected reference values rather than a direct crystallographic orientation measurement.

---

## 9. Rasterization Settings

Important parameters in `raster.py` include:

| Parameter | Description |
|---|---|
| `BLUR_SIGMA` | Gaussian smoothing strength |
| `THRESHOLD_MODE` | Binary thresholding method |
| `THRESHOLD_OFFSET` | Adjustment to the calculated threshold |
| `MASK_ERODE_PX` | Optional erosion of grain boundaries |
| `RASTER_DIRECTIONS` | Raster sampling directions |
| `RASTER_SPACING_PX` | Spacing between raster lines |
| `RASTER_LINE_WIDTH_PX` | Raster line thickness |
| `RASTER_PX_LEN_RATIO` | Pixel-to-micrometer calibration |
| `MIN_LINE_LENGTH_PX` | Minimum accepted sampled line length |
| `MIN_LINE_TRANSITIONS` | Minimum transitions required for a sampled line |
| `MIN_MAX_LINE_TRANSITIONS` | Grain-level transition quality threshold |

The program also includes binary-image quality checks that reject grains with excessive white or black pixel fractions.

**Important:** The raster pixel calibration is configured separately from the segmentation pixel calibration. Both must correspond to the actual processed image resolution.

---

## 10. Rasterization Outputs

Rasterization results are saved to:

```text
Segmentation/outputs/rast_results/
```

Each sample receives its own output directory.

The program generates several diagnostic images and data files.

### Individual grain outputs

These include:

- Extracted grain images
- Binary domain images
- Raster overlays
- Raster coordinates and analysis CSV files
- Contrast and thresholding diagnostic plots
- Intensity histograms

These outputs allow users to inspect the domain measurement process and identify potential processing errors.

### Sample-level outputs

Each sample receives a metadata CSV containing grain-level measurements and sample-level quantities.

The program also generates domain-width and alignment heatmaps when valid measurements are available.

### Batch summary

The file:

```text
outputs/rast_results/batch_summary.csv
```

contains aggregated sample-level results.

Reported features include:

- Mean magnetic domain width
- Median magnetic domain width
- Domain width standard deviation
- Minimum and maximum domain widths
- Mean and median grain areas
- Grain area standard deviation
- Mean and median magnetic alignment
- Grain count
- Grain density
- Total segmented grain area
- Aligned grain area

These features can subsequently be used in statistical analysis or machine learning.

**Note:** The current implementation calculates its reported grain statistics from grains that pass the domain-analysis acceptance criteria. Rejected grains are not necessarily included in the final statistics.

---

## 11. Recommended Workflow

For a new dataset:

1. Acquire corresponding structural and MOKE images.
2. Name the images according to the required convention.
3. Place the images in the input directory configured in `main.py`.
4. Verify the SAM checkpoint path and pixel calibration.
5. Run `main.py`.
6. Inspect the grain segmentation overlays.
7. Adjust segmentation parameters if necessary.
8. Run `raster.py`.
9. Inspect the binary images and raster overlays.
10. Review the generated metadata and batch summary CSV files.

Segmentation and domain-analysis results should be visually validated before using them for quantitative comparisons or machine learning.

---

## 12. Additional Notes

### Image calibration

Accurate physical measurements require correct pixel calibration.

Resizing an image changes its effective physical size per pixel. Calibration values must therefore correspond to the processed images rather than automatically assuming the original acquisition resolution.

### Reproducibility

Results depend on the selected preprocessing, segmentation, and rasterization parameters.

For reproducibility, users should document parameter changes and retain the corresponding diagnostic outputs.

### Machine learning

A separate machine learning pipeline uses the extracted image and metadata features to estimate magnet grades.

The machine learning architecture, training procedure, and inference instructions are documented separately.

---

## 13. Machine Learning Overview

The machine learning (ML) pipeline uses the microscopy images and quantitative measurements generated by the segmentation and rasterization programs to predict the grade of sintered NdFeB permanent magnets.

The ML program is located in:

```text
ML/
```

The model combines two types of inputs:

1. **Microscopy images:** Low-contrast (LC), high-contrast (HC), and polarized MOKE (P) images.
2. **Numerical metadata:** Magnetic domain widths, grain areas, alignment estimates, and other sample-level measurements.

The model uses a multimodal deep learning architecture consisting of:

- **ResNet18 convolutional neural network (CNN):** Extracts visual features from the microscopy images.
- **Multilayer perceptron (MLP):** Processes numerical metadata.
- **Fusion classifier:** Combines the image and metadata features to determine the most likely magnet grade.

The program also supports metadata-only classification, allowing the image-processing branch to be disabled.

The ML pipeline contains three primary programs:

| Program | Function |
|---|---|
| `train.py` | Trains the magnet grade classification model |
| `predict.py` | Uses a trained model to predict magnet grades |
| `explain.py` | Optional model evaluation and quality-control utility |

The model supports classification among the magnet grades represented in its training dataset.

---

## 14. Machine Learning Input Preparation

### Directory structure

The ML directory is organized as follows:

```text
ML/
├── inputs/
│   ├── grain_img/
│   └── metadata/
│       ├── metadata.csv
│       └── metadata_predict.csv
│
├── models/
│   └── best_model.pth
│
├── outputs/
│   └── predictions.csv
│
└── src/
    ├── dataset.py
    ├── model.py
    ├── train.py
    ├── predict.py
    └── explain.py
```

The input and output directories may initially be empty.

Users must supply the appropriate images and metadata before training or running predictions.

### Image inputs

**Important:** The ML program requires the processed microscopy images generated by the segmentation program, not the original microscope images.

During segmentation, the original images undergo preprocessing, including cropping to remove the scale bar and other excluded regions.

Consequently, the images supplied to the ML model must correspond to the processed images used during segmentation and rasterization.

These images are available in:

```text
Segmentation/outputs/images/
```

Copy the required LC, HC, and P images into:

```text
ML/inputs/grain_img/
```

The processed images can be copied directly without additional cropping.

For example:

```text
ML/inputs/grain_img/
├── N48_001_LC.png
├── N48_001_HC.png
└── N48_001_P.png
```

Each sample requires three corresponding images when image processing is enabled.

The three images must represent the same sample region.

**Do not substitute the original, uncropped microscope images for the processed segmentation outputs.**

### Image processing within the ML program

The ML dataset loader performs the following operations:

1. Loads each image in grayscale.
2. Resizes the image to 224 × 224 pixels.
3. Converts the image into a numerical tensor.
4. Combines the LC, HC, and P images into a three-channel input.

The resulting image tensor has dimensions:

```text
3 × 224 × 224
```

The three channels represent the three microscopy image types rather than conventional red, green, and blue color channels.

### Metadata inputs

The numerical metadata are obtained from the segmentation and rasterization programs.

The primary batch summary is located at:

```text
Segmentation/outputs/rast_results/batch_summary.csv
```

This file contains aggregated measurements for individual magnet samples.

For ML training, the measurements must be organized into a CSV file containing one row per sample.

The training metadata file is:

```text
ML/inputs/metadata/metadata.csv
```

The prediction metadata file is:

```text
ML/inputs/metadata/metadata_predict.csv
```

Both files use the same general structure.

### Required metadata columns

The current ML implementation uses the following columns:

| Column | Description |
|---|---|
| `uid` | Unique sample identifier |
| `low_contrast_img` | Filename of the processed LC image |
| `high_contrast_img` | Filename of the processed HC image |
| `moke_img` | Filename of the processed P image |
| `sample_avg_domain_width_um` | Mean magnetic domain width |
| `sample_median_domain_width_um` | Median magnetic domain width |
| `sample_std_domain_width_um` | Standard deviation of magnetic domain width |
| `sample_min_domain_width_um` | Minimum magnetic domain width |
| `sample_max_domain_width_um` | Maximum magnetic domain width |
| `sample_avg_grain_area_um2` | Mean grain area |
| `sample_median_grain_area_um2` | Median grain area |
| `sample_std_grain_area_um2` | Standard deviation of grain area |
| `sample_avg_alignment_pct` | Mean estimated magnetic alignment |
| `sample_median_alignment_pct` | Median estimated magnetic alignment |
| `sample_std_alignment_pct` | Standard deviation of estimated magnetic alignment |
| `sample_image_area_um2` | Analyzed image area |
| `sample_total_grain_area_um2` | Total measured grain area |
| `sample_aligned_grain_area_um2` | Total aligned grain area |
| `sample_grain_count` | Number of accepted grains |
| `sample_grain_density_grains_per_um2` | Grain density |
| `sample_grain_count_alignment_pct` | Grain-count-based alignment percentage |
| `sample_grain_area_alignment_pct` | Grain-area-based alignment percentage |
| `target_grade` | Known magnet grade used as the training label |

The numerical metadata must correspond to the same sample represented by the three microscopy images.

The `uid` column is used for sample identification and is not supplied to the model as a numerical feature.

The image filename columns identify the corresponding image files.

The `target_grade` column provides the correct classification during supervised training.

### Preparing metadata from segmentation results

After running the rasterization program:

1. Locate the generated `batch_summary.csv`.
2. Verify that each sample has a corresponding row.
3. Prepare the metadata columns required by the ML program.
4. Add the corresponding processed image filenames.
5. Add the known magnet grade in `target_grade`.
6. Save the completed CSV as `metadata.csv`.

For prediction, prepare the corresponding `metadata_predict.csv`.

The metadata column names and numerical feature ordering should remain consistent with those used during training.

**Important:** The current dataset loader expects a `target_grade` column even when running predictions. For samples whose grades are unknown, the prediction workflow must be adapted to support unlabeled inputs before removing this column. The supplied reference prediction dataset should therefore retain the column for compatibility with the current implementation.

A reference metadata dataset is provided to demonstrate the expected format.

---

## 15. Machine Learning Model Architecture

The model architecture is defined in:

```text
ML/src/model.py
```

The model contains two parallel feature-extraction branches followed by a classification network.

### Image feature extraction

The image-processing branch uses a pretrained ResNet18 CNN.

ResNet18 is a residual neural network containing convolutional layers that learn spatial features from images.

The pretrained network is adapted for the three-channel microscopy input.

Its original classification layer is removed, allowing the CNN to function as an image feature extractor.

The CNN produces a:

**512-dimensional image feature vector**

These features represent learned visual characteristics of the combined microscopy images.

They may include information associated with grain morphology, magnetic domain patterns, structural contrast, and other image characteristics.

The CNN learns these features during training rather than receiving predefined measurements of individual microstructural properties.

### Numerical metadata processing

The numerical metadata are processed using a multilayer perceptron (MLP).

The metadata branch contains two fully connected layers, each containing 32 neurons.

The architecture is:

```text
Numerical Metadata
        |
        v
Fully Connected Layer
     32 Neurons
        |
        v
       ReLU
        |
        v
Fully Connected Layer
     32 Neurons
        |
        v
       ReLU
        |
        v
 32 Metadata Features
```

The MLP learns relationships among the numerical measurements.

Its output is a:

**32-dimensional metadata feature vector**

### Feature fusion

The image and metadata feature vectors are combined through concatenation.

When image processing is enabled:

```text
Image Features:       512
Metadata Features:     32
                     ----
Combined Features:    544
```

Concatenation joins the two feature vectors into a single representation without directly averaging or modifying their values.

The resulting 544-dimensional vector is supplied to the classification network.

### Final classification

The fusion classifier contains:

1. A fully connected hidden layer containing 64 neurons.
2. A ReLU activation function.
3. An output layer containing one neuron for each magnet grade.

For seven magnet grades, the final output layer contains seven neurons.

The network produces a classification score, or logit, for each grade.

During prediction, the softmax function converts these scores into normalized class probabilities.

The grade with the highest probability is selected as the predicted magnet grade.

### Metadata-only operation

The model can also operate without the CNN image-processing branch.

In this configuration, the 32-dimensional metadata feature vector is passed directly to the classification network.

This allows users to compare classification performance with and without the additional image-derived features.

---

## 16. Model Training

The training program is:

```text
ML/src/train.py
```

Training requires a labeled metadata dataset containing samples with known magnet grades.

When image processing is enabled, the corresponding processed microscopy images must also be available.

### Training configuration

Before running the program, review the configuration in `train.py`.

**Image processing**

```python
USE_IMAGES = True
```

Options:

- `True` — Use microscopy images and numerical metadata.
- `False` — Use numerical metadata only.

When images are disabled, the CNN branch is excluded from the model.

**Training epochs**

```python
num_epochs = 20
```

An epoch represents one complete pass through the training dataset.

The number of epochs may be adjusted depending on dataset size and training performance.

### Training procedure

The program divides the available dataset into:

| Subset | Fraction |
|---|---|
| Training | 80% |
| Validation | 20% |

The split uses a fixed random seed for reproducibility.

The training procedure uses:

| Parameter | Configuration |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Loss function | Cross-entropy |
| Batch size | 4 |
| Default epochs | 20 |
| Model selection | Lowest validation loss |

During training, the model processes the supplied images and metadata, calculates classification error, and updates its trainable parameters through backpropagation.

The validation dataset is used to evaluate classification performance without updating the model weights.

### Running training

Navigate to the ML source directory:

```bash
cd ML/src
```

Run:

```bash
python train.py
```

The program reports training loss, validation loss, and validation accuracy after each epoch.

Example:

```text
Epoch 1/20 | Train Loss: ... | Val Loss: ... | Val Acc: ...
Epoch 2/20 | Train Loss: ... | Val Loss: ... | Val Acc: ...
```

The values depend on the training dataset and model configuration.

### Saved model

The trained model is saved to:

```text
ML/models/best_model.pth
```

The program retains the model checkpoint associated with the lowest validation loss.

The checkpoint contains:

- Trained model weights
- Metadata feature column names
- Magnet grade class names
- Number of metadata features
- Number of output classes
- Image-processing configuration

The checkpoint records whether the model was trained with images enabled or disabled.

This setting is automatically used during prediction.

**Important:** Running `train.py` again may overwrite the existing `best_model.pth`. Users should preserve copies of trained models they wish to retain.

### Training considerations

Training performance depends on the size, quality, and diversity of the available dataset.

Users should ensure that:

- Magnet grades are correctly labeled.
- Metadata and images correspond to the same samples.
- The training dataset represents the intended range of magnet grades.
- Validation samples are not duplicated in the training subset.
- Image acquisition conditions are reasonably consistent.

High validation accuracy does not guarantee equivalent performance on independent magnet samples.

---

## 17. Magnet Grade Prediction

The prediction program is:

```text
ML/src/predict.py
```

This program uses an existing trained model to classify new magnet samples.

**Training is not required when a compatible trained checkpoint is already available.**

### Preparing prediction inputs

Before running prediction:

**1. Prepare the images**

Copy the processed LC, HC, and P images from:

```text
Segmentation/outputs/images/
```

into:

```text
ML/inputs/grain_img/
```

Only the processed images should be used.

If the trained model uses metadata only, image files are not required.

**2. Prepare the metadata**

Place the prediction metadata CSV at:

```text
ML/inputs/metadata/metadata_predict.csv
```

Each row represents one magnet sample.

The numerical metadata must use the same features and ordering as the training dataset.

The current implementation also requires the `target_grade` column to be present.

**3. Verify the model checkpoint**

The trained model must be located at:

```text
ML/models/best_model.pth
```

The checkpoint must be compatible with the metadata and magnet grades used by the prediction program.

### Running prediction

From the ML source directory, run:

```bash
python predict.py
```

The program automatically loads the trained model and determines whether images are required.

For example, if the model was trained using:

```python
USE_IMAGES = False
```

the saved checkpoint records this configuration.

The prediction program then automatically uses metadata-only classification.

Users do not need to manually change the image-processing setting in `predict.py`.

### Prediction procedure

For each sample, the program:

1. Loads the corresponding metadata.
2. Loads the processed microscopy images when required.
3. Passes the inputs through the trained model.
4. Generates classification scores for the available magnet grades.
5. Converts the scores into class probabilities.
6. Selects the grade with the highest probability.
7. Saves the prediction results.

Each sample receives one final magnet grade prediction.

---

## 18. Machine Learning Outputs

Prediction results are saved to:

```text
ML/outputs/predictions.csv
```

The CSV contains one row per sample.

### Prediction results

The output includes:

| Column | Description |
|---|---|
| `uid` | Sample identifier |
| `predicted_grade` | Magnet grade selected by the model |
| `confidence` | Softmax probability assigned to the predicted grade |
| `prob_N35` | Probability assigned to N35 |
| `prob_N35AH` | Probability assigned to N35AH |
| `prob_N42` | Probability assigned to N42 |
| `prob_N42SH` | Probability assigned to N42SH |
| `prob_N48` | Probability assigned to N48 |
| `prob_N52` | Probability assigned to N52 |
| `prob_N52SH` | Probability assigned to N52SH |

The available probability columns depend on the grades represented in the trained model.

### Example prediction

An illustrative prediction might be:

```text
Sample: N48_001

Predicted Grade: N48

N35:    0.01
N35AH:  0.01
N42:    0.05
N42SH:  0.02
N48:    0.82
N52:    0.07
N52SH:  0.02
```

The probabilities sum to approximately 1.0.

In this example, the model assigns the highest probability to N48.

The reported confidence would therefore be:

```text
0.82
```

**Important:** These values are model-generated classification probabilities. They should not automatically be interpreted as calibrated statistical probabilities that the prediction is correct.

A reference prediction output is provided to demonstrate the expected format.

---

## 19. Optional Model Evaluation

An additional model evaluation program is available:

```text
ML/src/explain.py
```

This program was developed as a quality-control and diagnostic tool for evaluating the trained classification model.

It uses permutation-based feature importance to estimate how strongly the model's classification accuracy depends on individual numerical metadata features.

When image processing is enabled, the combined image input can also be evaluated as a feature group.

The program is provided for users interested in examining model behavior.

**Running `explain.py` is not required for model training or magnet grade prediction.**

Feature-importance results should be interpreted as diagnostic estimates rather than direct measurements of physical causality.

---

## 20. Complete Segmentation and Machine Learning Workflow

The complete repository supports a workflow beginning with microscopy images and ending with magnet grade classification.

### Stage 1: Image acquisition

Acquire three corresponding microscopy images:

```text
LC
HC
P
```

Ensure that the images represent the same sample region.

### Stage 2: Grain segmentation

Run:

```bash
cd Segmentation
python main.py
```

The program identifies grains, generates segmentation masks, and saves processed microscopy images.

Inspect the segmentation overlays to verify that grain identification is reasonable.

### Stage 3: Magnetic domain analysis

Run:

```bash
python raster.py
```

The program uses the segmentation masks to calculate magnetic domain widths, grain areas, and estimated magnetic alignment.

Review the resulting diagnostic images and metadata.

### Stage 4: Prepare ML inputs

Copy the processed microscopy images from:

```text
Segmentation/outputs/images/
```

to:

```text
ML/inputs/grain_img/
```

Prepare the corresponding sample-level metadata CSV using the rasterization results.

Save the metadata to:

```text
ML/inputs/metadata/
```

### Stage 5: Train the model

If a compatible trained model is not available, navigate to:

```text
ML/src/
```

and run:

```bash
python train.py
```

The trained checkpoint is saved to:

```text
ML/models/best_model.pth
```

This stage can be skipped when using an existing compatible model.

### Stage 6: Predict magnet grades

Prepare:

```text
ML/inputs/metadata/metadata_predict.csv
```

Run:

```bash
python predict.py
```

The program generates a magnet grade prediction for each sample.

### Stage 7: Review the results

Open:

```text
ML/outputs/predictions.csv
```

Review the predicted magnet grades and their associated class probabilities.

Where known reference grades are available, compare the predictions against the correct classifications.

---

## 21. Additional Machine Learning Notes

### Reference data

The repository may include example images, metadata, and prediction outputs to demonstrate the required input and output formats.

These reference files are intended to help users understand the expected directory structure and prepare their own datasets.

Users should replace the example inputs with their own processed microscopy images and corresponding measurements when performing new analyses.

### Model compatibility

A trained model is associated with the numerical features and magnet grade classes used during training.

Prediction inputs must be compatible with the saved checkpoint.

Changing the metadata features, their ordering, or the set of magnet grades may require retraining or modifications to the prediction pipeline.

### Image preprocessing consistency

The ML model should receive images prepared using the same segmentation preprocessing procedure applied to the training data.

In particular, the processed images from `Segmentation/outputs/images/` should be used instead of the original microscope images.

Differences in cropping, magnification, illumination, or image processing can influence classification performance.

### Classification limitations

The model learns statistical relationships between the supplied microscopy images, numerical metadata, and known magnet grades.

It does not directly measure intrinsic magnetic properties such as coercivity or remanence.

The predicted grade is an estimate based on the features learned from the training dataset.

Performance may be reduced for samples with substantially different microstructures, manufacturing conditions, or imaging characteristics.

### Reproducibility

For reproducibility, users should retain:

- The trained model checkpoint
- The metadata used during training
- The corresponding processed microscopy images
- The model configuration and training parameters
- The segmentation and rasterization settings

Model predictions should be evaluated against independent samples whenever possible.

---

## 13. Acknowledgments

This project uses the Segment Anything Model (SAM), developed by Meta AI Research, for automated image segmentation.

Users should consult the original SAM publication and repository for model documentation, licensing, and citation information.

