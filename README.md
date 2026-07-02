# Age Estimation from Face Images using ResNet-50, DeiT, and Smart Fusion

This repository contains the code sample for my bachelor's thesis project on age estimation from facial images. The project compares convolutional and transformer-based models and then combines their trained feature representations using a smart fusion head.

## Project summary

The task is to estimate a person's age from a face image using the UTKFace dataset. The project is implemented in two prediction settings:

1. **Direct regression**: the model predicts age as a continuous value and is trained with L1/MAE loss.
2. **Regression through classification**: ages 1 to 100 are treated as classes, and the predicted class is converted back to an age for MAE evaluation.

The main models are:

- **ResNet-50** for local convolutional facial features.
- **DeiT** (`facebook/deit-base-distilled-patch16-224`) for transformer-based facial context.
- **Smart fusion**, which concatenates the trained ResNet-50 and DeiT feature vectors and trains a new prediction head.

## Folder structure

```text
Age-Estimation-Using-Vision-Transformers/
├── bad_photos/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_regression_resnet_kfold.ipynb
│   ├── 02_regression_deit_kfold.ipynb
│   ├── 03_regression_resnet_deit_smart_fusion_kfold.ipynb
│   ├── 04_regression_through_classification_resnet_kfold.ipynb
│   ├── 05_regression_through_classification_deit_kfold.ipynb
│   └── 06_regression_through_classification_smart_fusion_kfold.ipynb
└── preprocessing/
    ├── distribution.py
    ├── remove_bad.py
    ├── separate_ages.py
    └── split_folds.py
```

The dataset folders are not included in the repository. They should be created locally when running the code.

## Dataset

The code uses UTKFace-style image names, where the age is the first part of the filename:

```text
age_gender_race_date.jpg
```

Example:

```text
25_0_2_20170116174525125.jpg
```

The UTKFace dataset is not included in this repository. Before preprocessing, place the original aligned UTKFace images directly inside:

```text
UTKFace_aligned/
```

At this stage, the folder can look like this:

```text
UTKFace_aligned/
├── 1_0_0_20161219140623097.jpg
├── 25_0_2_20170116174525125.jpg
├── 40_1_0_20170117163224512.jpg
└── ...
```

Then `separate_ages.py` organizes those images into age folders. After that preprocessing step, the folder becomes:

```text
UTKFace_aligned/
├── 1/
├── 2/
├── 3/
└── ...
```

If the images are already organized into age folders, skip `separate_ages.py` and go directly to `split_folds.py`.

The K-fold train/test folders are written separately to:

```text
UTKFace_folds/
```

This keeps the original age-organized images separate from the generated fold folders.

## Installation

Run this from the repository root:

```bash
pip install -r requirements.txt
```

## Preprocessing

### 1. Organize images by age

Run this only if the images are still directly inside `UTKFace_aligned/`:

```bash
python preprocessing/separate_ages.py --source_dir UTKFace_aligned
```

This moves each image into an age folder based on the first value in the filename.

### 2. Remove bad images, if needed

The manually selected bad images should be placed inside a folder named `bad_photos/` inside `UTKFace_aligned/`. They can be placed directly inside `bad_photos/` or organized into subfolders such as `bad_quality/`, `fake_pics/`, and `no_face/`.

```bash
python preprocessing/remove_bad.py --base_dir UTKFace_aligned --bad_dir_name bad_photos
```

The script removes matching filenames from the age folders.

### 3. Check the age distribution, if needed

```bash
python preprocessing/distribution.py --base_dir UTKFace_aligned --csv_path data.csv --plot_path age_distribution.png
```

This saves a CSV file and a distribution plot.

### 4. Create 5-fold train/test folders

```bash
python preprocessing/split_folds.py --source_dir UTKFace_aligned --folds_dir UTKFace_folds --k 5 --seed 42 --min_age 1 --max_age 100
```

This reads the age folders from `UTKFace_aligned/` and creates:

```text
UTKFace_folds/
├── fold_0/
│   ├── train/
│   └── test/
├── fold_1/
│   ├── train/
│   └── test/
├── fold_2/
│   ├── train/
│   └── test/
├── fold_3/
│   ├── train/
│   └── test/
└── fold_4/
    ├── train/
    └── test/
```

The notebooks expect this exact folder name:

```python
FOLDS_ROOT = Path("./UTKFace_folds")
```

## Manually selected bad photos

These are the bad photos mentioned in the dataset cleaning step. They were hand-picked by me and placed in a `bad_photos/` folder during dataset cleaning. The preprocessing script `remove_bad.py` removes these images from the age-organized dataset before creating the final K-fold training and validation folders.

The images are grouped by the reason they were excluded. These groups are used for documentation. If the images are stored in category subfolders inside `bad_photos/`, the removal script still finds them because it searches the folder recursively.

### Summary

- `bad_quality/`: 45 images
- `fake_pics/`: 3 images
- `no_face/`: 40 images

## Recommended run order

Run the notebooks from the repository root so the relative paths work correctly.

### Direct regression pipeline

1. `notebooks/01_regression_resnet_kfold.ipynb`
2. `notebooks/02_regression_deit_kfold.ipynb`
3. `notebooks/03_regression_resnet_deit_smart_fusion_kfold.ipynb`

Notebook 03 loads the fold-specific checkpoints generated by notebooks 01 and 02.

### Regression-through-classification pipeline

4. `notebooks/04_regression_through_classification_resnet_kfold.ipynb`
5. `notebooks/05_regression_through_classification_deit_kfold.ipynb`
6. `notebooks/06_regression_through_classification_smart_fusion_kfold.ipynb`

Notebook 06 loads the fold-specific checkpoints generated by notebooks 04 and 05.

## Checkpoints and outputs

The notebooks save results under:

```text
outputs/
checkpoints/
```

The fusion notebooks use fold-specific checkpoint paths such as:

```text
checkpoints/regression_deit_kfold/fold_0_best.pth
checkpoints/regression_resnet_kfold/fold_0_best.pth
```

The fusion model must load the base model checkpoint from the same fold. For example, fold 0 fusion should use fold 0 ResNet and fold 0 DeiT checkpoints.

## Training setup

All notebooks use 5-fold cross-validation. Each fold is trained for a maximum of 200 epochs with early stopping patience of 7 based on validation MAE.

The base model settings follow the best-performing models selected in the thesis tables:

- Direct regression ResNet-50: last 39 convolutional layers fine-tuned.
- Direct regression DeiT: last 3 transformer blocks fine-tuned.
- Regression-through-classification ResNet-50: last 48 convolutional layers fine-tuned.
- Regression-through-classification DeiT: last 1 transformer block fine-tuned.

## Main architecture details

### Direct regression fusion

- DeiT feature vector: 384 dimensions
- ResNet-50 feature vector: 512 dimensions
- Concatenated feature vector: 896 dimensions
- Fusion head: `896 -> 512 -> 256 -> 1`
- Loss: L1 Loss / MAE
- Evaluation metric: MAE

### Regression-through-classification fusion

- DeiT feature vector: 768 dimensions
- ResNet-50 feature vector: 512 dimensions
- Concatenated feature vector: 1280 dimensions
- Fusion head: `1280 -> 640 -> 100`
- Loss: Cross-Entropy
- Evaluation metric: MAE after converting the predicted class to age

## Notes

- The dataset and trained checkpoints are not included in this repository.
- The preprocessing scripts use ages 1 to 100 by default because the classification notebooks use 100 age classes.
- The K-fold split is stratified by age folder, so each fold keeps the age distribution as balanced as possible.
