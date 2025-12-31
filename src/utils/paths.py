"""
Chemins centralisés pour Google Drive.
À importer dans tous les notebooks et scripts.
"""

# Racine Google Drive (après mount)
DRIVE_ROOT = "/content/drive/MyDrive"

# Projet
PROJECT_ROOT = f"{DRIVE_ROOT}/ENSA_Deep_Learning"

# Datasets
DATASETS_DIR = f"{PROJECT_ROOT}/datasets"
RAW_DATA_DIR = f"{DATASETS_DIR}/raw"
PROCESSED_DATA_DIR = f"{DATASETS_DIR}/processed"

# Datasets bruts
CARDD_DIR = f"{RAW_DATA_DIR}/CarDD_release/CarDD_COCO"
STANFORD_DIR = f"{RAW_DATA_DIR}/stanford_cars_224/car_data"

# Splits processed
TRAIN_DIR = f"{PROCESSED_DATA_DIR}/train"
VAL_DIR = f"{PROCESSED_DATA_DIR}/val"
TEST_DIR = f"{PROCESSED_DATA_DIR}/test"

# Checkpoints
CHECKPOINTS_DIR = f"{PROJECT_ROOT}/checkpoints"
MODEL_A_CKPT = f"{CHECKPOINTS_DIR}/model_a"
MODEL_B_CKPT = f"{CHECKPOINTS_DIR}/model_b"

# Outputs
OUTPUTS_DIR = f"{PROJECT_ROOT}/outputs"
FIGURES_DIR = f"{OUTPUTS_DIR}/figures"
LOGS_DIR = f"{OUTPUTS_DIR}/logs"
