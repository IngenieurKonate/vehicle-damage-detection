#===============================================
# Ce code permet de redimentionner les images dans le dataset
#==============================================

import os
import json
from PIL import Image

# ========================
# CONFIG
# ========================
RAW_IMG_DIR = "D:/Users/HP/Bureau/vehicle-damage-detection/data/raw/CarDD_COCO/train2017"
RAW_ANN_PATH = "D:/Users/HP/Bureau/vehicle-damage-detection/data/raw/CarDD_COCO/annotations/instances_train2017.json"

OUT_IMG_DIR = "D:/Users/HP/Bureau/vehicle-damage-detection/data/processed/images_train"
OUT_ANN_PATH = "D:/Users/HP/Bureau/vehicle-damage-detection/data/processed/annotations_train.json"
TARGET_SIZE = 512

os.makedirs(OUT_IMG_DIR, exist_ok=True)

# ========================
# LOAD ANNOTATIONS
# ========================
with open(RAW_ANN_PATH, "r", encoding="utf-8") as f:
    coco = json.load(f)

# Build image_id -> image info
image_id_to_info = {img["id"]: img for img in coco["images"]}

# Group annotations by image
from collections import defaultdict
image_to_annotations = defaultdict(list)
for ann in coco["annotations"]:
    image_to_annotations[ann["image_id"]].append(ann)

new_annotations = []
new_images = []

ann_id = 1

# ========================
# PROCESS IMAGES
# ========================
for image_id, anns in image_to_annotations.items():
    img_info = image_id_to_info[image_id]
    img_path = os.path.join(RAW_IMG_DIR, img_info["file_name"])

    if not os.path.exists(img_path):
        continue

    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size

    # Resize with aspect ratio
    scale = min(TARGET_SIZE / orig_w, TARGET_SIZE / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    # Padding
    padded_img = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), (0, 0, 0))
    pad_x = (TARGET_SIZE - new_w) // 2
    pad_y = (TARGET_SIZE - new_h) // 2
    padded_img.paste(img_resized, (pad_x, pad_y))

    # Save image
    padded_img.save(os.path.join(OUT_IMG_DIR, img_info["file_name"]))

    # Save image info
    new_images.append({
        "id": image_id,
        "file_name": img_info["file_name"],
        "width": TARGET_SIZE,
        "height": TARGET_SIZE
    })

    # Process annotations
    for ann in anns:
        x, y, w, h = ann["bbox"]

        # Convert to center
        cx = x + w / 2
        cy = y + h / 2

        # Resize + pad
        cx = cx * scale + pad_x
        cy = cy * scale + pad_y
        w = w * scale
        h = h * scale

        # Normalize
        cx /= TARGET_SIZE
        cy /= TARGET_SIZE
        w /= TARGET_SIZE
        h /= TARGET_SIZE

        new_annotations.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": ann["category_id"],
            "bbox": [cx, cy, w, h],  # normalized center-based
            "iscrowd": 0
        })

        ann_id += 1

# ========================
# SAVE NEW COCO FILE
# ========================
processed_coco = {
    "images": new_images,
    "annotations": new_annotations,
    "categories": coco["categories"]
}

with open(OUT_ANN_PATH, "w", encoding="utf-8") as f:
    json.dump(processed_coco, f, indent=2)

print("Preprocessing terminé.")
print(f"Images sauvegardées dans : {OUT_IMG_DIR}")
print(f"Annotations sauvegardées dans : {OUT_ANN_PATH}")
