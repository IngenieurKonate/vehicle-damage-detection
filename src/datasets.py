#============================
# Ce code sert à faire le lien entre les données prétraitées et PyTorch.
#==========================
import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

class CarDDDataset(Dataset):
    def __init__(self, images_dir, annotations_path, transform=None):
        self.images_dir = images_dir
        self.transform = transform

        with open(annotations_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        # images
        self.images = coco["images"]

        # image_id -> annotations
        from collections import defaultdict
        self.img_to_anns = defaultdict(list)
        for ann in coco["annotations"]:
            self.img_to_anns[ann["image_id"]].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.images_dir, img_info["file_name"])

        image = Image.open(img_path).convert("RGB")

        anns = self.img_to_anns.get(img_info["id"], [])

        # bbox: (cx, cy, w, h) normalisées
        boxes = torch.tensor([ann["bbox"] for ann in anns], dtype=torch.float32)
        labels = torch.tensor([ann["category_id"] for ann in anns], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, boxes, labels
