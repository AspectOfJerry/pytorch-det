import os
import shutil
import xml

import torch
from PIL import Image

from cc import cc


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, data_split, transform=None, device="cpu"):
        self.root_dir = root_dir
        self.data_split = data_split  # "train" and "test"
        self.image_dir = os.path.join(root_dir, "images", data_split)
        self.annotation_dir = os.path.join(root_dir, "annotations", data_split)
        self.image_files = os.listdir(self.image_dir)
        self.transform = transform
        self.device = device

        self.label_map = {
            # Label indices start from 1, 0 is reserved for the background class
            "0": 1  # "0" is the name of the class in the annotation file for notes
        }

    def __getitem__(self, idx):
        while True:
            image_file = os.path.join(self.image_dir, self.image_files[idx])
            image = Image.open(image_file).convert("RGB")

            # Load and parse the XML annotation file
            xml_file = os.path.join(self.annotation_dir, f"{os.path.splitext(self.image_files[idx])[0]}.xml")
            bounding_boxes = self.parse_xml_annotation(xml_file)

            if bounding_boxes is None:
                # Move the image and XML file to another folder
                empty_dir = os.path.join(self.root_dir, "empty")
                os.makedirs(os.path.join(empty_dir, "images"), exist_ok=True)
                os.makedirs(os.path.join(empty_dir, "annotations"), exist_ok=True)
                print(cc("RED", f"Discarding empty image {self.image_files[idx]} and its annotation file."))
                shutil.move(image_file, os.path.join(empty_dir, "images", os.path.basename(image_file)))
                shutil.move(xml_file, os.path.join(empty_dir, "annotations", os.path.basename(xml_file)))
                idx += 1
                if idx >= len(self.image_files):
                    raise IndexError("No more images to process.")
                continue

            # Create a list of bounding boxes
            target_boxes = torch.tensor([bb["boxes"] for bb in bounding_boxes], dtype=torch.float32).to(self.device)

            # Convert the list of labels to a list of class indices
            labels = [label for bb in bounding_boxes for label in bb["labels"]]
            label_indices = torch.tensor([self.label_map[label] for label in labels], dtype=torch.int64).to(self.device)

            if self.transform:
                image = self.transform(image)

            targets = {
                "boxes": target_boxes,
                "labels": label_indices,
            }

            return image, targets

    def __len__(self):
        return len(self.image_files)

    def parse_xml_annotation(self, xml_file):
        print(f"Parsing {xml_file}")
        tree = xml.etree.ElementTree.parse(xml_file)
        root = tree.getroot()

        bounding_boxes = []
        for obj in root.findall("object"):
            label = str(obj.find("name").text)
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            bounding_boxes.append({"labels": [label], "boxes": [xmin, ymin, xmax, ymax]})
            # bounding_boxes.append({"labels": ["note"], "boxes": [xmin, ymin, xmax, ymax]})

        # print(cc("GRAY", f"- Bounding boxes: {bounding_boxes}"))

        if len(bounding_boxes) == 0:
            return None
        return bounding_boxes
