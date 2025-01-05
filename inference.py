import cv2
import torch
import torchinfo

import modules.model as models
import modules.utils as utils
from cc import cc

# Configuration
MODEL_PATH = "output/models/inference_graph.pth"
MIN_PRED_CONF = 0.3
box_color_map = {"high": (0, 255, 0), "medium": (0, 255, 255), "low": (0, 0, 255)}
label_map = {
    1: "note"
}
NUM_CLASSES = len(label_map)

# Initialize model
print(cc("YELLOW", "Initializing model..."))
model = models.new_model(out_features=NUM_CLASSES + 1)  # +1 for background class
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Model summary
print(cc("GRAY", "Model summary:"))
print(cc("GRAY", str(torchinfo.summary(
    model,
    input_size=(1, 3, 512, 512),
    verbose=0,
    col_names=("input_size", "output_size", "num_params", "mult_adds"),
    row_settings=["var_names"]
))))

# Configure device
print(cc("YELLOW", "Configuring devices..."))
DEVICE = utils.get_device(torch.cuda.is_available())
model.to(DEVICE)

# Image inference
image_path = "dataset/images/test/000000000457.jpeg"
image = cv2.imread(image_path)
image_resized = cv2.resize(image, (512, 512))
image_normalized = image_resized / 255.0
image_tensor = torch.tensor(image_normalized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

# Inference
with torch.no_grad():
    output = model(image_tensor)

# Extract bboxes, labels, and scores
boxes = output[0]["boxes"]
labels = output[0]["labels"]
scores = output[0]["scores"]

scale_factor = 512 / max(image.shape[0], image.shape[1])  # scale factor for resizing the bounding boxes

print(cc("CYAN", f"Predictions above {MIN_PRED_CONF} confidence score:"))
for box, label, score in zip(boxes, labels, scores):
    if score < MIN_PRED_CONF:
        continue

    box = box.int()
    # Scale the bounding box coordinates because the image was resized
    x, y, x_max, y_max = int(box[0] * scale_factor), int(box[1] * scale_factor), int(box[2] * scale_factor), int(box[3] * scale_factor)
    label_id = int(label)
    label_name = label_map.get(label_id, f"Label {label_id}")
    score = round(score.item(), 2)

    if score >= 0.8:
        color = box_color_map["high"]
        print(cc("GREEN", f"Label: {label_name}, Score: {score}"))
    elif score >= 0.5:
        color = box_color_map["medium"]
        print(cc("YELLOW", f"Label: {label_name}, Score: {score}"))
    else:
        color = box_color_map["low"]
        print(cc("RED", f"Label: {label_name}, Score: {score}"))

    cv2.rectangle(image, (x, y), (x_max, y_max), color, 2)
    cv2.putText(image, f"Label: {label_name}, Score: {score}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the image
cv2.imshow("Image with Predictions", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
