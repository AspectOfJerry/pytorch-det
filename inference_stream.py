import time

import cv2
import torch

import modules.model as models
import modules.utils as utils
from cc import cc

# Configuration
MODEL_PATH = "output/inference_graph.pth"
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

# Configure device
print(cc("YELLOW", "Configuring devices..."))
DEVICE = utils.get_device(torch.cuda.is_available())
model.to(DEVICE)

# Video capture
cap = cv2.VideoCapture(0)
previousT = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    # Resize and normalize frame
    scaleFactorX = 640 / frame.shape[1]
    scaleFactorY = 480 / frame.shape[0]
    frame = cv2.resize(frame, (0, 0), fx=scaleFactorX, fy=scaleFactorY)
    model_frame = frame / 255.0
    model_frame = torch.tensor(model_frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        output = model(model_frame)

    # Extract bboxes, labels, and scores
    boxes = output[0]["boxes"]
    labels = output[0]["labels"]
    scores = output[0]["scores"]

    # Draw bboxes
    for box, label, score in zip(boxes, labels, scores):
        if score < MIN_PRED_CONF:
            continue

        box = box.int()
        x, y, x_max, y_max = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        x = int(x * scaleFactorX)
        y = int(y * scaleFactorY)
        x_max = int(x_max * scaleFactorX)
        y_max = int(y_max * scaleFactorY)

        label_id = int(label)
        label_name = label_map.get(label_id, f"Label {label_id}")
        score = round(score.item(), 2)

        if score >= 0.8:
            color = box_color_map["high"]
        elif score >= 0.5:
            color = box_color_map["medium"]
        else:
            color = box_color_map["low"]

        cv2.rectangle(frame, (x, y), (x_max, y_max), color, 2)
        cv2.putText(frame, f"Label: {label_name}, Score: {score}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Calculate FPS
    currentT = time.time()
    fps = 1 / (currentT - previousT)
    previousT = currentT
    cv2.putText(frame, f"{fps:.2f} fps", (20, 40), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    # Display frame
    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
