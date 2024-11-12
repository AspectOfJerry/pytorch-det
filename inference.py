import cv2
import torch
import torchinfo

from cc import cc
from modules.model import new_model
from modules.utils import get_device

"""
Run configuration
"""
# Confidence threshold
MODEL_PATH = "output/inference_graph.pth"
MIN_PRED_CONF = 0.3

# Label indices start from 1, 0 is reserved for the background class
label_map = {
    1: "note"
}
NUM_CLASSES = len(label_map)  # automatically calculated

# Colors for bounding boxes
box_color_map = {
    "high": (0, 255, 0),  # Green
    "medium": (0, 255, 255),  # Yellow
    "low": (0, 0, 255)  # Red
}

"""
Creating the model
"""
print(cc("YELLOW", "Initializing model..."))
model = new_model(out_features=NUM_CLASSES + 1)  # add 1 for the background class
# Load the trained model
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

"""
Configuring devices
"""
print(cc("YELLOW", "Configuring devices..."))
DEVICE = get_device(torch.cuda.is_available())
print(cc("GRAY", "-------------------------"))

# Move model to configured device
model.to(DEVICE)

"""
Image inference
"""
# Image path
image = cv2.imread("dataset/images/test/000000000457.jpeg")  # satisfactory
# image = cv2.imread("dataset/images/test/000000000454.jpeg")  # passing
# image = cv2.imread("dataset/images/test/000000000489.jpeg")  # unsatisfactory
# image = cv2.imread("dataset/images/test/000000000455.jpeg")  # failing

# Normalize the image
# image = cv2.resize(image, (512, 512))  # Resize to match the input size
image = image / 255.0  # Normalize the image to values between 0 and 1
image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

# Inference
with torch.no_grad():
    # Forward pass
    output = model(image)

# Extract bboxes, labels, and scores
boxes = output[0]["boxes"]
labels = output[0]["labels"]
scores = output[0]["scores"]

# Convert image back to NumPy format
image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
scale_factor = 512 / max(image.shape[0], image.shape[1])

print(cc("CYAN", f"Predictions above {MIN_PRED_CONF} confidence score:"))

# Draw the bboxes
for box, label, score in zip(boxes, labels, scores):
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
    elif score >= MIN_PRED_CONF:
        color = box_color_map["low"]
        print(cc("RED", f"Label: {label_name}, Score: {score}"))
    else:
        continue

    cv2.rectangle(image, (x, y), (x_max, y_max), color, 2)
    cv2.putText(image, f"Label: {label_name}, Score: {score}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the image
cv2.imshow("Image with Predictions", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
Video inference (unfinished)
"""

exit()

# change the id if needed (multiple cameras, 0 default)
cap = cv2.VideoCapture(0)

previousT = 0
currentT = 0

while True:
    success, frame = cap.read()

    if not success:
        break

    # model_frame = frame.copy()

    # Resize image to fit within the display window
    scaleFactorX = 640 / frame.shape[1]
    scaleFactorY = 480 / frame.shape[0]

    frame = cv2.resize(frame, (0, 0), fx=scaleFactorX, fy=scaleFactorY)

    # Normalize
    model_frame = frame / 255.0
    model_frame = torch.tensor(model_frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(model_frame)

    boxes = output[0]["boxes"]
    labels = output[0]["labels"]
    scores = output[0]["scores"]

    print(scores)

    for box, label, score in zip(boxes, labels, scores):
        box = box.int()
        x, y, x_max, y_max = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        # Scale the bounding box coordinates
        x = int(x * scaleFactorX)
        y = int(y * scaleFactorY)
        x_max = int(x_max * scaleFactorX)
        y_max = int(y_max * scaleFactorY)

        label_id = int(label)
        label_name = label_map.get(label_id, f"Label {label_id}")
        score = round(score.item(), 2)

        if score <= 0.4:
            continue
        print(score)

        if score >= 0.8:
            color = box_color_map["high"]
            cv2.rectangle(frame, (x, y), (x_max, y_max), color, 3)
            cv2.putText(frame, f"Label: {label_name}, Score: {score}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        elif score >= 0.5:
            color = box_color_map["medium"]
            cv2.rectangle(frame, (x, y), (x_max, y_max), color, 2)
            cv2.putText(frame, f"Label: {label_name}, Score: {score}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # else:
        #     color = color_mapping["low"]
        #     cv2.rectangle(frame, (x, y), (x_max, y_max), color, 1)

    currentT = time.time()
    fps = 1 / (currentT - previousT)
    previousT = currentT

    cv2.putText(frame, str(round(fps, 4)) + " fps", (20, 40), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    cv2.imshow("Object Detection [cone, cube]", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
