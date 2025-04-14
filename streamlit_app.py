import streamlit as st
import torch
from PIL import Image
import io
import numpy as np
import cv2

# Load pre-trained YOLOv5 model (upgrade to yolov5m)
@st.cache_resource
def load_model():
    model = torch.hub.load("ultralytics/yolov5", "yolov5m", pretrained=True)
    return model

model = load_model()

# Title of the app
st.title("Seat Detection")

# Instructions
st.write("Use your camera to take a picture, and the model will detect empty and occupied chairs.")

# Camera input widget
picture = st.camera_input("Take a picture")

# Function to calculate Intersection over Union (IoU)
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    xi1 = max(x1, x3)
    yi1 = max(y1, y3)
    xi2 = min(x2, x4)
    yi2 = min(y2, y4)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# Function to calculate distance between box centers
def calculate_center_distance(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    center1_x, center1_y = (x1 + x2) / 2, (y1 + y2) / 2
    center2_x, center2_y = (x3 + x4) / 2, (y3 + y4) / 2
    return np.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)

# Process the image with YOLOv5
if picture is not None:
    # Display the captured image
    st.image(picture, caption="Captured Image", use_container_width=True)

    # Convert Streamlit's BytesIO to PIL Image
    img = Image.open(picture)

    # Run inference
    results = model(img)
    detections = results.pandas().xyxy[0]

    # Filter detections
    chairs = detections[detections['name'] == 'chair']
    people = detections[detections['name'] == 'person']
    belongings = detections[detections['name'].isin(['backpack', 'handbag', 'bottle', 'suitcase', 'book', 'laptop'])]

    # Classify chairs as empty or occupied
    chair_status = {}
    for chair_idx, chair in chairs.iterrows():
        chair_box = [chair['xmin'], chair['ymin'], chair['xmax'], chair['ymax']]
        is_occupied = False

        # Check for person overlap (sitting) or proximity
        for _, person in people.iterrows():
            person_box = [person['xmin'], person['ymin'], person['xmax'], person['ymax']]
            iou = calculate_iou(chair_box, person_box)
            distance = calculate_center_distance(chair_box, person_box)
            chair_width = chair_box[2] - chair_box[0]
            # Lower IoU threshold and add proximity check
            if iou > 0.3 or (distance < chair_width * 1.5 and iou > 0.1):
                is_occupied = True
                break

        # Check for belongings overlap
        if not is_occupied:
            for _, belonging in belongings.iterrows():
                belonging_box = [belonging['xmin'], belonging['ymin'], belonging['xmax'], belonging['ymax']]
                iou = calculate_iou(chair_box, belonging_box)
                if iou > 0.3:
                    is_occupied = True
                    break

        chair_status[chair_idx] = "Occupied" if is_occupied else "Empty"

    # Display results
    if not chairs.empty:
        st.write("Chair Status:")
        for chair_idx, status in chair_status.items():
            chair = chairs.loc[chair_idx]
            st.write(f"- Chair at ({int(chair['xmin'])}, {int(chair['ymin'])}): {status} (Confidence: {chair['confidence']:.2f})")
    else:
        st.write("No chairs detected in the image.")

    # Render image with custom labels
    img_array = np.array(img)  # PIL Image to numpy array (RGB)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for cv2
    for chair_idx, status in chair_status.items():
        chair = chairs.loc[chair_idx]
        xmin, ymin, xmax, ymax = int(chair['xmin']), int(chair['ymin']), int(chair['xmax']), int(chair['ymax'])
        color = (0, 0, 255) if status == "Occupied" else (0, 255, 0)  # Red for occupied, green for empty
        cv2.rectangle(img_array, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(img_array, status, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the image with detections
    st.image(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB), caption="Image with Chair Status", use_container_width=True)