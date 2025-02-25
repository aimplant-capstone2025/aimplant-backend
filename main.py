from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import base64
from ultralytics import YOLO
from fastapi.responses import JSONResponse
import shutil
import os
from io import BytesIO
import uvicorn

app = FastAPI()

# Load YOLOv8 Pose Model
model = YOLO("./models/yolov8-pose_9img_3epoch.pt")
THRESHOLD = 0
keypoint_colors = [
    (0, 0, 255),   # Red 1
    (255, 0, 0),   # Blue 2
    (0, 255, 0),   # Green 3 
    (0, 255, 255), # Yellow 4
    (0, 165, 255), # Orange 5
    (128, 0, 128)  # Purple 6
]

def encode_image_to_base64(img):
    """Convert image (NumPy array) to base64 string"""
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def skeleton_draw(img, valid_points):
    connection_ind = [(0,5), (1,4), (2,3), (0,1), (1,2), (0,2), (5,4), (4,3), (5,3)]
    line_lengths = []
    for i, j in connection_ind:
        if i < len(valid_points) and j < len(valid_points):  # Ensure index is valid
            x1, y1 = valid_points[i]
            x2, y2 = valid_points[j]

            # Draw line
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green line

            # Calculate line length
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            line_lengths.append(length)
    
    boneloss = max(line_lengths[3], line_lengths[6]) / max((line_lengths[3]+line_lengths[4]), (line_lengths[6]+line_lengths[7])) * 100

    return img, boneloss

def draw_result(result, img):
    keypoints = result.keypoints.data
    result_img = img.copy()
    pad = img.shape[1] / 100  # Padding size
    implants = []

    # Draw keypoints
    for keypoint in keypoints:
        valid_points = []
        implant_img = img.copy()

        for i in range(len(keypoint)):
            x, y, conf = keypoint[i] 
            if conf > THRESHOLD:  # Filter low-confidence keypoints
                valid_points.append((x.item(), y.item()))
                cv2.circle(result_img, (int(x), int(y)), 5, keypoint_colors[i], -1)
                cv2.circle(implant_img, (int(x), int(y)), 5, keypoint_colors[i], -1)

        if len(valid_points) < 2:  # Skip if not enough points
            continue

        # Compute bounding box
        x_min = int(max(0, min(p[0] for p in valid_points) - pad))
        y_min = int(max(0, min(p[1] for p in valid_points) - pad))
        x_max = int(min(result_img.shape[1], max(p[0] for p in valid_points) + pad))
        y_max = int(min(result_img.shape[0], max(p[1] for p in valid_points) + pad))

        # Draw skeleton
        implant_img, boneloss = skeleton_draw(implant_img, valid_points)
        implant_img = implant_img[y_min:y_max, x_min:x_max]  # Crop

        implants.append({
            "img": encode_image_to_base64(implant_img),
            "class": "Peri-implantitis",
            "boneloss": boneloss
        })

    return encode_image_to_base64(result_img), implants

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded image temporarily
        file_path = f"./uploads/{file.filename}"
        os.makedirs("./uploads", exist_ok=True)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Read image
        img = cv2.imread(file_path)
        if img is None:
            return JSONResponse(content={"error": "Invalid image format"}, status_code=400)

        # Run YOLO Pose Detection
        results = model(img)
        result = results[0]

        # Process results
        result_img_b64, implants = draw_result(result, img)

        # Delete temporary image
        os.remove(file_path)

        return JSONResponse(content={
            "result_img": result_img_b64, 
            "implants": implants
        })
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
