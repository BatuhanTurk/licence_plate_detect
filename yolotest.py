import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import numpy as np
import re

# Load models once
vehicle_model = YOLO("./yolov8m.pt")
plate_model = YOLO("./license_plate_detector.pt")

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="en")  # English language and angle detection

def preprocess_plate_image(plate_image):
    # Convert to grayscale
    gray_plate = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred_plate = cv2.GaussianBlur(gray_plate, (3, 3), 0)

    # Apply Adaptive Threshold to binarize the image
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Apply sharpening filter to enhance the image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp_plate = cv2.filter2D(adaptive_thresh, -1, kernel)

    return sharp_plate


def is_valid_turkey_plate(plate_text):
    # Türkiye plaka formatlarını kontrol eden bir regex deseni
    pattern = r"^[0-9]{2}[A-Z]{1,3}[0-9]{1,4}$|^[0-9]{2} [A-Z]{1,3} [0-9]{1,4}$"
    return re.match(pattern, plate_text) is not None


def read_license_plate(plate_image):
    # Preprocess the plate image (Optional: if needed for OCR)
    sharp_plate = preprocess_plate_image(plate_image)

    # Use PaddleOCR to read the plate text
    ocr_result = ocr.ocr(sharp_plate)

    detected_plates = []

    # Check if the OCR result is valid and not empty
    if ocr_result is None or len(ocr_result) == 0:
        print("No OCR result found.")
        return detected_plates  # Return an empty list if no result

    # Iterate over the result
    for line in ocr_result:
        if line is not None:  # Ensure the line is not None
            for box, (text, confidence) in line:
                if confidence >= 0.8:  # Only show results with confidence >= 80%
                    # Check if the detected text matches Turkey's plate format
                    if is_valid_turkey_plate(text):
                        detected_plates.append((text, confidence))
                        print(f"Detected plate: {text}, Confidence: {confidence}")

    return detected_plates


def detect_vehicle_and_plate(frame):
    # Detect vehicle
    vehicle_results = vehicle_model(frame)
    detected_plates = []

    for vehicle_result in vehicle_results:
        for (
            v_box
        ) in vehicle_result.boxes.xyxy:  # Directly iterating over bounding boxes
            x1, y1, x2, y2 = map(int, v_box)  # Convert to int
            vehicle_frame = frame[y1:y2, x1:x2]

            # Detect plate within the vehicle bounding box
            plate_results = plate_model(vehicle_frame)
            for plate_result in plate_results:
                for p_box in plate_result.boxes.xyxy:
                    px1, py1, px2, py2 = map(int, p_box)
                    plate_frame = vehicle_frame[py1:py2, px1:px2]

                    # Read the plate text using PaddleOCR
                    plate_text = read_license_plate(plate_frame)
                    if plate_text:
                        detected_plates.append(plate_text)

                        # Draw rectangle around plate
                        cv2.rectangle(
                            vehicle_frame, (px1, py1), (px2, py2), (0, 0, 255), 2
                        )

            # Draw rectangle around vehicle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame, detected_plates


# Load video file
video_path = "./exampleVideo.webm"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing (optional)
    frame = cv2.resize(frame, (640, 480))

    # Detect vehicle and plate
    frame, detected_plates = detect_vehicle_and_plate(frame)

    # Display the output
    cv2.imshow("Vehicle and Plate Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
