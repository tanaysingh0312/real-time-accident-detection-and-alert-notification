import os
import cv2
import sys
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import smtplib 
from email.message import EmailMessage 

# --------------------------------------------------
# CONFIGURATION & CONSTANTS
# --------------------------------------------------
MODEL_NAME = "yolov8n.pt" 
# Filter for objects involved in traffic/accidents
ACCIDENT_CLASSES = {"car", "bus", "truck", "motorcycle", "bicycle", "person"}
IOU_THRESHOLD = 0.25      # Overlap threshold for collision inference
MULTIPLE_VEHICLES_THRESHOLD = 3 # Clustering rule for pile-ups

# Flask directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
STATIC_FOLDER = os.path.join(BASE_DIR, "static")
RESULTS_FOLDER = os.path.join(STATIC_FOLDER, "results")

# Spam prevention flag (retained for potential video stream future use)
alert_sent = False 

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def iou(boxA, boxB):
    """Calculates the Intersection over Union (IoU) of two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea == 0:
        return 0
    return interArea / unionArea

def send_email_alert():
    """Sends an email alert using hardcoded credentials."""
    # Hardcoded credentials as per user request
    SENDER_EMAIL = "stanay657@gmail.com"
    APP_PASSWORD = "zgceeztzobrzsury"
    RECEIVER_EMAIL = "tanay.singh@spsu.ac.in"

    msg = EmailMessage()
    msg.set_content(
    "🚨 ACCIDENT ALERT 🚨\n\n"
    "The system has detected a possible road accident based on vehicle collision analysis.\n"
    "Time-sensitive intervention is recommended.\n\n"
    "This is an automated alert."
    )

    msg["Subject"] = "Accident Alert"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL

    try:
        # Use smtplib.SMTP_SSL for secure connection on port 465 (Gmail)
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.send_message(msg)
        print("✅ Email alert sent successfully.")
        return True
    except Exception as e:
        print(f"❌ Email alert failed: {e}")
        print("Check if the sender email has 'Less secure app access' enabled or if the App Password is correct.")
        return False


# --------------------------------------------------
# FLASK APP INITIALIZATION
# --------------------------------------------------
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_FOLDER, template_folder="templates")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --------------------------------------------------
# LOAD YOLO MODEL (ONCE)
# --------------------------------------------------
print(f"Loading YOLO model: {MODEL_NAME} (This will download if not cached)")

model = None
try:
    # Use yolov8n.pt for general object detection as a base
    model = YOLO(MODEL_NAME)
    print("YOLOv8n model loaded successfully (using COCO pre-trained weights)")
except Exception as e:
    print(f"CRITICAL ERROR: YOLO model failed to load. Error: {e}")
    model = None


# --------------------------------------------------
# ROUTES
# --------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    """Renders the main upload page (GET) or handles accident inference (POST)."""
    global alert_sent
    message = "Upload a traffic image to check for accident events."
    result_image_url = None
    detections = []
    accident_detected = False
    
    # Reset alert status for new file upload
    alert_sent = False 

    if request.method == "POST":
        
        if model is None:
            message = "Error: AI Model failed to initialize. Cannot perform detection."
            return render_template("home.html", message=message)

        file = request.files.get("uploadedImage")

        if not file or file.filename == "":
            message = "No file selected."
        
        elif file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            try:
                filename = secure_filename(file.filename)
                upload_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(upload_path)

                print(f"File uploaded to: {upload_path}")

                results = model.predict(source=upload_path, save=False, verbose=False) 
                result = results[0] 

                img = cv2.imread(upload_path)
                
                boxes = [] # Stores [x1, y1, x2, y2] for IOU calculation
                labels = [] # Stores class names
                
                # --- DETECTION LOOP ---
                if len(result.boxes) > 0:
                    for box in result.boxes:
                        cls_id = int(box.cls[0].item())
                        conf = round(box.conf[0].item(), 2)
                        cls_name = result.names.get(cls_id, f"Class {cls_id}")

                        # Filter for accident-relevant classes
                        if cls_name not in ACCIDENT_CLASSES:
                            continue

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        boxes.append([x1, y1, x2, y2])
                        labels.append(cls_name)
                        
                        detections.append({
                            "class": cls_name, 
                            "confidence": conf,
                            "bbox": [x1, y1, x2, y2],
                        })
                        
                        # Draw bounding box (default GREEN)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            img,
                            f"{cls_name} {conf}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )

                # --- ACCIDENT INFERENCE LOGIC (IOU and Clustering) ---

                # 1. Collision Rule (IOU > threshold) 
                if len(boxes) >= 2:
                    for i in range(len(boxes)):
                        for j in range(i + 1, len(boxes)):
                            if iou(boxes[i], boxes[j]) > IOU_THRESHOLD:
                                accident_detected = True
                                print(f"Collision detected between object {i} ({labels[i]}) and object {j} ({labels[j]}) with IOU={iou(boxes[i], boxes[j]):.2f}")
                                break
                        if accident_detected:
                            break

                # 2. Clustering Rule (Multiple vehicles/persons) 
                if len(boxes) >= MULTIPLE_VEHICLES_THRESHOLD and not accident_detected:
                    accident_detected = True
                    print(f"Clustering detected: {len(boxes)} relevant objects found.")


                # --- VISUALIZATION, OUTPUT, AND ALERTING ---

                if accident_detected:
                    # Redraw bounding boxes for involved entities in RED
                    for box in boxes:
                        x1, y1, x2, y2 = box
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4) 
                    
                    message = "🚨 ACCIDENT DETECTED! Processing email alert..."
                    
                    output_name = f"detected_{filename}"
                    output_path = os.path.join(RESULTS_FOLDER, output_name)
                    cv2.imwrite(output_path, img)
                    result_image_url = url_for("static", filename=f"results/{output_name}")

                    # --- TRIGGER EMAIL ALERT ONLY ---
                    if send_email_alert():
                         message += " Email alert sent successfully."
                    else:
                         message += " Email alert failed/skipped."

                    alert_sent = True 
                        
                else:
                    # No accident detected
                    message = "✅ No accident detected."
                    # Save the image with green boxes
                    output_name = f"detected_{filename}"
                    output_path = os.path.join(RESULTS_FOLDER, output_name)
                    cv2.imwrite(output_path, img)
                    result_image_url = url_for("static", filename=f"results/{output_name}")


            except Exception as e:
                print(f"An unexpected error occurred during detection: {e}")
                import traceback
                traceback.print_exc()
                message = f"Processing error: {str(e)}"
        else:
             message = "Invalid file type. Please upload a PNG, JPG, or JPEG file."


    # Render template
    return render_template(
        "home.html",
        message=message,
        result_image_url=result_image_url,
        detections=detections,
    )


# --------------------------------------------------
# MAIN ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=1234, debug=True)
