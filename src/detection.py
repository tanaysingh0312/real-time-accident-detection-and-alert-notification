import cv2
import os
import sys
from ultralytics import YOLO

# --- CRITICAL FIX: Robust Path Handling ---
# Get the directory where THIS file (detection.py) is located (i.e., .../src)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root (one level up from 'src')
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# Add project root to sys.path so we can safely import env_vars
if project_root not in sys.path:
    sys.path.append(project_root)

# Now safe to import env_vars
import env_vars as ev
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from twilio.rest import Client

# --- Model Loading Logic ---
# Construct the absolute path to the model to avoid FileNotFoundError
# Check 'ML part/best.pt' first (as per your folder structure)
model_path = os.path.join(project_root, "ML part", "best.pt")

# Fallback: Check project root if not found in 'ML part'
if not os.path.exists(model_path):
    model_path = os.path.join(project_root, "best.pt")

print(f"Loading model from: {model_path}")

try:
    model = YOLO(model_path)
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None


def send_whatsapp_message(to_whatsapp_number, message):
    """Sends a WhatsApp message using credentials from env_vars."""
    try:
        account_sid = ev.account_sid
        auth_token = ev.auth_token
        from_whatsapp_number = ev.from_whatsapp_number
        
        if not all([account_sid, auth_token, from_whatsapp_number]):
            print("Twilio credentials missing or incomplete.")
            return False

        client = Client(account_sid, auth_token)
        # Twilio requires the 'From' number to have the 'whatsapp:' prefix
        from_number = "whatsapp:" + from_whatsapp_number if not from_whatsapp_number.startswith("whatsapp:") else from_whatsapp_number
        to_number = "whatsapp:" + to_whatsapp_number if not to_whatsapp_number.startswith("whatsapp:") else to_whatsapp_number

        msg = client.messages.create(
            body=message,
            from_=from_number,
            to=to_number,
        )
        print("WhatsApp message sent successfully:", msg.sid)
        return True
    except Exception as e:
        print("Error sending WhatsApp message:", e)
        return False


def send_email_with_frame(frame_path, class_name, rounded_conf, location):
    """Sends an email with the detected frame attached."""
    try:
        if not all([ev.from_email, ev.to_email, ev.smtp_username, ev.smtp_password]):
            print("Email credentials missing or incomplete.")
            return False

        msg = MIMEMultipart()
        msg["From"] = ev.from_email
        msg["To"] = ev.to_email
        msg["Subject"] = "ACCIDENT DETECTED!"

        body = f"Alert: An accident of type '{class_name}' was detected with {rounded_conf * 100:.2f}% confidence.\nLocation: {location}"
        msg.attach(MIMEText(body, "plain"))

        # Attach the image file
        with open(frame_path, "rb") as fp:
            img = MIMEImage(fp.read())
            img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(frame_path))
            msg.attach(img)

        # Connect to Gmail SMTP server
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(ev.smtp_username, ev.smtp_password)
        server.send_message(msg)
        server.quit()
        print("Email sent successfully!")
        return True
    except Exception as e:
        print("Error sending email:", e)
        return False


class Detection:
    @staticmethod
    def prediction(image_path):
        """
        Runs YOLO prediction on the given image path.
        Returns: [detected_class_name, confidence_score]
        """
        if model is None:
            return ["Model Error", 0]

        # Run prediction
        # save=True will save annotated images to 'runs/detect/predict...'
        results = model.predict(source=image_path, save=True, show=False)
        result = results[0]

        detected_class = "No Accident"
        confidence = 0.0

        if len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                c_name = result.names[cls_id]
                conf = round(box.conf[0].item(), 2)

                # Threshold for a valid detection (e.g., 50% confidence)
                if conf >= 0.5:
                    detected_class = c_name
                    confidence = conf
                    
                    # Trigger Alerts
                    location = "Uploaded File"
                    print(f"Triggering alerts for {c_name}...")
                    
                    # Send WhatsApp
                    send_whatsapp_message(
                        ev.to_whatsapp_number, 
                        f"Accident detected: {c_name} with {conf * 100:.1f}% confidence."
                    )
                    
                    # Send Email
                    send_email_with_frame(image_path, c_name, conf, location)
                    
                    # Return immediately after the first valid detection
                    return [detected_class, confidence]

        return [detected_class, confidence]
