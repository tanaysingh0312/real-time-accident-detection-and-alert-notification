Real-Time Accident Detection and Alert Notification using YOLOv8

This project implements a real-time accident detection system using the YOLOv8 object detection model. The system is capable of detecting potential road accidents from live video streams, static images, and pre-recorded video files. It is designed to support rapid response scenarios and provides a foundation for automated alert and notification pipelines.

The repository includes the complete detection pipeline, model inference logic, user interface components, and Raspberry Pi–optimized scripts for lightweight edge deployment.

Setup Instructions
1. Create and Activate a Virtual Environment
python -m venv venv


Windows

venv\Scripts\activate


Linux / macOS

source venv/bin/activate

2. Install Dependencies
pip install -r requirements.txt

Clone the Repository
git clone https://github.com/tanaysingh0312/real-time-accident-detection-and-alert-notification.git
cd real-time-accident-detection-and-alert-notification

Accident Detection Modes
2.1 Static Image or Video Detection

This mode processes images or recorded video files and saves the detection output.

Script Location

src/detect_static.py


Run

python src/detect_static.py


Functionality

Loads the YOLOv8 model

Processes the selected image or video

Detects accident-related patterns

Saves annotated results to the results/ directory

2.2 Real-Time Video Stream Detection

This mode performs real-time accident detection using a webcam or live video feed.

Script Location

src/detect_stream.py


Run

python src/detect_stream.py


Functionality

Captures frames from the configured live stream

Performs YOLOv8 inference on each frame

Displays and optionally saves detection results

Deployment (Optional)
Raspberry Pi Deployment

The RaspberryPi/ directory contains scripts optimized for edge deployment, enabling low-power real-time accident detection on Raspberry Pi devices.

User Interface

The UI/ directory contains user interface components for visualization and interaction with the detection system.

Project Structure
real-time-accident-detection-and-alert-notification/
│
├── src/              # Core YOLOv8 detection scripts
├── UI/               # User interface components
├── RaspberryPi/      # Edge deployment scripts
├── requirements.txt  # Project dependencies
├── README.md         # Project documentation
└── .gitignore

Author

Tanay Singh

GitHub: https://github.com/tanaysingh0312

LinkedIn: https://www.linkedin.com/in/stanay657/
