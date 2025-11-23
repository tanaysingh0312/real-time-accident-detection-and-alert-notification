Accident Detection and Notification using YOLOv8

This project implements a real-time accident detection system using the YOLOv8 object detection model. It can detect potential road accidents from live video streams as well as from static images and uploaded video files. The system is designed to support rapid response by enabling automated alerting or integration with notification pipelines.

The repository includes the full detection pipeline, model inference logic, UI components, and Raspberry Pi–ready scripts for lightweight deployment.

1. Setup
Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate   # Linux/Mac

Install dependencies
pip install -r requirements.txt

Clone the repository

If someone wants to clone your repo:

git clone https://github.com/tanaysingh0312/real-time-accident-detection-computer-vision.git
cd real-time-accident-detection-computer-vision

2. Accident Detection
2.1 Static Image or Video Detection

Use the script located in src/detect_static.py

Run:

python src/detect_static.py


The model will load YOLOv8, process the selected image/video, and save the output to the results/ directory.

2.2 Real-Time Video Stream Detection

Use the script located in src/detect_stream.py

Run:

python src/detect_stream.py


The script captures frames from the configured stream_url, performs YOLOv8 inference on each frame, and displays/saves the detection results.

3. Deployment (Optional)

The RaspberryPi/ folder contains scripts optimized for edge deployment on a Raspberry Pi for low-power accident detection scenarios.

The UI/ folder contains your user interface components for visualization and interaction.

4. Project Structure
real-time-accident-detection-computer-vision/
│
├── src/                   # Core detection scripts
├── UI/                    # User interface files
├── RaspberryPi/           # Pi-compatible scripts
├── requirements.txt       # Dependencies
├── README.md              # Documentation
└── .gitignore

Author

Tanay Singh
GitHub: https://github.com/tanaysingh0312

LinkedIn: https://www.linkedin.com/in/stanay657/
