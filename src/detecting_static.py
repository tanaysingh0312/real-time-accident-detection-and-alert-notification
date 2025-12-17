from ultralytics import YOLO

# Load a model
# Assuming 'best.pt' is in the project root (../ from src/)
model = YOLO("../best.pt")

# path variables
# Assuming results will be saved in a 'results' folder inside 'src' (../src/results/)
save_path = "../src/results/" 
# Paths for image/video updated to be relative to the current file (in src/) inside 'inputs' folder
image_path = "inputs/images/image3.jpg"
video_path = "inputs/videos/video3.mp4"

# detection
results = model.predict(source=image_path, project=save_path, save=True, show=True)
result = results[0]
box = result.boxes[0]

# extracting data to appropriate variables
for box in result.boxes:
    class_id = result.names[box.cls[0].item()]
    cords = box.xyxy[0].tolist()
    cords = [round(x) for x in cords]
    conf = round(box.conf[0].item(), 2)

    print("Object type:", class_id)
    print("Coordinates:", cords)
    print("Probability:", conf)
    print("---")
