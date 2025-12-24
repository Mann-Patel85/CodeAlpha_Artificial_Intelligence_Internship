import cv2
from ultralytics import YOLO

# --- CHECKLIST ITEM 2 & 3: Load Pre-trained Model (YOLO) ---
# We load the 'nano' model (yolov8n.pt) because it is fast and runs on laptops.
# It will download automatically the first time you run it.
print("🧠 Loading YOLO AI Model...")
model = YOLO('yolov8n.pt')

# --- CHECKLIST ITEM 1: Real-time Video Input ---
# 0 usually means your main webcam. If you have an external one, try 1.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("🎥 Webcam started! Press 'Q' to quit.")

while True:
    # Read a frame from the webcam
    success, frame = cap.read()
    if not success:
        break

    # --- CHECKLIST ITEM 4 & 5: Process Frame, Detect, Track, and Display ---
    # model.track does everything: detection + tracking IDs
    # persist=True tells the AI "this is a video, remember objects from previous frames"
    results = model.track(frame, persist=True)

    # Plot the results (draw boxes, labels, and IDs) directly on the frame
    annotated_frame = results[0].plot()

    # Show the video on your screen
    cv2.imshow("YOLOv8 Object Tracking - Task 4", annotated_frame)

    # Press 'q' to quit the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()