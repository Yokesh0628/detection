import torch
import cv2

# Load the custom YOLOv5 model
model_path = 'runs/train/exp/weights/yolov5s.pt'  # Update with your correct path
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

# Open the input video
video_path = 'C:/Users/asus/yolov5/animal_dataset/cat.mp4'  # Update with your video path
cap = cv2.VideoCapture(video_path)

# Define the output video parameters
output_path = 'output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLOv5 detection on the frame
    results = model(frame)
    
    # Render the results on the frame
    annotated_frame = results.render()[0]
    
    # Write the frame to the output video
    out.write(annotated_frame)
    
    # Display the frame (optional)
    cv2.imshow('Animal Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Detection complete. Video saved to: {output_path}")
