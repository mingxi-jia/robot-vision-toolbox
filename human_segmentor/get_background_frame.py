import cv2

video_path = "/path/to/your/video.mp4"
output_path = "first_frame.png"

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

if ret:
    cv2.imwrite(output_path, frame)
    print(f"✅ First frame saved to {output_path}")
else:
    print("❌ Failed to read video or grab first frame.")

cap.release()
