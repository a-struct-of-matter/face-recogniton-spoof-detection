import cv2

rtsp_url = "rtsp://admin:L2B81D42@192.168.137.110:554/cam/realmonitor?channel=1&subtype=1"
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Failed to open the video stream")
    exit()

# Initialize variables for FPS calculation
frame_count = 0
start_time = cv2.getTickCount()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to fetch frame")
        break

    # Increment frame count
    frame_count += 1

    # Calculate elapsed time
    elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()

    # Calculate FPS
    fps = frame_count / elapsed_time

    # Resize the frame
    window_width = 800  # Set desired window width
    aspect_ratio = frame.shape[1] / frame.shape[0]
    window_height = int(window_width / aspect_ratio)
    resized_frame = cv2.resize(frame, (window_width, window_height))

    # Display FPS on the frame
    cv2.putText(resized_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Live Feed", resized_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()