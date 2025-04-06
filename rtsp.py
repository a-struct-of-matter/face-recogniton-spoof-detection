import cv2

from cam_feed import rtsp_url

rtsp_url='rtsp://admin:L2B81D42@192.168.29.177:554/cam/realmonitor?channel=1&subtype=1'
# Open RTSP stream
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("❌ Failed to connect to the RTSP stream. Check credentials and network.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to retrieve frame.")
        break

    cv2.imshow("RTSP Stream", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#rtsp://192.168.29.169:5543/live/channel1