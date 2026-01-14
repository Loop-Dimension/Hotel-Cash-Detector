import cv2

rtsp_url = "rtsp://admin:adminadmin!@175.213.81.91:554/stream1"

cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("❌ Cannot open RTSP stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame")
        break

    cv2.imshow("Dahua Live Stream", frame)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
