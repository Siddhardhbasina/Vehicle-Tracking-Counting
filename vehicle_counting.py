import cv2

# Load Video
cap = cv2.VideoCapture("videos/traffic.mp4")

# Vehicle Detector
car_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_car.xml"
)

vehicle_count = 0
detected_centers = []

line_y = 350

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    vehicles = car_cascade.detectMultiScale(
        gray,
        1.1,
        2
    )

    cv2.line(
        frame,
        (0, line_y),
        (1200, line_y),
        (0, 255, 255),
        2
    )

    for (x, y, w, h) in vehicles:

        center_x = x + w // 2
        center_y = y + h // 2

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.circle(
            frame,
            (center_x, center_y),
            4,
            (0, 0, 255),
            -1
        )

        if line_y - 10 < center_y < line_y + 10:

            if (center_x, center_y) not in detected_centers:
                vehicle_count += 1
                detected_centers.append((center_x, center_y))

    cv2.putText(
        frame,
        f"Vehicle Count: {vehicle_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.imshow("Vehicle Tracking & Counting", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()