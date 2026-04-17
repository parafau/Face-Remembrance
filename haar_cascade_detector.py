import cv2

# Load the Haar Cascade XML file (pre-trained model)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Start webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Convert to grayscale (Haar works on grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        pad = 0.1  # 20% padding

        cv2.rectangle(
            frame,
            (int(x - pad*w), int(y - pad*h)),
            (int(x + w + pad*w), int(y + h + pad*h)),
            (255, 0, 0),
            2
        )
    # Show output
    cv2.imshow('Face Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()