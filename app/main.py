from fastapi import FastAPI

# Create the FastAPI app
app = FastAPI()

# Define a simple route
@app.get("/")
def read_root():
    return {"message": "Welcome to your FastAPI app!"}

import cv2
import torch
from torchvision import transforms
from PIL import Image

# Load the trained model
def load_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model

# Preprocess image for the CNN
def preprocess_image(face):
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    face_pil = Image.fromarray(face)
    face_tensor = transform(face_pil)
    return face_tensor.unsqueeze(0)  # Add batch dimension

# Main function to run real-time smile detection
def smile_detection(model_path="models/smile_cnn.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    # Load OpenCV face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise RuntimeError("Failed to load Haar Cascade XML file for face detection.")

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot access the webcam.")

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Crop the face
            face = frame[y:y + h, x:x + w]

            # Preprocess the face and make a prediction
            face_tensor = preprocess_image(face).to(device)
            with torch.no_grad():
                output = model(face_tensor).item()

            # Add text based on prediction
            if output > 0.5:
                cv2.putText(frame, "Smiling", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                print("\a")  # Ping alert
            else:
                cv2.putText(frame, "Not Smiling", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Display the result
        cv2.imshow("Smile Detector", frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import uvicorn
    # Run the app with uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
    smile_detection()

