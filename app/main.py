from fastapi import FastAPI

# Create the FastAPI app
app = FastAPI()

# Define a simple route
@app.get("/")
def read_root():
    return {"message": "Welcome to your FastAPI app!"}

# SmileDetection
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from torch import nn

# CNN Model Definition (should match the one used for training)
class SmileCNN(nn.Module):
    def __init__(self):
        super(SmileCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128),  # Assuming input size 48x48
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Load the trained model
def load_model(model_path, device):
    model = SmileCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
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

