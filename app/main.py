from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import datetime
import cv2
import torch
from torchvision import transforms
from PIL import Image
from cnn import SmileCNN
import numpy as np

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hilfsfunktion zum Laden eines Modells
def load_model(model_path, device):
    model = SmileCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model

# Bildvorverarbeitung: Resizing, Tensor-Konvertierung und Normalisierung
def preprocess_image(face):
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    face_pil = Image.fromarray(face)
    face_tensor = transform(face_pil)
    return face_tensor.unsqueeze(0)  # Batch-Dimension hinzufügen

# Setup: Modelle laden und Face-Detector initialisieren
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = load_model("models/smile_cnn.pth", device)
# model2 = load_model("models/smile_cnn2.pth", device) --> Wir müssen hier noch eins hinzufügen
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    raise RuntimeError("Konnte Haar Cascade XML Datei nicht laden.")

# Funktion zur Smile-Erkennung: Es wird ein Bild (BGR) verarbeitet, das Gesicht gesucht und anhand des Modells evaluiert.
def detect_smile(image_array, model):
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    for (x, y, w, h) in faces:
        face = image_array[y:y+h, x:x+w]
        face_tensor = preprocess_image(face).to(device)
        with torch.no_grad():
            output = model(face_tensor).item()
        if output > 0.5:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return {"smile_detected": True, "timestamp": timestamp}  # Lächeln erkannt
    return False

# Endpoint für Model 1
@app.post("/detect/model1")
async def detect_model1(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        return JSONResponse({"error": "Ungültiges Bild"}, status_code=400)
    
    smile = detect_smile(image, model1)
    result = {"model": "model1", "smile_detected": smile}
    if smile:
        print("Smile detected")
        now = datetime.datetime.now()
        timestamp = now.strftime("%H:%M:%S.%f")[:-3]
        result["timestamp"] = timestamp
    return JSONResponse(result)

# Endpoint für Model 2
# @app.post("/detect/model2")
# async def detect_model2(file: UploadFile = File(...)):
#     contents = await file.read()
#     np_arr = np.frombuffer(contents, np.uint8)
#     image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#     if image is None:
#         return JSONResponse({"error": "Ungültiges Bild"}, status_code=400)
    
#     smile = detect_smile(image, model2)
#     result = {"model": "model2", "smile_detected": smile}
#     if smile:
#         now = datetime.datetime.now()
#         timestamp = now.strftime("%H:%M:%S.%f")[:-3]
#         result["timestamp"] = timestamp
#     return JSONResponse(result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
