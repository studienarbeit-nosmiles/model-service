from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import datetime
import cv2
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load MobileNetV3 with custom output layer
from torchvision.models import mobilenet_v3_small

def load_mobilenet_model(model_path, device):
    # Anzahl Klassen im trainierten Modell: 2
    model = mobilenet_v3_small()
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)


# Image preprocessing: Resize, convert to tensor, normalize
def preprocess_image(image: np.ndarray) -> torch.Tensor:
    # Falls das Bild nur 1 Kanal hat (Graustufen), konvertiere zu 3 Kanälen
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # In PIL-Image umwandeln
    pil_image = Image.fromarray(image)

    # Transforms definieren
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 3-Kanal Normalisierung
    ])

    tensor = transform(pil_image).unsqueeze(0)  # Batch-Dimension hinzufügen
    return tensor

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = load_mobilenet_model("saved_models/mobilenet_v3_small.pth", device)
model2 = load_mobilenet_model("models_checkpointed/mnetv3_s_fer2013_happy_final.pth", device)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    raise RuntimeError("Konnte Haar Cascade XML Datei nicht laden.")

# Smile detection function
def detect_smile(image_array, model):
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

    if len(faces) == 0:
        print("Kein Gesicht erkannt")
        return False

    for (x, y, w, h) in faces:
        face = image_array[y:y+h, x:x+w]
        face_tensor = preprocess_image(face).to(device)
        with torch.no_grad():
            output = model(face_tensor)
            probs = torch.softmax(output, dim=1)
            smile_prob = probs[0][1].item()  # Index 1: "Smile"

        if smile_prob > 0.5:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return {"smile_detected": True, "timestamp": timestamp}
    return False


# Model 1 Endpoint
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
        now = datetime.datetime.now()
        result["timestamp"] = now.strftime("%H:%M:%S.%f")[:-3]
    return JSONResponse(result)

# Model 2 Endpoint
@app.post("/detect/model2")
async def detect_model2(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        return JSONResponse({"error": "Ungültiges Bild"}, status_code=400)

    smile = detect_smile(image, model2)
    result = {"model": "model2", "smile_detected": smile}
    if smile:
        now = datetime.datetime.now()
        result["timestamp"] = now.strftime("%H:%M:%S.%f")[:-3]
    return JSONResponse(result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
