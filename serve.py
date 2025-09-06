from fastapi import FastAPI, File, UploadFile
from transformers import pipeline
from PIL import Image
from io import BytesIO

app = FastAPI(title="DeepFake Detector")

classifier = pipeline(
    "image-classification",
    model="prithivMLmods/Deep-Fake-Detector-v2-Model",
    device=-1  # CPU
)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read())).convert("RGB")
    result = classifier(image)
    return {"prediction": result}
