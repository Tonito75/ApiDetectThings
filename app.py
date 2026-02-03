from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, HttpUrl
from ultralytics import YOLO
from PIL import Image
import io
import httpx

app = FastAPI()
model = YOLO("yolov8l.pt")

class ImageUrl(BaseModel):
    url: HttpUrl

@app.post("/detect-animal")
async def detect_animal(file: UploadFile = File(...)) -> bool:
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    results = model(image)
    return any(len(r.boxes) > 0 for r in results)

@app.post("/detect-animal-url")
async def detect_animal_url(data: ImageUrl) -> bool:
    async with httpx.AsyncClient() as client:
        response = await client.get(str(data.url))
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
    results = model(image)
    return any(len(r.boxes) > 0 for r in results)