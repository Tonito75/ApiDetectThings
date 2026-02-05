from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import torch
import time
import io

app = FastAPI()

# MegaDetector v5a : 0=animal, 1=person, 2=vehicle
CLASS_NAMES = {0: "animal", 1: "person", 2: "vehicle"}
CONFIDENCE_THRESHOLD = 0.2

print("[INFO] Loading MegaDetector v5a...")
model = torch.hub.load(
    "/app/yolov5", "custom",
    path="/app/md_v5a.0.0.pt",
    source="local"
)
model.conf = 0.1  # seuil bas pour tout voir dans les logs
print("[INFO] MegaDetector v5a loaded.")


@app.get("/health")
async def health():
    return {"status": "ok", "model": "MegaDetector v5a"}


@app.post("/detect-animal")
async def detect_animal(file: UploadFile = File(...)) -> bool:
    print(f"[REQUEST] file={file.filename} content_type={file.content_type}")

    data = await file.read()
    print(f"[INFO] File size: {len(data)} bytes")

    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    w, h = image.size
    print(f"[INFO] Image: {w}x{h}")

    # Redimensionner pour accélérer l'inférence CPU
    max_side = 1280
    if max(w, h) > max_side:
        ratio = max_side / max(w, h)
        image = image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        print(f"[INFO] Resized to {image.size[0]}x{image.size[1]}")

    start = time.time()
    results = model(image)
    elapsed = time.time() - start
    print(f"[INFO] Inference: {elapsed:.1f}s")
    detections = results.xyxy[0].cpu().numpy()

    print(f"[INFO] {len(detections)} detection(s)")

    animal_found = False
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        cls = int(cls)
        name = CLASS_NAMES.get(cls, f"unknown({cls})")
        tag = ">>> MATCH" if cls == 0 and conf >= CONFIDENCE_THRESHOLD else ""
        print(f"  [{name}] conf={conf:.3f} bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] {tag}")
        if cls == 0 and conf >= CONFIDENCE_THRESHOLD:
            animal_found = True

    print(f"[RESULT] {'ANIMAL DETECTED' if animal_found else 'No animal'}")
    print("-" * 50)
    return animal_found
