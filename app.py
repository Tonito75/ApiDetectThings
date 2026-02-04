from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# Utilise un modèle YOLO11 (plus récent et performant que YOLOv8)
model = YOLO("yolo11l.pt")

# Classes COCO d'animaux + person (IDs CORRECTS du dataset COCO)
ANIMAL_CLASSES = {
    0: 'person',     # personne
    14: 'bird',      # oiseau
    15: 'cat',       # chat
    16: 'dog',       # chien
    17: 'horse',     # cheval
    18: 'sheep',     # mouton
    19: 'cow',       # vache
    20: 'elephant',  # éléphant
    21: 'bear',      # ours
    22: 'zebra',     # zèbre
    23: 'giraffe'    # girafe
}

def detect_animals(results, confidence_threshold: float = 0.4) -> bool:
    """
    Détecte uniquement les animaux + personnes avec un seuil de confiance
    Retourne True si au moins un animal/personne est détecté
    """
    animals_found = []
    all_detections = []
    
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = r.names[class_id]
            
            # Log toutes les détections pour debug
            all_detections.append({
                "class_name": class_name,
                "class_id": class_id,
                "confidence": round(confidence, 3),
                "is_animal": class_id in ANIMAL_CLASSES
            })
            
            # Filtrer: seulement les animaux/personnes avec confiance > seuil
            if class_id in ANIMAL_CLASSES and confidence > confidence_threshold:
                animals_found.append({
                    "animal": ANIMAL_CLASSES[class_id],
                    "confidence": round(confidence, 3)
                })
    
    # Logs détaillés
    print(f"[DEBUG] Total detections: {len(all_detections)}")
    if all_detections:
        print(f"[DEBUG] All detections: {all_detections}")
    
    if animals_found:
        print(f"[ANIMAL DETECTED] {len(animals_found)} animal(s)/person(s) found:")
        for animal in animals_found:
            print(f"  - {animal['animal']}: {animal['confidence']:.3f}")
    else:
        print("[NO ANIMAL] No animals/persons detected above threshold")
    
    return len(animals_found) > 0

@app.post("/detect-animal")
async def detect_animal(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.4  # Threshold par défaut abaissé à 0.4
) -> bool:
    print(f"[INFO] detect-animal called with file: {file.filename}")
    print(f"[INFO] Content-Type: {file.content_type}")
    print(f"[INFO] Confidence threshold: {confidence_threshold}")
    
    # 1️⃣ Lire le fichier uploadé
    try:
        data = await file.read()
        print(f"[INFO] File size: {len(data)} bytes")
    except Exception as e:
        print(f"[ERROR] Failed to read file: {e}")
        raise HTTPException(status_code=400, detail="Failed to read uploaded file")
    
    # 2️⃣ Charger l'image avec PIL
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        print(f"[INFO] Image dimensions: {image.size[0]}x{image.size[1]}")
    except Exception as e:
        print(f"[ERROR] PIL failed to open image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    # 3️⃣ YOLO inference
    try:
        print("[INFO] Running YOLO inference...")
        results = model(image)
    except Exception as e:
        print(f"[ERROR] YOLO inference failed: {e}")
        raise HTTPException(status_code=500, detail="Model inference failed")
    
    # 4️⃣ Détection d'animaux avec logs
    detected = detect_animals(results, confidence_threshold)
    
    print(f"[RESULT] Final answer: {detected}")
    print("-" * 50)
    
    return detected