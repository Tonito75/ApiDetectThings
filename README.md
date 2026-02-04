# API Detect Animals

API FastAPI + YOLO11 pour détecter animaux et personnes dans une image.

## Build & Run

```bash
docker build -t apianimal .
docker run -p 8000:8000 apianimal
```

## Usage

```bash
curl -X POST http://localhost:8000/detect-animal -F "file=@image.jpg"
```

Retourne `true` si un animal ou une personne est détecté, `false` sinon.

## Options

| Param | Description | Défaut |
|-------|-------------|--------|
| `confidence_threshold` | Seuil de confiance (0-1) | 0.4 |

```bash
curl -X POST "http://localhost:8000/detect-animal?confidence_threshold=0.6" -F "file=@image.jpg"
```
