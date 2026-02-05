# API Detect Animals

API FastAPI + Pytorch Wildlife pour détecter animaux et personnes dans une image.

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
