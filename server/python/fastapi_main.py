from fastapi import FastAPI
from routes import face
from routes import api

app = FastAPI(title="SatyaAI FastAPI API")
app.include_router(face.router, prefix="/api/face", tags=["Face"])
app.include_router(api.router, prefix="/api", tags=["API"])

@app.get("/api/face/hello")
def hello():
    return {"message": "Hello from FastAPI!"} 