import base64
import io
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rapid_latex_ocr import LatexOCR

app = FastAPI(title="MeyzNote Math OCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

model = LatexOCR()

class MathBase64Request(BaseModel):
    image_base64: str

@app.get("/")
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/api/v1/recognize-math-base64")
async def recognize_math_base64(req: MathBase64Request):
    try:
        img_data = req.image_base64
        if "," in img_data:
            img_data = img_data.split(",", 1)[1]
        img_bytes = base64.b64decode(img_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Geçersiz görüntü: {e}")

    latex_str, _ = model(img_bytes)
    return {"latex": latex_str.strip()}
