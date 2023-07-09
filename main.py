import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from typing import Annotated
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask
from functools import lru_cache
from networks.cfrgan_face_frontalizer import CFRGANFaceFrontalizer
from config import Settings
from networks.ov_bsrgan_face_enhancer import OVBSRGANFaceEnhancer
from networks.ov_face_detector import OVFaceDetector

#from delib_face_detector import DLibFaceDetector
#from face_frontalizer import FaceFrontalizer
#from rrdb_esrgan_face_enhancer import RRDB_ESRGANFaceEnhancer

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)
#oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

ASSETS_FOLDER = Path(os.getcwd()) / 'assets'
face_detector = OVFaceDetector(ASSETS_FOLDER)
face_enhancer = OVBSRGANFaceEnhancer(ASSETS_FOLDER)
face_frontalizer = CFRGANFaceFrontalizer(ASSETS_FOLDER)


@lru_cache()
def get_settings():
    return Settings()


class Box(BaseModel):
    x1: Annotated[int, "X1"]
    x2: Annotated[int, "X2"]
    y1: Annotated[int, "Y1"]
    y2: Annotated[int, "Y2"]


@app.get("/faces/", response_model=list[Box])
async def detect_faces(image_url: Annotated[str, "Full image URL to detect faces from."]) -> list[Box]:
    faces = face_detector.detect_faces(image_url=image_url)
    return faces


def remove_file(path: str) -> None:
    os.unlink(path)


@app.post("/enhance/")
async def enhance_face(image_url: Annotated[str, "Full image URL."],
                       box: Annotated[Box, "Box dimensions for cropping."]) -> FileResponse:
    enhanced_face_path = face_enhancer.enhance_face(image_url=image_url, box=box)
    return FileResponse(enhanced_face_path,
                        media_type="image/png",
                        background=BackgroundTask(remove_file, enhanced_face_path), )


@app.post("/frontalize/", response_class=FileResponse)
async def frontalize_face(image_url: Annotated[str, "Cropped face image URL"]) -> FileResponse:
    enhanced_face_path = face_frontalizer.frontaliza_face(image_url=image_url)
    return FileResponse(enhanced_face_path,
                        media_type="image/png",
                        background=BackgroundTask(remove_file, enhanced_face_path), )

@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
