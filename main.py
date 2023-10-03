import os
from pathlib import Path
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from api.models import Box
from auth import api_key_auth
from networks.cfrgan_face_frontalizer import CFRGANFaceFrontalizer
from config import get_settings
from networks.gfpgan_face_ehnancer import GFPGANFaceEnhancer
from networks.ort_realesrgan_face_enhancer import ORTRealESRGANFaceEnhancer
from networks.ort_retinaface import ORTFaceDetector
from fastapi import FastAPI, Body, Depends
from starlette_admin.contrib.sqla import Admin, ModelView

description = """
This site should not be used except by Atfalmafkoda scheduled jobs.

Code running this API can be found here: <https://github.com/yousseb/atfal-ai>

## Faces

* Detect faces and returns **boxes** where faces could be within a given image.
* Enhances faces
* Frontalizes faces

"""
app = FastAPI(
    title="Atfalmafkoda AI APIs 🚀",
    description=description,
    summary="AI API engine to help extract and match missing cases.",
    version="0.0.5",
    contact={
        "name": "Atfalmafkoda",
        "url": "https://atfalmafkoda.com/",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/license/mit/",
    },
    swagger_ui_parameters={
        'Bearer': {
            'type': 'apiKey',
            'name': 'Authorization',
            'in': 'header',
            'description': '<hr/>'
                           'Enter the word <tt>Token</tt> followed by space then your apiKey <br/><br/> '
                           '<b>Example:</b> <pre>Token f4bff35e0f6427860ae31bde0b5f2352cbf73d80</pre>'
                           '<hr/><br/>'
        }
    }
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

ASSETS_FOLDER = Path(os.getcwd()) / 'assets'
face_detector = ORTFaceDetector(ASSETS_FOLDER)
# face_enhancer = ORTRealESRGANFaceEnhancer(ASSETS_FOLDER)
face_enhancer = GFPGANFaceEnhancer(ASSETS_FOLDER)
face_frontalizer = CFRGANFaceFrontalizer(ASSETS_FOLDER)


@app.get("/faces/", dependencies=[Depends(api_key_auth)])
async def detect_faces(image_url: str):
    faces = face_detector.detect_faces(image_url=image_url)
    return faces


def remove_file(path: str) -> None:
    os.unlink(path)


@app.post("/enhance/", dependencies=[Depends(api_key_auth)])
async def enhance_face(image_url: str,
                       box: Box) -> FileResponse:
    enhanced_face_path = face_enhancer.enhance_face(image_url=image_url, box=box)
    return FileResponse(enhanced_face_path,
                        media_type="image/png",
                        background=BackgroundTask(remove_file, enhanced_face_path), )


@app.post("/frontalize/", response_class=FileResponse, dependencies=[Depends(api_key_auth)])
async def frontalize_face(image_url: str) -> FileResponse:
    enhanced_face_path = face_frontalizer.frontaliza_face(image_url=image_url)
    return FileResponse(enhanced_face_path,
                        media_type="image/png",
                        background=BackgroundTask(remove_file, enhanced_face_path), )


@app.get("/hello/{name}", dependencies=[Depends(api_key_auth)])
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
