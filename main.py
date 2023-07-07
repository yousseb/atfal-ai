import os
from pathlib import Path
from fastapi import FastAPI, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask
from functools import lru_cache
from config import Settings
from face_detector import FaceDetector
from face_enhancer import FaceEnhancer

app = FastAPI()

ASSETS_FOLDER = Path(os.getcwd()) / 'assets'

face_detector = FaceDetector(ASSETS_FOLDER)
face_enhancer = FaceEnhancer(ASSETS_FOLDER)


@lru_cache()
def get_settings():
    return Settings()


@app.get("/faces/")
async def detect_faces(image_url: str):
    faces = face_detector.detect_faces(image_url=image_url)
    return {"image_url": f"{image_url}",
            "faces": faces}


class Box(BaseModel):
    x1: int
    x2: int
    y1: int
    y2: int


def remove_file(path: str) -> None:
    os.unlink(path)


@app.post("/enhance/")
async def enhance_face(image_url: str, box: Box):
    enhanced_face_path = face_enhancer.enhance_face(image_url=image_url, box=box)
    return FileResponse(enhanced_face_path,
                        media_type="image/png",
                        background=BackgroundTask(remove_file, enhanced_face_path), )


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


experiment_type = 'ffhq_frontalize'
CODE_DIR = 'pixel2style2pixel'


def get_download_model_command(file_id, file_name):
    """ Get wget download command for downloading the desired model and save to directory ../pretrained_models. """
    current_directory = os.getcwd()
    save_path = os.path.join(os.path.dirname(current_directory), CODE_DIR, "pretrained_models")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    url = r"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={FILE_ID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={FILE_ID}" -O {SAVE_PATH}/{FILE_NAME} && rm -rf /tmp/cookies.txt""".format(
        FILE_ID=file_id, FILE_NAME=file_name, SAVE_PATH=save_path)
    return url


MODEL_PATHS = {
    "ffhq_encode": {"id": "1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0", "name": "psp_ffhq_encode.pt"},
    "ffhq_frontalize": {"id": "1_S4THAzXb-97DbpXmanjHtXRyKxqjARv", "name": "psp_ffhq_frontalization.pt"},
    "celebs_super_resolution": {"id": "1ZpmSXBpJ9pFEov6-jjQstAlfYbkebECu", "name": "psp_celebs_super_resolution.pt"},
}

path = MODEL_PATHS[experiment_type]
get_download_model_command(file_id=path["id"], file_name=path["name"])
