#! /usr/bin/env python

import os
from pathlib import Path

from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi import FastAPI, Depends
from starlette.background import BackgroundTask
from loguru import logger

from app.api.models import Box
from app.core.auth import api_key_auth
from app.core.config import get_app_settings
from app.networks.cfrgan_face_frontalizer import CFRGANFaceFrontalizer
from app.networks.gfpgan_face_ehnancer import GFPGANFaceEnhancer
from app.networks.ort_retinaface import ORTFaceDetector


def get_application() -> FastAPI:
    settings = get_app_settings()

    settings.configure_logging()

    application = FastAPI(**settings.fastapi_kwargs)

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_hosts,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.add_middleware(GZipMiddleware, minimum_size=1000)

    # application.add_event_handler(
    #     "startup",
    #     create_start_app_handler(application, settings),
    # )
    # application.add_event_handler(
    #     "shutdown",
    #     create_stop_app_handler(application),
    # )
    #
    # application.add_exception_handler(HTTPException, http_error_handler)
    # application.add_exception_handler(RequestValidationError, http422_error_handler)
    #
    # application.include_router(api_router, prefix=settings.api_prefix)

    return application


app = get_application()
logger.info("Starting up. Loading models.")

ASSETS_FOLDER = Path(__file__).resolve().parents[1] / 'app' / 'assets'
face_detector = ORTFaceDetector(ASSETS_FOLDER)
# face_enhancer = ORTRealESRGANFaceEnhancer(ASSETS_FOLDER)
face_enhancer = GFPGANFaceEnhancer(ASSETS_FOLDER)
face_frontalizer = CFRGANFaceFrontalizer(ASSETS_FOLDER)

logger.info("Models loaded...")


@app.get("/faces/", dependencies=[Depends(api_key_auth)])
def detect_faces(image_url: str):
    faces = face_detector.detect_faces(image_url=image_url)
    return faces


def remove_file(path: str) -> None:
    os.unlink(path)


@app.post("/enhance/", dependencies=[Depends(api_key_auth)])
def enhance_face(image_url: str,
                       box: Box) -> FileResponse:
    enhanced_face_path = face_enhancer.enhance_face(image_url=image_url, box=box)
    return FileResponse(enhanced_face_path,
                        media_type="image/png",
                        background=BackgroundTask(remove_file, enhanced_face_path), )


@app.post("/frontalize/", response_class=FileResponse, dependencies=[Depends(api_key_auth)])
def frontalize_face(image_url: str) -> FileResponse:
    enhanced_face_path = face_frontalizer.frontaliza_face(image_url=image_url)
    return FileResponse(enhanced_face_path,
                        media_type="image/png",
                        background=BackgroundTask(remove_file, enhanced_face_path), )


@app.get("/hello/{name}", dependencies=[Depends(api_key_auth)])
def say_hello(name: str):
    return {"message": f"Hello {name}"}
