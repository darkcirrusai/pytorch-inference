"""
the main app accepting images for inference
"""
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from typing import Dict
import uvicorn
from location_model_test import main_inference
from utils.save_uploads import convert_b64_jpeg

app = FastAPI()


@app.get("/",
         response_class=JSONResponse)
def main_app():
    return {"message": "Hello World"}


@app.post("/predict",
          response_class=JSONResponse)
def predict(image: Dict[str, str] = Body(),):
    """
    takes in an image in base64 string and returns the prediction
    :param image:
    :return: prediction json
    """
    # save image path
    saved_image_path = "saved_images/test_image.jpg"

    # convert base64 to jpeg
    convert_b64_jpeg(image["images"], save_path=saved_image_path)

    box = main_inference(test_image_path=saved_image_path)
    return {"gauge_location": box}


if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8055, reload=True)
