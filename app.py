from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import Union
import shutil
from pathlib import Path
from io import BytesIO

import cv2                                                        
import numpy as np
from keras.models import load_model
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageEnhance


app = FastAPI()

upload_dir = Path("uploaded_images")

upload_dir.mkdir(parents=True, exist_ok=True)

def load_ml_model():
    model=load_model('glaucoma.h5')
    print("Model Loaded")
    return model

model = load_ml_model()

def read_Imagefile(file_content: bytes):
    image = Image.open(BytesIO(file_content))
    return image

def predict(image: Image.Image):

    '''test_image = image.load_img('/content/datasets/val/glau/1 ('+str(i)+').png',                           target_size=(240,240))
    test_image = image.img_to_array(test_image)'''

    image = np.asarray(image.resize((240, 240)))[..., :3]

    image = np.expand_dims(image, axis=0)
    
    # image = image / 127.5 - 1.0

    result = decode_predictions(model.predict(image), 2)[0]
    response = []
    for i, res in enumerate(result):
        resp = {}
        resp["class"] = res[1]
        resp["confidence"] = f"{res[2]*100:0.2f} %"
        response.append(resp)

@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/api/upload")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        extensions = file.filename.split('.')[-1] in ("jpg","jpeg","png")

        if not extensions:
            #TODO:change status code also
             return JSONResponse(content={"error": "Image should in jpg, jpeg or png format"}, status_code=400)
        else:
            
            file_path = upload_dir / file.filename
            with file_path.open('wb') as buffer:
                shutil.copyfileobj(file.file, buffer)

            print("===File Uploaded Successfully!!!===")

                    # Read the uploaded image using the read_Imagefile function
            file.file.seek(0)  # Reset the file pointer to the beginning
            uploaded_image = read_Imagefile(file.file.read())


            print("===Read the Image===")

            # prediction = predict(image)

            # print("===Doing Prediction")

            
            return {"filename": file.filename, "file_path": str(file_path), "width": uploaded_image.width, "height": uploaded_image.height}
        

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


