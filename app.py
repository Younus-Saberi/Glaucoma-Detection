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

def predict(image: Image.Image) -> dict:
    model = load_model('glaucoma.h5')
    test_image = image.resize((240,240))
    test_image_array = np.array(test_image)
    test_image_array = np.expand_dims(test_image_array, axis=0)
    result = model.predict(test_image_array)
    if result[0][0]!=1:
        return {"Prediction": "Glaucoma"}
    else:
        return {"Prediction":"Normal"}

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


            print("===Doing Prediction")

            prediction_result = predict(uploaded_image)

            # Return the prediction result as JSON response
            return JSONResponse(content=prediction_result)
        

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


