from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
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
            #TODO: convert image to suitable format 
            return JSONResponse(content={"error": "Image should in jpg, jpeg or png format"}, status_code=400)
        else:
            
            file_path = upload_dir / file.filename

            with file_path.open('wb') as buffer:
                shutil.copyfileobj(file.file, buffer)

            print("===File Uploaded Successfully!!!===")

                    # Read the uploaded image using the read_Imagefile function
            file.file.seek(0)  # Reset the file pointer to the beginning
            uploaded_image = read_Imagefile(file.file.read())


            print("===Reading the Image===")


            print("===Doing Prediction")

            prediction_result = predict(uploaded_image)

            # Return the prediction result as JSON response
            return JSONResponse(content=prediction_result)
        

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/image-processing")
async def image_processing(file: UploadFile = File(...)):
    try:

        extensions = file.filename.split('.')[-1] in ("jpg","jpeg","png")

        if not extensions:
            #TODO: convert image to suitable format 
            return JSONResponse(content={"error": "Image should in jpg, jpeg or png format"}, status_code=400)
        else:
            
            '''file_path = 'uploaded_images\cirrus cloud.jpg'

            with file_path.open('wb') as buffer:
                shutil.copyfileobj(file.file, buffer)'''

            print("===File Uploaded Successfully!!!===")

                    # Read the uploaded image using the read_Imagefile function
            '''file.file.seek(0)  # Reset the file pointer to the beginning
            uploaded_image = read_Imagefile(file.file.read())'''
            path = r'uploaded_images\\glaucoma.jpg'
            res_path = r'temp\\glaucoma2.jpg'

            process_image_ellipse(path)
            print("function called")
            # Return the prediction result as JSON response
            return FileResponse(path=res_path, status_code=200)
        

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
def process_image_ellipse(path:str):
    #load image in cv2
    img = cv2.imread(path,cv2.IMREAD_COLOR)

    #define ellipse parameters
    center = (100,100)
    axes_length = (50,30)
    angle = 30

    #draw ellipse on image
    color = (0,255,0)
    thickness = 2
    cv2.ellipse(img, center, axes_length, angle, 0,360,color, thickness)

    cv2.imwrite('temp\glaucoma2.jpg', img)

    