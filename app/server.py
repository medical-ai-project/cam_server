from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from zipfile import ZipFile
from PIL import Image
import os
import numpy as np
import io

# import my func
from efficientNet_model import generate_grad_cam_images, get_prediction
from tools import delete_all_files_in_folder

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "efficientNet_xray4.h5")
model = load_model(model_path)


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_stream = io.BytesIO(await file.read())
    image = Image.open(image_stream).convert('RGB')
    image = image.resize((600, 600))
    image_array = np.array(image) / 255.0 
    input_img = np.expand_dims(image_array, axis=0)

    loss = get_prediction(input_img, model)
    return {"loss": loss.tolist()[0]}  # Numpy array to list


@app.post("/generate_cam/")
async def generate_cam(file: UploadFile = File(...)):
    image_stream = io.BytesIO(await file.read())
    image = Image.open(image_stream).convert('RGB')
    image = image.resize((600, 600))
    image_array = np.array(image) / 255.0 
    input_img = np.expand_dims(image_array, axis=0)

    folder_path = os.path.join(base_dir, "output")
    delete_all_files_in_folder(folder_path)
    generate_grad_cam_images(input_img, model)
    
    # Zip the generated images
    zip_file_path = os.path.join(base_dir, "output", "cam_images.zip") 
    with ZipFile(zip_file_path, 'w') as zipObj:
        for i in range(11):
            file_path = os.path.join(base_dir, "output", f"grad_cam{i}.png")
            zipObj.write(file_path, f'grad_cam{i}.png')

    return FileResponse(zip_file_path)
