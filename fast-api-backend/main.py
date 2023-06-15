import tensorflow as tf
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from io import BytesIO

app = FastAPI(timeout=300)
    
def images_preprocessing(image):
    image = tf.io.decode_image(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = image / 255.
    image_tensor = tf.expand_dims(image, 0)
    image_tensor = image_tensor.numpy().tolist()
    return image_tensor

@app.get("/")
async def root():
    return {"message": "Hello World"}
    
@app.post("/predict")
async def predict_nsfw(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        image_tensor = images_preprocessing(image_bytes)
        json_data = {
            "instances": image_tensor
        }
        endpoint = "https://nsfw-model-xq5fdaetqa-et.a.run.app//v1/models/nsfw_model:predict"
        # endpoint = "http://localhost:8605/v1/models/nsfw_model:predict"
        response = requests.post(endpoint, json=json_data)
        response_data = response.json()
        if "predictions" not in response_data:
            raise HTTPException(status_code=500, detail={"result": "Unexpected response from NSFW model"})
        prediction = tf.argmax(response_data['predictions'][0]).numpy()
        result = round(response_data['predictions'][0][0])
        map_labels = {0: "Nude", 1: "Safe"}
        return {"result": map_labels[result]}
    except Exception as e:
        raise HTTPException(status_code=500, detail={"result": "Internal Error"}) from e

