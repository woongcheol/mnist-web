# model_api.py

from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from io import BytesIO

# to test the service using curl:
# curl -X POST "http://127.0.0.1:8000/predict" -H  "accept: application/json" -H  "Content-Type: multipart/form-data" -F "image=@mnist_sample.jpg;type=image/jpeg"

# create a fastapi app instance.
app = FastAPI()

# load the keras model.
loaded_model = load_model('h5_model.h5')

@app.post("/predict")
# define a form with a multipart input, which will be the image in this case.
async def predict(image: UploadFile = File(...)):
    contents = await image.read()
    loaded_image = Image.open(BytesIO(contents))
    loaded_image = np.expand_dims(loaded_image, axis=0) # (1,28,28)
    loaded_image = loaded_image[..., np.newaxis] # (1,28,28,1)
    prediction = np.argmax(loaded_model.predict(loaded_image))
    return {"label": str(prediction)}