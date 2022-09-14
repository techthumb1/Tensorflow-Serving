import matplotlib.pyplot as plt
import requests
import json
import numpy as np
from tensorflow.keras.datasets.mnist import load_data

#load MNIST dataset
(_, _), (x_test, y_test) = load_data()

# reshape data to have a single channel
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
# normalize pixel values
x_test = x_test.astype('float32') / 255.0

url = 'http://localhost:8501/v1/models/img_classifier:predict'

def make_predictions(instances):
    data = json.dumps({"signature_name": "serving_default", "instances": instances.tolist()})
    headers = {"content-type": "application/json"}
    json_response = requests.post(url, data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']
    return predictions

# get predictions for the first 4 images
predictions = make_predictions(x_test[:4])

for pred in predictions:
    print(np.argmax(pred))

for i, pred in enumerate(predictions):
    print(f"True Value: {y_test[i]}, Predicted Value: {np.argmax(pred)}")