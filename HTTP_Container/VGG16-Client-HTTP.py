#!/usr/bin/env python
# coding: utf-8

# ## Loading libraries


import tensorflow
import numpy as np
import json
import requests
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions


# ## Loading quantized client model
interpreter = tensorflow.lite.Interpreter(model_path="VGG16_Client.tflite")
# Allocation of tensors
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# ## Loading and Pre-processing of Image
# load an image from file
image = load_img('lion.jpg', target_size=(224, 224))
# convert the image pixels to a numpy array
input_shape = input_details[0]['shape']
input_tensor= np.array(np.expand_dims(image,0), dtype=np.float32)
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]
interpreter.set_tensor(input_index, input_tensor)

interpreter.invoke()
output_details = interpreter.get_output_details()
predictions = interpreter.get_tensor(output_index)


# Establishment of REST api endpoint
endpoint = 'http://10.131.36.51:8501/v1/models/VGG16_Server:predict'
data = json.dumps({"instances": predictions.tolist()})


# ## REST implementation
# POST to server for inference
json_response = requests.post(endpoint, data)
prediction = json.loads(json_response.text)
# Decodes Top 4 predictions by confidence into human readable form after they have been converted to a numpy array
result = decode_predictions(np.array(prediction['predictions']), top=4)
print(result)





