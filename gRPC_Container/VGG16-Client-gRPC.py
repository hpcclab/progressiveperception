#!/usr/bin/env python
# coding: utf-8
# Inference with Quantized Client model and Non-quantized Server model
# ## Loading libraries
import tensorflow
import numpy as np
import json
import requests
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions


# ## Creating gRPC inference function

# Establishing gRPC endpoint and header to indicate the use of json
endpoint = 'http://10.131.36.51:8501/v1/models/VGG16_Server:predict'
headers = {"content-type": "application/json"}
def run_prediction(data,headers,endpoint):
    json_response = requests.post(endpoint,data=data,headers=headers)
    response = json.loads(json_response.text)
    yhat = response['predictions']
    # Converting back to numpy array for decoding to human readable text
    yhat = np.array(yhat)
    label = decode_predictions(yhat)
    return (label)


# ## Loading Quantized client model
interpreter = tensorflow.lite.Interpreter(model_path="VGG16_Client.tflite")
# Allocating tensors
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# ## Inference using quantized client model

# load an image from file
image = load_img('cat.jpg', target_size=(224, 224))
# convert the image pixels to a numpy array
input_shape = input_details[0]['shape']
input_tensor= np.array(np.expand_dims(image,0), dtype=np.float32)
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]
interpreter.set_tensor(input_index, input_tensor)

interpreter.invoke()
output_details = interpreter.get_output_details()
predictions = interpreter.get_tensor(output_index)


# ## Sending tensor to TensorFlow server to process the rest of the NN

# In[5]:


data = json.dumps({"instances": predictions.tolist()})
run_prediction(data,headers,endpoint)


# In[ ]:




