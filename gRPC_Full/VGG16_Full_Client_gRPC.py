# ## Loading Libraries
import tensorflow
import numpy as np
import json
import requests
import sys
import os
from os import listdir
from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications.vgg16 import VGG16
# Loading the pre-trained non-quantized client model
model = tensorflow.keras.models.load_model('VGG16_Client.h5', compile=False)


# ## gRPC function(same as non-quantized)

# Identical to quantized gRPC function
# Only the image pre-processing is different
endpoint = 'http://10.131.36.51:8501/v1/models/VGG16_Server:predict'
headers = {"content-type": "application/json"}
def run_prediction(data,headers,endpoint):
    json_response = requests.post(endpoint,data=data,headers=headers)
    response = json.loads(json_response.text)
    yhat = response['predictions']
    #Converting back to array
    yhat = np.array(yhat)
    label = decode_predictions(yhat)
    return (label)


# ## Pre-processing of image

# The main difference in python code from quantized to non-quantized is the pre-processing of the image
def pre_process (img):
    # load an image from file
    image = load_img(img, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the client model
    image = preprocess_input(image)
    return (image)


# ## Directory iteration

# Default path images are mounted to in the docker command if no command line argument
if len(sys.argv)==1:
    folder_dir = "/opt/app/"
# Only used if client supplies a path(I don't think this is needed at all. Might just be around from an old design)
else:
    folder_dir = sys.argv[1]
# prints if image directory is empty of all file types(for testing purposes to make sure input was correct)
if len(os.listdir(folder_dir))==0:
    print("Empty directory")
    print("folder_dir = "+folder_dir)
    print("Directory = "+os.listdir(folder_dir))
# Iterates through the directory, grabs .jpg or .jpeg, pre-processes them, and serves the inference to the server
for images in os.listdir(folder_dir):
    if images.endswith(".jpg") or images.endswith(".jpeg"):
        print("Serving inference for "+images)
        # Pre-processing of image for non-quantized model
        image = pre_process(folder_dir+images)
        # Intermediate prediction after client model
        yhat = model.predict(image)
        # Sending to gRPC
        data = json.dumps({"instances": yhat.tolist()})
        # Printing of gRPC inference result
        print(run_prediction(data,headers,endpoint))
