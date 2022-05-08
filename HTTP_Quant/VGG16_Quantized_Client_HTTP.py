# # This program serves an HTTP inference to the client using the quantized client model

# ## Loading Libraries
import tensorflow
import numpy as np
import json
import requests
import sys
import os
from os import listdir
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

# HTTP endpoint for tensorflow/serving
endpoint = 'http://10.131.36.51:8501/v1/models/VGG16_Server:predict'


# ## Directory iteration & Inference via HTTP

# In[ ]:


# Default path that images are mounted to in the docker command if no command line argument
if len(sys.argv)==1:
    # This can change to whatever the dockerfile says the volume is mounted to
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
        # load an image from directory
        image = load_img(folder_dir+images, target_size=(224, 224))
        # Pre-processing of image when using quantized model
        input_shape = input_details[0]['shape']
        input_tensor= np.array(np.expand_dims(image,0), dtype=np.float32)
        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]
        interpreter.set_tensor(input_index, input_tensor)

        interpreter.invoke()
        output_details = interpreter.get_output_details()
        predictions = interpreter.get_tensor(output_index)
        data = json.dumps({"instances": predictions.tolist()})
        
        # HTTP POST
        json_response = requests.post(endpoint, data)
        prediction = json.loads(json_response.text)
        result = decode_predictions(np.array(prediction['predictions']), top=4)
        print(result)       

