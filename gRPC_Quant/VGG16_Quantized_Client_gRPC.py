
print("test test test")

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


# ## Creating gRPC inference function

# In[24]:


endpoint = 'http://10.131.36.51:8501/v1/models/VGG16_Server:predict'
headers = {"content-type": "application/json"}
def run_prediction(data,headers,endpoint):
    json_response = requests.post(endpoint,data=data,headers=headers)
    response = json.loads(json_response.text)
    yhat = response['predictions']
    #Converting back to array
    yhat = np.array(yhat)
    label = decode_predictions(yhat)
    #label = label[0][0]
    return (label)


# ## Loading Quantized client model

# In[25]:


interpreter = tensorflow.lite.Interpreter(model_path="VGG16_Client.tflite")
#allocate the tensors
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# ## Inference using quantized client model

# In[28]:




# ## Iterating a dir for all images .jpg & .jpeg

# In[ ]:


folder_dir = sys.argv[1]
secondary_dir = "/opt/app/images/"
if len(os.listdir(folder_dir))==0:
    print("Empty directory")
    print("folder_dir = "+folder_dir)
    print("Directory = "+os.listdir(folder_dir))
for images in os.listdir(folder_dir):
    if images.endswith(".jpg") or images.endswith(".jpeg"):
        image = load_img(folder_dir+images, target_size=(224, 224))
        print("Serving inference for "+images)
        # convert the image pixels to a numpy array
        input_shape = input_details[0]['shape']
        input_tensor= np.array(np.expand_dims(image,0), dtype=np.float32)
        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]
        interpreter.set_tensor(input_index, input_tensor)

        interpreter.invoke()
        output_details = interpreter.get_output_details()
        predictions = interpreter.get_tensor(output_index)
        data = json.dumps({"instances": predictions.tolist()})
        print(run_prediction(data,headers,endpoint))


# ## Sending tensor to TensorFlow server to process the rest of the NN

# In[29]:


#data = json.dumps({"instances": predictions.tolist()})
#run_prediction(data,headers,endpoint)


# In[ ]:




