# # Partitioning VGG16

# We are going to partition VGG16 into two models, one consists of convolutional model(client) and one consist of fully connected network(server)

import numpy as np
import dill
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import decode_predictions
from tensorflow import keras
import zipfile
model = VGG16()

#Fetching VG16 layers and defining in layer_x format
for i in range(len(model.layers)):
    exec(f'layer_{i}=model.get_layer(index={i})')


print(model.summary())


# Generating Convolutional partition of the model
# This model will be used by the client
Client_model = keras.Sequential([
    layer_0,
    layer_1,
    layer_2,
    layer_3,
    layer_4,
    layer_5,
    layer_6,
    layer_7,
    layer_8,
    layer_9,
    layer_10,
    layer_11,
    layer_12,
    layer_13,
    layer_14,
    layer_15,
    layer_16,
    layer_17,
    layer_18,
    layer_19,
])

# Generating Fully Connected partition of the model
# This model will be on the server
Server_model = keras.Sequential([
    layer_20,
    layer_21,
    layer_22,
])

# Saving models
Client_model.save('VGG16_Client.h5')
Server_model.save('VGG16_Server')





