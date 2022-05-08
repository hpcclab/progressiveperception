# # Quantization of Client_model for VGG16
# This converts the float32 values into float16 values, halving the size of the model

# ## Loading the model
import tensorflow as tf
model = tf.keras.models.load_model('VGG16_Client.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# ## Quantization and writing to the disk
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
open("VGG16_Client.tflite", "wb").write(tflite_quant_model)

