FROM python:3.6
RUN mkdir -p "/opt/app"

WORKDIR /opt/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 

COPY VGG16_Client.tflite .
COPY VGG16_Quantized_Client_gRPC.py .

ENTRYPOINT ["/bin/sh", "-c", "python3", "VGG16_Quantized_Client_gRPC.py"]
CMD ["dirPath"]