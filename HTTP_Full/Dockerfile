FROM python:3.6
RUN mkdir -p "/opt/app"

MAINTAINER Michael Rahaim <michael.rahaim@gmail.com>

WORKDIR /opt/app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt 

COPY VGG16_Client.h5 .
COPY VGG16_Full_Client_HTTP.py .

ENTRYPOINT ["python3", "VGG16_Full_Client_HTTP.py"]