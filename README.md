# progressiveperception
This project will enable progressive perception of items using Machine Learning and Edge-Cloud systems.

The 4 directories are created for the 4 containers that load the quantized or non-quantized version of the client model and use either gRPC or HTTP to contact the server.

Each directory has an identical requirements.txt file with the software dependencies needed for docker containerization.
Each directory also has a Dockerfile that instructs Docker on how to build the image. The Dockerfile is mostly complete, but there is still a problem with containerization.

The images directory holds three different images used for testing purposes.

The python scripts in the gRPC_Quant and HTTP_Quant directories accept a directory path as a command line argument. This path leads to the directory where the images are located.

The python scripts can be run on their own. The python script operates on a directory of images. 
    $ python3 <name-of-script>.py /path/to/images/
    
In order to build a docker container using this Dockerfile:
    cd /path/of/Dockerfile
    docker build -t <name-of-image> .

In order to run the successfully created Docker image:
    cd /local/path/to/img/dir:/container/img/path <name of >
