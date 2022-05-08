# progressiveperception
This project will enable progressive perception of items using Machine Learning and Edge-Cloud systems.

The 4 directories are created for the 4 containers that load the quantized or non-quantized version of the client model and use either gRPC or HTTP to contact the server.

Each directory has an identical requirements.txt file with the software dependencies needed for docker containerization.
Each directory also has a Dockerfile that instructs Docker on how to build the image. The Dockerfile is mostly complete, but there is still a problem where the docker container will not run the python script.

The images directory holds three different images locally used for testing purposes.

When running via python3, the python script in each directory takes a single command line argument for the path to where the images are stored. When running via the container, you don't need a command line argument, but you will need to change the default directory in the python script file based on the container and dockerfile working directories.

The python scripts can be run on their own. The python script operates on a directory of images. 
    $ python3 <name-of-script>.py /path/to/images/
    
I have included python files that partition the models and quantize them as well for reference.
    
The server side uses the TensorFlow/Serving container from Docker and loads the server model into it.
    
*************************
****** IN PROGRESS ******
*************************
    
In order to build a docker container using this Dockerfile:
    cd /path/of/Dockerfile
    docker build -t <name-of-image> .

In order to run the successfully created Docker image:
    docker run -v /local/image/dir:/container/image/dir <name-of-image>
