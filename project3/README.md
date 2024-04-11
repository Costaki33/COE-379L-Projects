# Hurricane Harvey Damage Prediction with the Use of Neural Networks 

The repository focuses on classifying different satellite images from the Hurricane Harvey disaster. We are seeking to classify houses if they are "damaged" or "not damaged" using TensorFlow-based Neural Networks. We explored 3 different models: Artificial Neural Network, LeNet-5 & Alternate LeNet-5 Convolutional Neural Networks. We deployed the NN's using a Flask-based inference server for you the user to interact with. 

Note: Not all models have been uploaded to Github as they exceed the maximum file size that can be uploaded. 

## Using the Container 
Run: 
```
$ git clone https://github.com/Costaki33/COE-379L-Projects.git
$ cd COE-379L-Projects/project3
```
Vim into the Dockerfile and change where it says 'costaki33' with your Dockerhub login information. Be sure to log into to Docker on your terminal or you won't be able to run the following commands: 
```
$ ubuntu@costaki-coe379vm:~/COE-379L-Projects/project3$ cat Dockerfile 
# Image: costaki33/project3 <- Change this to your user 

FROM python:3.11

RUN pip install tensorflow==2.15
RUN pip install Flask==3.0

COPY models ./models
COPY api.py ./api.py

CMD ["python3", "api.py"]

$ docker build -t costaki33/project3 .
$ docker run -it --rm -p 5000:5000 costaki33/project3
```
Now, load a Jupyter Notebook that has access to the Docker image. We can run HTTP GET & POST Requests
```
import requests

# make the GET request:
rsp = requests.get("http://172.17.0.1:5000/model/info")

# print the json response
rsp.json()

{
    'description': 'Classifies images containing damaged and undamaged buildings from Hurricane Harvey',
    'name': 'altlenet5',
    'version': 'v1'
}
```
```
import numpy as np
from PIL import Image

l = np.array(Image.open('./data/split/test/damage/-93.528502_30.987438.jpeg')).tolist()

# make the POST request passing the sinlge test case as the `image` field:
rsp = requests.post("http://172.17.0.1:5000/model/predict", json={"image": l})

# print the json response
rsp.json()

{'result': [[0.0]]}
```
The expected output from the POST command is a prediction array that binarily states [damaged, not_damaged]. For example, if a house is damaged, it will output [1.0, 0.0]
