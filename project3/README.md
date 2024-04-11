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
$ docker build -t costaki33/project3 .
$ docker run -it --rm -p 5000:5000 costaki33/project3
```
Now, load a Jupyter Notebook that has access to the Docker image 
