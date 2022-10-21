# JunkViz Model
An AI Model capable of detecting and classifying trash, using the EfficientNet model.

## Using the model
To run the model on your machine, first make sure it supports CUDA and ensure it supports `CUDA >= 11.2`

Now, begin by building the docker image via `docker build .`. From there, run `bash docker.bash`

Next, download the pretrained efficient net model weights from [here](https://drive.google.com/drive/u/0/folders/1wNWoH8rdkG05sBw-OCXp3J73uJPxhcxH.)

Place weights under `src/detect-waste/efficeintdet`, and then run `src/detect-waste/demo_function.py` in order to start running the model
