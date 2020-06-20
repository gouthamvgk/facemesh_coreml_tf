# Face Mesh - Coreml

This repository contains the code for converting tflite models of blazeface and facemesh present in the [Mediapipe](https://github.com/google/mediapipe/tree/master/mediapipe/models) library to coreml and tensorflow. Blazeface is intended for realtime face detection while facemesh is used to detect 468-3D facial landmarks.

## Requirements

 - tensorflow == 2.2.0
 - coremltools == 3.4
 - matplotlib
 - opencv
 - PIL

## Blazeface conversion
Run `python convert_blazeface.py` to convert the tflite model of blazeface present in `tflite_models` folder to tensorflow and keras version. Converted models are placed in `keras_models` and `coreml_models` folders. Blazeface accepts inputs of size 128x128x3 and outputs 896 proposals where each proposal contains a bounding box and 6 facial landmarks along with confidence. NMS should be run on the proposal boxes to filter duplicates. Original mediapipe version uses weighted NMS while here normal NMS is used for simplicity. Use [Netron](https://github.com/lutzroeder/netron) to visualize the network from .h5 or .mlmodel file

##  Blazeface CoreML pipeline
Run `create_blazeface_coreml_pipeline.py` to create a pipeline CoreML model that includes blazeface and NMS in a single mlmodel. This model takes an input image and gives out the bounding boxes and confidence after NMS.

##  Facemesh conversion
Run `convert_facemesh.py` to convert tflite model of facemesh to CoreML and tensorflow. Facemesh takes an input image of shape 192x192x3 and outputs 468 3-D facial landmarks.

## Running on live video
Run `live_demo.py` to take the camera feed and run both the blazeface, facemesh in sequence. Input frame is first fed to blazeface to find the faces present in the image. After this step bounding boxes are slightly adjusted to have some extra space around face region as required by facemesh. These cropped faces are then fed to the facemesh network to find the landmarks. All the conversion between input image space and cropped face are handled in `utils.py`
![Results](https://github.com/gouthamvgk/facemesh_coreml_tf/blob/master/results/usain.gif)