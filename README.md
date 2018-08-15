# UiPath Document OCR
## Inspiration
The line between RPA and Artificial Intelligence is getting closer. We provide a cognitive Optical Character Recognition (OCR) engine developed with Deep Neural Networks that can be deployed in a server and used from a RPA project through an API.

This cognitive OCR engine has been specifically designed to extract text from images of Spanish ID cards and Spanish driving licences.

For instance, this project may be used to extract texts from ID cards sent by customers and match them with the information of the company's database.

## What it does
UiPath makes an API call to the inference server in order to apply OCR to the given PDF or image file.

## How we built it
We built this in UiPath using the HTTP Request Activity which makes an API call to the inference server.  

The inference server is built with flask and gunicorn. It exposes an API endpoint that applies a pipeline of text detection and text recognition using different models trained with Tensorflow. Results are returned in JSON format.

More details about how to deploy an inference server can be found in [Configuring an inference server](https://github.com/mccm-innovations/UiPath_Document_OCR/tree/master/inference_server).

## Video demo
[![UiPath Document OCR Video Demo](https://img.youtube.com/vi/gFfVApTNiqI/0.jpg)](https://www.youtube.com/watch?v=gFfVApTNiqI "UiPath Document OCR Video Demo")
