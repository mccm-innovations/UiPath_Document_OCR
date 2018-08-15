# Inference server
## Dependencies

- [NGINX](https://www.nginx.com/). You can install it with `sudo apt-get install nginx` if you are using an Ubuntu OS.
- Library dependencies for the Python code.  You need to install these with
`pip install -r requirements.txt` before you can run this.

**NOTE:** This server has been tested in Ubuntu 16.04 and Python 3.5.

## How to run it
### Download models
Several text detection and recognition models are needed to use this server. They can be downloaded from Google Drive:

- [Spanish ID Card 2 Text Detection](https://drive.google.com/open?id=17o-RadqelGHsFlPXVEB1jkKFXmzhyzZd)
- [Spanish ID Card 3.0 Text Detection](https://drive.google.com/open?id=1pg7255H50DNc4_IIpMastj98SVjhJ9Tv)
- [Spanish Driving Licence Text Detection](https://drive.google.com/open?id=1ZuhQ8pcJaTxc0YVkDC_P8x_-POi1qnno)
- [Text Recognition](https://drive.google.com/open?id=1dAStR947m_TrgRBk54kvKyN4YQaF4Ch4)

Please, move the text detection model files to the **inference_server** folder. Also, unzip the recognition model file and create an environment variable named `TEXT_RECOG_MODEL_PATH` with the absolute path to the unziped folder.

### Environment variables
Set the following environment variables:

| Parameter         | Environment Variable | Default Value           |
|-------------------|----------------------|-------------------------|
| number of workers | MODEL_SERVER_WORKERS | the number of CPU cores |
| timeout           | MODEL_SERVER_TIMEOUT | 800 seconds             |
| nginx config path | NGINX_CONF_PATH      | /etc/nginx/nginx.conf   |
| text recog model path | TEXT_RECOG_MODEL_PATH      | -   |

Example:
```bash
export MODEL_SERVER_WORKERS=1
export MODEL_SERVER_TIMEOUT=800
export NGINX_CONF_PATH=/home/user/UiPath_Document_OCR/inference_server/nginx.conf
export TEXT_RECOG_MODEL_PATH=/home/user/UiPath_Document_OCR/inference_server/recog_model
```
### Run the inference server
By default, this server uses the port 1234. Run it with the following command:
```bash
sudo -E ./serve
```
### Time to try it!
Send a file to the server using curl:
```bash
curl -X POST -F "file=@PATH_TO_YOUR_FILE" "http://localhost:1234/ocr:spanish_id_card3"
```
Result:
```json
{  
    "name":"JOHN",  
    "last_name1":"DOE",  
    "last_name2":"DOE",  
    "nationality":"ESP",  
    "gender":"M",  
    "support_num":"ABC123456",  
    "dob":"01 01 1980",  
    "expires":"01 01 2030",  
    "id_num":"12345678J",  
    "can_num":"123456",  
}
```

## Acknowledgements
### Text Recognition
Our text recognition model is based on a model by [Qi Guo](http://qiguo.ml/) and [Yuntian Deng](https://github.com/da03). You can find the original model here: [da03/Attention-OCR](https://github.com/da03/Attention-OCR).
### Text Detection
[Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
```
"Speed/accuracy trade-offs for modern convolutional object detectors."
Huang J, Rathod V, Sun C, Zhu M, Korattikara A, Fathi A, Fischer I, Wojna Z,
Song Y, Guadarrama S, Murphy K, CVPR 2017
```
