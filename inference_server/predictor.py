import io
import numpy as np
from PIL import Image
import flask
from pdf2image import convert_from_bytes
import tensorflow as tf
import json
import utils
from aocr.predict import TextLineOCR
import time
import logging

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif'}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Inference Server')


def get_file_extension(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower()


# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class TextRecognitionService(object):
    model = None

    @classmethod
    def get_model(cls):
        if cls.model is None:
            cls.model = TextLineOCR()
        return cls.model

    @classmethod
    def predict(cls, images):
        words = []
        textline_res = cls.get_model().rec(images)
        for res in textline_res:
            word = ''.join([str(r, 'utf-8') for r in res])
            words.append(word)
        return words


class TextDetectionServiceSpanishIdCardV3(object):
    model = None

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            cls.model = tf.Graph()
            with cls.model.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile('detect_model_idcardv3.pb', 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
        return cls.model

    @classmethod
    def predict(cls, image_np, min_score_thres=0.5):
        detection_graph = cls.get_model()
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                classes = np.squeeze(classes).astype(np.int32)
                # Filter boxes by score threshold
                scores_filter = np.argwhere(scores[scores >= min_score_thres])
                scores = np.squeeze(scores[scores_filter])
                boxes = np.squeeze(boxes[scores_filter])
                classes = np.squeeze(classes[scores_filter])
                num_detections = len(boxes)
                return boxes, scores, classes, num_detections


class TextDetectionServiceSpanishIdCardV2(object):
    model = None

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            cls.model = tf.Graph()
            with cls.model.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile('detect_model_idcardv2.pb', 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
        return cls.model

    @classmethod
    def predict(cls, image_np, min_score_thres=0.5):
        detection_graph = cls.get_model()
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                classes = np.squeeze(classes).astype(np.int32)
                # Filter boxes by score threshold
                scores_filter = np.argwhere(scores[scores >= min_score_thres])
                scores = np.squeeze(scores[scores_filter])
                boxes = np.squeeze(boxes[scores_filter])
                classes = np.squeeze(classes[scores_filter])
                num_detections = len(boxes)
                return boxes, scores, classes, num_detections


class TextDetectionServiceSpanishDrivingLicence(object):
    model = None

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            cls.model = tf.Graph()
            with cls.model.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile('detect_model_driving_licence.pb', 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
        return cls.model

    @classmethod
    def predict(cls, image_np, min_score_thres=0.5):
        detection_graph = cls.get_model()
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                classes = np.squeeze(classes).astype(np.int32)
                # Filter boxes by score threshold
                scores_filter = np.argwhere(scores[scores >= min_score_thres])
                scores = np.squeeze(scores[scores_filter])
                boxes = np.squeeze(boxes[scores_filter])
                classes = np.squeeze(classes[scores_filter])
                num_detections = len(boxes)
                return boxes, scores, classes, num_detections


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping_ocr():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = TextRecognitionService.get_model() is not None \
             and TextDetectionServiceSpanishIdCardV3.get_model() is not None \
             and TextDetectionServiceSpanishIdCardV2.get_model() is not None \
             and TextDetectionServiceSpanishDrivingLicence.get_model() is not None
    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')


@app.route('/ocr:spanish_id_card3', methods=['POST'])
def ocr_spanish_id_card_v3():
    return _run_ocr(TextDetectionServiceSpanishIdCardV3, utils.process_words_spanish_id_card_v3)

@app.route('/ocr:spanish_id_card2', methods=['POST'])
def ocr_spanish_id_card_v2():
    return _run_ocr(TextDetectionServiceSpanishIdCardV2, utils.process_words_spanish_id_card_v2)

@app.route('/ocr:spanish_driving_licence', methods=['POST'])
def ocr_spanish_driving_licence():
    return _run_ocr(TextDetectionServiceSpanishDrivingLicence, utils.process_words_spanish_driving_licence)

def _run_ocr(detection_service, process_fn):
    if flask.request.files.get('file'):
        logger.info('Processing input image')
        start_time = time.time()
        file = flask.request.files['file']
        file_extension = get_file_extension(file.filename)
        if file and file_extension in ALLOWED_EXTENSIONS:
            if file_extension == 'pdf':
                pdf = file.read()
                images = convert_from_bytes(pdf, fmt='jpg')
                image = images[0]
            else:
                image = Image.open(io.BytesIO(file.read()))
                if file_extension == 'png' or file_extension == 'gif':
                    image = image.convert('RGB')
            image = image.resize((640, 390), resample=Image.ANTIALIAS)
            image_np = utils.load_image_into_numpy_array(image)

            logger.info('Running text detection')
            boxes, scores, classes, num_detections = detection_service.predict(image_np)
            logger.info('Text detection performed')
            # preprocess the image and prepare it for classification
            logger.info('Running text recognition')
            word_images, final_boxes = utils.boxes_to_np_crops(image_np, boxes, filter_bboxes=True)
            words = TextRecognitionService.predict(word_images)
            words_dict = process_fn(words)
            logger.info('Text recognition performed')
            js = json.dumps(words_dict)
            exec_time = time.time() - start_time
            logger.info('Execution time: {}'.format(exec_time))
            return flask.Response(response=js, status=200, mimetype='application/json')
        else:
            return flask.Response(response='File extension not allowed. Allowed:[pdf, png, jpg, jpeg, gif]',
                                  status=415, mimetype='text/plain')
    else:
        return flask.Response(response='Parameter [file] is required.', status=415, mimetype='text/plain')