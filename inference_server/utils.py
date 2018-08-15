import numpy as np
from PIL import Image
import io
from skimage import exposure
import distance
import pandas as pd
import re


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def boxes_to_im_bytes_list(im, boxes):
    images = []
    im_width, im_height = im.shape[1], im.shape[0]
    boxes = np.array(sorted(boxes, key=lambda x: (x[0], x[1])))
    for i, box in enumerate(boxes):
        ymin, xmin, ymax, xmax = box
        xmin, ymin, xmax, ymax = int(xmin * im_width), int(ymin * im_height), \
                                 int(xmax * im_width), int(ymax * im_height)
        padding = 3
        crop = im[ymin - padding:ymax + padding, xmin - padding:xmax + padding]
        crop_im = Image.fromarray(crop)
        with io.BytesIO() as output:
            crop_im.save(output, 'PNG')
            image_bytes = output.getvalue()
        images.append(image_bytes)

    return images


def boxes_to_np_crops(im, boxes, filter_bboxes=False):
    images = []
    boxes_abs = []
    im_width, im_height = im.shape[1], im.shape[0]
    # boxes = np.array(sorted(boxes, key=lambda x: (x[0])))
    for i, box in enumerate(boxes):
        ymin, xmin, ymax, xmax = box
        xmin, ymin, xmax, ymax = int(xmin * im_width), int(ymin * im_height), \
                                 int(xmax * im_width), int(ymax * im_height)
        boxes_abs.append((xmin, ymin, xmax, ymax))
    boxes_abs = np.array(boxes_abs)
    final_boxes = non_max_suppression_fast(boxes_abs, .3)
    final_boxes = np.array(sorted(final_boxes, key=lambda x: (x[1])))

    if filter_bboxes:
        df = pd.DataFrame(final_boxes, columns=('xmin', 'ymin', 'xmax', 'ymax'))
        df['w'] = df['xmax'] - df['xmin']
        df['h'] = df['ymax'] - df['ymin']
        # Remove label texts
        # get bboxes with height greater than 10
        df = df[df['h'] >= 10].reset_index(drop=True)
        final_boxes = df[['xmin', 'ymin', 'xmax', 'ymax']].values

    for i, box in enumerate(final_boxes):
        xmin, ymin, xmax, ymax = box
        padding = 5
        crop = im[ymin - padding:ymax + padding, xmin - padding:xmax + padding]
        crop = exposure.rescale_intensity(crop)
        images.append(crop)

    return images, final_boxes

def process_words_spanish_id_card_v3(words):
    labels = ['APELLIDOS', 'NOMBRE', 'SEXO', 'NACIONALIDAD', 'VALIDEZ', 'NUM SOPORT', 'FECHA DE NACIMIENTO']
    words_simil = []
    for word in words:
        w = replace_word_simil(word, labels, 2)
        words_simil.append(w)
    labels.append('DNI')
    words = [x for x in words_simil if x not in labels]

    num_unknown_type = 0
    num_word_type = 0
    words_type = {}
    for i, word in enumerate(words):
        # date
        regex = '^\d{2} \d{2} \d{4}$'
        m = re.search(regex, word)
        if m is not None and ('expires' not in words_type.keys() or 'dob' not in words_type.keys()):
            w_type = 'expires'
            if 'dob' not in words_type.keys():
                w_type = 'dob'
            words_type[w_type] = word
            continue
        # gender
        regex = '^(F|M)$'
        m = re.search(regex, word)
        if m is not None and 'gender' not in words_type.keys():
            w_type = 'gender'
            words_type[w_type] = word
            continue
        # nationality
        regex = '^[A-Z]{3}$'
        m = re.search(regex, word)
        if m is not None and 'nationality' not in words_type.keys():
            w_type = 'nationality'
            words_type[w_type] = word
            continue
        # id card num
        regex = '^([0-9]{8}[A-Z]{1})$'
        m = re.search(regex, word)
        if m is not None and 'id_num' not in words_type.keys():
            w_type = 'id_num'
            words_type[w_type] = word
            continue
        # support_num
        regex = '^[A-Z]{3}[0-9]{6}$'
        m = re.search(regex, word)
        if m is not None and 'support_num' not in words_type.keys():
            w_type = 'support_num'
            words_type[w_type] = word
            continue
        # can_num
        regex = '^[0-9]{6}$'
        m = re.search(regex, word)
        if m is not None and 'can_num' not in words_type.keys():
            w_type = 'can_num'
            words_type[w_type] = word
            continue
        # word/s
        regex = '^([A-Z\s])+$'
        m = re.search(regex, word)
        if m is not None:
            w_type = 'words'
            if i == 0 and 'last_name1' not in words_type.keys():
                w_type = 'last_name1'
            elif i == 1 and 'last_name2' not in words_type.keys():
                w_type = 'last_name2'
            elif i == 2 and 'name' not in words_type.keys():
                w_type = 'name'
            elif w_type == 'words':
                w_type = 'words{}'.format(num_word_type)
                num_word_type += 1
            words_type[w_type] = word
            continue
        words_type['unknown{}'.format(num_unknown_type)] = word
        num_unknown_type += 1
    return words_type

def process_words_spanish_id_card_v2(words):
    labels = ['PRIMER APELLIDO', 'SEGUNDO APELLIDO', 'NOMBRE', 'SEXO', 'NACIONALIDAD', 'VALIDO HASTA',
              'FECHA DE NACIMIENTO', 'DNI NUM']
    words_simil = []
    for word in words:
        w = replace_word_simil(word, labels, 2)
        words_simil.append(w)
    labels.append('IDESP')
    words = [x for x in words_simil if x not in labels]

    num_unknown_type = 0
    num_word_type = 0
    words_type = {}
    for i, word in enumerate(words):
        # date
        regex = '^\d{2} \d{2} \d{4}$'
        m = re.search(regex, word)
        if m is not None and ('expires' not in words_type.keys() or 'dob' not in words_type.keys()):
            w_type = 'expires'
            if 'dob' not in words_type.keys():
                w_type = 'dob'
            words_type[w_type] = word
            continue
        # gender
        regex = '^(F|M)$'
        m = re.search(regex, word)
        if m is not None and 'gender' not in words_type.keys():
            w_type = 'gender'
            words_type[w_type] = word
            continue
        # nationality
        regex = '^[A-Z]{3}$'
        m = re.search(regex, word)
        if m is not None and 'nationality' not in words_type.keys():
            w_type = 'nationality'
            words_type[w_type] = word
            continue
        # id card num
        regex = '^([0-9]{8}[A-Z]{1})$'
        m = re.search(regex, word)
        if m is not None and 'id_num' not in words_type.keys():
            w_type = 'id_num'
            words_type[w_type] = word
            continue
        # support_num
        regex = '^[A-Z]{3}[0-9]{6}$'
        m = re.search(regex, word)
        if m is not None and 'support_num' not in words_type.keys():
            w_type = 'support_num'
            words_type[w_type] = word
            continue
        # word/s
        regex = '^([A-Z\s])+$'
        m = re.search(regex, word)
        if m is not None:
            w_type = 'words'
            if i == 0 and 'last_name1' not in words_type.keys():
                w_type = 'last_name1'
            elif i == 1 and 'last_name2' not in words_type.keys():
                w_type = 'last_name2'
            elif i == 2 and 'name' not in words_type.keys():
                w_type = 'name'
            elif w_type == 'words':
                w_type = 'words{}'.format(num_word_type)
                num_word_type += 1
            words_type[w_type] = word
            continue
        words_type['unknown{}'.format(num_unknown_type)] = word
        num_unknown_type += 1
    return words_type

def process_words_spanish_driving_licence(words):
    labels = ['1', '2', '3', '4a', '4b', '4c', '5', '6', '7', '8', '9']
    words = [x for x in words if x not in labels]
    num_unknown_type = 0
    num_word_type = 0
    words_type = {}
    for i, word in enumerate(words):
        # date
        regex = '^\d{2} \d{2} \d{4}$'
        m = re.search(regex, word)
        if m is not None and (
                'expires' not in words_type.keys() or 'dob' not in words_type.keys() or 'expedited' not in words_type.keys()):
            w_type = 'expires'
            if 'dob' not in words_type.keys():
                w_type = 'dob'
            elif 'expedited' not in words_type.keys():
                w_type = 'expedited'
            words_type[w_type] = word
            continue
        # word/s
        regex = '^([A-Z\s])+$'
        m = re.search(regex, word)
        if m is not None:
            w_type = 'words'
            if 'last_name1' not in words_type.keys():
                w_type = 'last_name1'
            elif 'last_name2' not in words_type.keys():
                w_type = 'last_name2'
            elif 'name' not in words_type.keys():
                w_type = 'name'
            elif 'nationality' not in words_type.keys():
                w_type = 'nationality'
            elif w_type == 'words':
                w_type = 'codes{}'.format(num_word_type)
                num_word_type += 1
            words_type[w_type] = word
            continue
        # id card num
        regex = '^([0-9]{8}\s?[A-Z]{1})$'
        m = re.search(regex, word)
        if m is not None and 'id_num' not in words_type.keys():
            w_type = 'id_num'
            words_type[w_type] = word
            continue
        # expedited_by
        regex = '^[0-9]{2}\s?[0-9]{2}$'
        m = re.search(regex, word)
        if m is not None and 'expedited_by' not in words_type.keys():
            w_type = 'expedited_by'
            words_type[w_type] = word
            continue
        words_type['unknown{}'.format(num_unknown_type)] = word
        num_unknown_type += 1
    return words_type



def replace_word_simil(word, labels, thres=1):
    res_word = None
    for dict_label in labels:
        dist = distance.levenshtein(word, dict_label)
        if dist > 0 and dist <= thres:
            res_word = dict_label
            break
        else:
            res_word = word
    return res_word


# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


def invert_dict(d):
    return dict([(v, k) for k, v in d.items()])
