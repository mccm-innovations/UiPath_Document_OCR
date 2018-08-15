import os
import numpy as np
from PIL import Image
import random, math
from .bucketdata import BucketData
from .voc_keys import alphabet
alphabet = [e.encode('utf-8') for e in alphabet]
alphabet2idx_map = {}
alphabet2lex_map = {}
for a_idx, lex in enumerate(alphabet):
    alphabet2idx_map[lex] = a_idx + 3
    alphabet2lex_map[u'%d'%(a_idx + 3)] = lex

alphabet2idx_map[u'BOS'] = 1
alphabet2idx_map[u'EOS'] = 2
alphabet2lex_map[u'1'] = u'BOS'
alphabet2lex_map[u'2'] = u'EOS'

class DataGen(object):
    GO = 1
    BOS = 1
    EOS = 2

    def __init__(self,
                 data_root, annotation_fn,
                 evaluate = False,
                 valid_target_len = float('inf'),
                 img_width_range = (12, 320),
                 word_len = 70):
        """
        :param data_root:
        :param annotation_fn:
        :param lexicon_fn:
        :param img_width_range: only needed for training set
        :return:
        """

        img_height = 32
        self.data_root = data_root
        if os.path.exists(annotation_fn):
            self.annotation_path = annotation_fn
        else:
            self.annotation_path = os.path.join(data_root, annotation_fn)

        if evaluate:
            self.bucket_specs = [(int(math.floor(64 / 4)), int(word_len + 2)), (int(math.floor(108 / 4)), int(word_len + 2)),
                                 (int(math.floor(140 / 4)), int(word_len + 2)), (int(math.floor(256 / 4)), int(word_len + 2)),
                                 (int(math.floor(img_width_range[1] / 4)), int(word_len + 2))]
        else:
            self.bucket_specs = [(int(64 / 4), 9 + 2), (int(108 / 4), 15 + 2),
                             (int(140 / 4), 17 + 2), (int(256 / 4), 20 + 2),
                             (int(math.ceil(img_width_range[1] / 4)), word_len + 2)]

        self.bucket_min_width, self.bucket_max_width = img_width_range
        self.image_height = img_height
        self.valid_target_len = valid_target_len

        self.bucket_data = {i: BucketData()
                            for i in range(self.bucket_max_width + 1)}

    def clear(self):
        self.bucket_data = {i: BucketData()
                            for i in range(self.bucket_max_width + 1)}

    def get_size(self):
        with open(self.annotation_path, 'r') as ann_file:
            return len(ann_file.readlines())

    def gen(self, batch_size):
        valid_target_len = self.valid_target_len
        with open(self.annotation_path, 'r') as ann_file:
            lines = ann_file.readlines()
            random.shuffle(lines)
            for l in lines:
                img_path, lex = l.strip().split()
                try:
                    img_bw, word = self.read_data(img_path, lex)
                    if valid_target_len < float('inf'):
                        word = word[:valid_target_len + 1]
                    width = img_bw.shape[-1]

                    b_idx = min(width, self.bucket_max_width)
                    bs = self.bucket_data[b_idx].append(img_bw, word, os.path.join(self.data_root,img_path))
                    if bs >= batch_size:
                        b = self.bucket_data[b_idx].flush_out(
                                self.bucket_specs,
                                valid_target_length=valid_target_len,
                                go_shift=1)
                        if b is not None:
                            yield b
                        else:
                            assert False, 'no valid bucket of width %d'%width
                except IOError:
                    pass # ignore error images
        self.clear()


    def read_data(self, img_path, lex):
        # L = R * 299/1000 + G * 587/1000 + B * 114/1000
        with open(os.path.join(self.data_root, img_path), 'rb') as img_file:
            img = Image.open(img_file)
            w, h = img.size
            aspect_ratio = float(w) / float(h)
            if aspect_ratio < float(self.bucket_min_width) / self.image_height:
                img = img.resize(
                    (self.bucket_min_width, self.image_height),
                    Image.ANTIALIAS)
            elif aspect_ratio > float(
                    self.bucket_max_width) / self.image_height:
                img = img.resize(
                    (self.bucket_max_width, self.image_height),
                    Image.ANTIALIAS)
            elif h != self.image_height:
                img = img.resize(
                    (int(aspect_ratio * self.image_height), self.image_height),
                    Image.ANTIALIAS)

            img_bw = img.convert('L')
            img_bw = np.asarray(img_bw, dtype=np.uint8)
            img_bw = img_bw[np.newaxis, :]

        # 'a':97, '0':48
        word = [self.GO]
        if not len(lex) < self.bucket_specs[-1][1]:
            lex = lex[0:self.bucket_specs[-1][1]-1]
        for c in lex:
            utf_lex = c.encode('utf-8')
            if utf_lex not in alphabet2idx_map:
                print("WARNING : UNKNOW CHARACTER {}".format(utf_lex))
                #utf_lex = u'UNK'
                utf_lex = ' '
            word.append(alphabet2idx_map[utf_lex])
        word.append(self.EOS)
        word = np.array(word, dtype=np.int32)

        return img_bw, word