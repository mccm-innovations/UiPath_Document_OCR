from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import math
from PIL import Image
import tensorflow as tf

from .model.cnn import CNN
from .model.seq2seq_model import Seq2SeqModel
from .data_util.data_gen import alphabet2lex_map
from .data_util.bucketdata import BucketData

from .predict_config import config as cfg

text_recog_model_path = os.environ.get('TEXT_RECOG_MODEL_PATH', 'SET_TEXT_RECOG_MODEL_PATH_ENVIRONMENT_VAR')


class TextLineOCR():
    def __init__(self):
        img_width_range = cfg.img_width_range
        word_len = cfg.word_len
        self.batch_size = cfg.batch_size
        self.visualize = cfg.visualize
        self.img_data = tf.placeholder(tf.float32, shape=(None, 1, 32, None), name='img_data')
        self.zero_paddings = tf.placeholder(tf.float32, shape=(None, None, 512), name='zero_paddings')

        self.bucket_specs = [(int(math.floor(64 / 4)), int(word_len + 2)),
                             (int(math.floor(108 / 4)), int(word_len + 2)),
                             (int(math.floor(140 / 4)), int(word_len + 2)),
                             (int(math.floor(256 / 4)), int(word_len + 2)),
                             (int(math.floor(img_width_range[1] / 4)), int(word_len + 2))]
        buckets = self.buckets = self.bucket_specs

        self.decoder_inputs = []
        self.encoder_masks = []
        self.target_weights = []
        for i in range(int(buckets[-1][0] + 1)):
            self.encoder_masks.append(tf.placeholder(tf.float32, shape=[None, 1],
                                                     name="encoder_mask{0}".format(i)))
        for i in range(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                      name="weight{0}".format(i)))
        self.bucket_min_width, self.bucket_max_width = img_width_range
        self.image_height = cfg.img_height
        self.valid_target_len = cfg.valid_target_len
        self.forward_only = True

        self.bucket_data = {i: BucketData()
                            for i in range(self.bucket_max_width + 1)}

        gpu_device_id = '/cpu:0'
        with tf.device(gpu_device_id):
            cnn_model = CNN(self.img_data, True)  # (not self.forward_only))
            self.conv_output = cnn_model.tf_output()
            self.concat_conv_output = tf.concat(axis=1, values=[self.conv_output, self.zero_paddings])

            self.perm_conv_output = tf.transpose(self.concat_conv_output, perm=[1, 0, 2])

        with tf.device(gpu_device_id):
            self.attention_decoder_model = Seq2SeqModel(
                encoder_masks=self.encoder_masks,
                encoder_inputs_tensor=self.perm_conv_output,
                decoder_inputs=self.decoder_inputs,
                target_weights=self.target_weights,
                target_vocab_size=cfg.target_vocab_size,
                buckets=self.buckets,
                target_embedding_size=cfg.target_embedding_size,
                attn_num_layers=cfg.attn_num_layers,
                attn_num_hidden=cfg.attn_num_hidden,
                forward_only=self.forward_only,
                use_gru=cfg.use_gru)

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, device_count={'GPU': 0}))
        self.saver_all = tf.train.Saver(tf.global_variables())

        ckpt = tf.train.get_checkpoint_state(text_recog_model_path)
        self.saver_all.restore(self.sess, ckpt.model_checkpoint_path)

    # step, read one batch, generate gradients
    def step(self, encoder_masks, img_data, zero_paddings, decoder_inputs, target_weights,
             bucket_id, forward_only):
        # Check if the sizes match.
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                             " %d != %d." % (len(target_weights), decoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        input_feed[self.img_data.name] = img_data
        input_feed[self.zero_paddings.name] = zero_paddings
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]
        for l in range(int(encoder_size)):
            try:
                input_feed[self.encoder_masks[l].name] = encoder_masks[l]
            except Exception as e:
                pass

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([len(img_data)], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                           # self.gradient_norms[bucket_id],  # Gradient norm.
                           self.attention_decoder_model.losses[bucket_id],
                           self.summaries_by_bucket[bucket_id]]
            for l in range(decoder_size):  # Output logits.
                output_feed.append(self.attention_decoder_model.outputs[bucket_id][l])
        else:
            output_feed = [self.attention_decoder_model.losses[bucket_id]]  # Loss for this batch.
            for l in range(decoder_size):  # Output logits.
                output_feed.append(self.attention_decoder_model.outputs[bucket_id][l])
            if self.visualize:
                output_feed += self.attention_decoder_model.attention_weights_histories[bucket_id]

        outputs = self.sess.run(output_feed, input_feed)
        if not forward_only:
            return outputs[2], outputs[1], outputs[3:(
                3 + self.buckets[bucket_id][1])], None  # Gradient norm summary, loss, no outputs, no attentions.
        else:
            return None, outputs[0], outputs[1:(1 + self.buckets[bucket_id][1])], outputs[(
                1 + self.buckets[bucket_id][1]):]  # No gradient norm, loss, outputs, attentions.

    def rec(self, img_buckets):
        def gen_data(img_buffer):
            img_num = len(img_buffer)
            img_idx = 0
            for img in img_buffer:
                img = Image.fromarray(img)
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
                b_idx = self.bucket_max_width
                bs = self.bucket_data[b_idx].append(img_bw, [1, 3, 2], None)
                img_idx += 1
                if bs >= int(cfg.batch_size) or img_idx >= img_num:
                    b = self.bucket_data[b_idx].flush_out(
                        self.bucket_specs,
                        valid_target_length=cfg.valid_target_len,
                        go_shift=1)
                    if b is not None:
                        yield b

        ocr_out = []
        for batch in gen_data(img_buckets):
            bucket_id = batch['bucket_id']
            img_data = batch['data']
            zero_paddings = batch['zero_paddings']
            decoder_inputs = batch['decoder_inputs']
            target_weights = batch['target_weights']
            encoder_masks = batch['encoder_mask']

            grounds = [a for a in np.array([decoder_input.tolist() for decoder_input in decoder_inputs]).transpose()]
            _, step_loss, step_logits, step_attns = self.step(encoder_masks, img_data, zero_paddings, decoder_inputs,
                                                              target_weights, bucket_id, self.forward_only)

            step_outputs = [b for b in
                            np.array([np.argmax(logit, axis=1).tolist() for logit in step_logits]).transpose()]
            for idx, output in zip(range(len(grounds)), step_outputs):
                flag_out = True
                output_valid = []
                for j in range(len(output)):
                    s = output[j]
                    if s != 2 and flag_out:
                        output_valid.append(alphabet2lex_map[s.astype(np.unicode_)])
                    else:
                        flag_out = False
                ocr_out.append(output_valid)

        return ocr_out
