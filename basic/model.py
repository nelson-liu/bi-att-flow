import random

import itertools
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import BasicLSTMCell

from basic.read_data import DataSet
from my.tensorflow import get_initializer
from my.tensorflow.nn import softsel, get_logits, highway_network, multi_conv1d
from my.tensorflow.rnn import bidirectional_dynamic_rnn
from my.tensorflow.rnn_cell import SwitchableDropoutWrapper, AttentionCell


def get_multi_gpu_models(config):
    models = []
    for gpu_idx in range(config.num_gpus):
        with tf.name_scope("model_{}".format(gpu_idx)) as scope, tf.device("/{}:{}".format(config.device_type, gpu_idx)):
            model = Model(config, scope, rep=gpu_idx == 0)
            tf.get_variable_scope().reuse_variables()
            models.append(model)
    return models


class Model(object):
    def __init__(self, config, scope, rep=True):
        self.scope = scope
        self.config = config
        self.global_step = tf.get_variable('global_step', shape=[], dtype='int32',
                                           initializer=tf.constant_initializer(0), trainable=False)

        ##############################
        # Define forward inputs here #
        ##############################

        batch_size = config.batch_size
        max_word_size = config.max_word_size

        # self.x = tf.placeholder('int32', [batch_size, None, None], name='x')
        self.passage = tf.placeholder('int32', [batch_size, None, None], name='passage')

        # self.passage_characters = tf.placeholder('int32', [batch_size, None, None, max_word_size], name='cx')
        self.passage_characters = tf.placeholder('int32', [batch_size, None, None, max_word_size], name='passage_characters')

        # self.passage_mask = tf.placeholder('bool', [batch_size, None, None], name='x_mask')
        self.passage_mask = tf.placeholder('bool', [batch_size, None, None], name='passage_mask')

        # self.q = tf.placeholder('int32', [batch_size, None], name='q')
        self.question = tf.placeholder('int32', [batch_size, None], name='question')

        # self.cq = tf.placeholder('int32', [batch_size, None, max_word_size], name='cq')
        self.question_characters = tf.placeholder('int32', [batch_size, None, max_word_size], name='question_characters')

        # self.q_mask = tf.placeholder('bool', [batch_size, None], name='q_mask')
        self.question_mask = tf.placeholder('bool', [batch_size, None], name='question_mask')

        # self.y = tf.placeholder('bool', [batch_size, None, None], name='y')
        self.y = tf.placeholder('bool', [batch_size, None, None], name='y')

        # self.y2 = tf.placeholder('bool', [batch_size, None, None], name='y2')
        self.y2 = tf.placeholder('bool', [batch_size, None, None], name='y2')

        # self.is_train = tf.placeholder('bool', [], name='is_train')
        self.is_train = tf.placeholder('bool', [], name='is_train')

        # self.new_emb_mat = tf.placeholder('float', [None, config.word_emb_size], name='new_emb_mat')
        self.new_emb_mat = tf.placeholder('float', [None, config.word_emb_size], name='new_emb_mat')

        ###############
        # Define misc #
        ###############
        self.tensor_dict = {}

        #################################
        # Forward outputs / loss inputs #
        #################################
        self.logits = None
        self.yp = None
        self.var_list = None

        ################
        # Loss outputs #
        ################
        self.loss = None

        self._build_forward()
        self._build_loss()
        self.var_ema = None
        if rep:
            self._build_var_ema()
        if config.mode == 'train':
            self._build_ema()

        self.summary = tf.merge_all_summaries()
        self.summary = tf.merge_summary(tf.get_collection("summaries", scope=self.scope))

    def _build_forward(self):
        config = self.config
        batch_size = config.batch_size
        max_num_sentences = config.max_num_sents
        max_sentence_size = config.max_sent_size
        max_question_size = config.max_ques_size
        word_vocab_size = config.word_vocab_size
        char_vocab_size = config.char_vocab_size
        hidden_size = config.hidden_size
        max_word_size = config.max_word_size

        max_sentence_size = tf.shape(self.passage)[2]
        max_question_size = tf.shape(self.question)[1]
        max_num_sentences = tf.shape(self.passage)[1]

        char_emb_size = config.char_emb_size
        word_emb_size = config.word_emb_size
        char_conv_out_size = config.char_out_size

        with tf.variable_scope("emb"):
            if config.use_char_emb:
                with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
                    char_emb_mat = tf.get_variable("char_emb_mat", shape=[char_vocab_size, char_emb_size], dtype='float')

                with tf.variable_scope("char"):
                    # [batch_size, max_num_sentences, max_sentence_size, max_word_size, char_emb_size]
                    passage_char_embeddings = tf.nn.embedding_lookup(char_emb_mat, self.passage_characters)
                    # [batch_size, max_question_size, max_word_size, char_emb_size]
                    question_char_embeddings = tf.nn.embedding_lookup(char_emb_mat, self.question_characters)
                    passage_char_embeddings = tf.reshape(passage_char_embeddings, [-1, max_sentence_size, max_word_size, char_emb_size])
                    question_char_embeddings = tf.reshape(question_char_embeddings, [-1, max_question_size, max_word_size, char_emb_size])

                    filter_sizes = list(map(int, config.out_channel_dims.split(',')))
                    heights = list(map(int, config.filter_heights.split(',')))
                    assert sum(filter_sizes) == char_conv_out_size, (filter_sizes, char_conv_out_size)
                    with tf.variable_scope("conv"):
                        char_level_embedded_passage = multi_conv1d(passage_char_embeddings, filter_sizes, heights, "VALID",  self.is_train, config.keep_prob, scope="char_level_embedded_passage")
                        if config.share_cnn_weights:
                            tf.get_variable_scope().reuse_variables()
                            char_level_embedded_question = multi_conv1d(question_char_embeddings, filter_sizes, heights, "VALID", self.is_train, config.keep_prob, scope="char_level_embedded_passage")
                        else:
                            char_level_embedded_question = multi_conv1d(question_char_embeddings, filter_sizes, heights, "VALID", self.is_train, config.keep_prob, scope="char_level_embedded_question")
                        char_level_embedded_passage = tf.reshape(char_level_embedded_passage, [-1, max_num_sentences, max_sentence_size, char_conv_out_size])
                        char_level_embedded_question = tf.reshape(char_level_embedded_question, [-1, max_question_size, char_conv_out_size])

            if config.use_word_emb:
                with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
                    if config.mode == 'train':
                        word_emb_mat = tf.get_variable("word_emb_mat", dtype='float', shape=[word_vocab_size, word_emb_size], initializer=get_initializer(config.emb_mat))
                    else:
                        word_emb_mat = tf.get_variable("word_emb_mat", shape=[word_vocab_size, ], dtype='float')
                    if config.use_glove_for_unk:
                        word_emb_mat = tf.concat(0, [word_emb_mat, self.new_emb_mat])

                with tf.name_scope("word"):
                    # [batch_size, max_num_sentences, max_sentence_size, hidden_size]
                    word_level_embedded_passage = tf.nn.embedding_lookup(word_emb_mat, self.passage)
                    # [batch_size, max_question_size, hidden_size]
                    word_level_embedded_question = tf.nn.embedding_lookup(word_emb_mat, self.question)
                    self.tensor_dict['word_level_embedded_passage'] = word_level_embedded_passage
                    self.tensor_dict['word_level_embedded_question'] = word_level_embedded_question

                if config.use_char_emb:
                    # TODO: NOT SURE WHAT "di" indicates in the comment below.
                    # [batch_size, max_num_sentences, max_sentence_size, di]
                    embedded_passage = tf.concat(3, [char_level_embedded_passage, word_level_embedded_passage])
                    # [batch_size, max_question_size, di]
                    embedded_question = tf.concat(2, [char_level_embedded_question, word_level_embedded_question])
                else:
                    embedded_passage = word_level_embedded_passage
                    embedded_question = word_level_embedded_question

        # highway network
        if config.highway:
            with tf.variable_scope("highway"):
                embedded_passage = highway_network(embedded_passage, config.highway_num_layers, True, wd=config.wd, is_train=self.is_train)
                tf.get_variable_scope().reuse_variables()
                embedded_question = highway_network(embedded_question, config.highway_num_layers, True, wd=config.wd, is_train=self.is_train)

        self.tensor_dict['embedded_passage'] = embedded_passage
        self.tensor_dict['embedded_question'] = embedded_question

        cell = BasicLSTMCell(hidden_size, state_is_tuple=True)
        cell_with_dropout = SwitchableDropoutWrapper(cell, self.is_train, input_keep_prob=config.input_keep_prob)
        # [batch_size, max_num_sentences]
        passage_len = tf.reduce_sum(tf.cast(self.passage_mask, 'int32'), 2)
        # [batch_size]
        question_len = tf.reduce_sum(tf.cast(self.question_mask, 'int32'), 1)

        with tf.variable_scope("prepro"):
            # [batch_size, J, hidden_size], [batch_size, hidden_size]
            (fw_outputs, bw_outputs), ((_, fw_final_state), (_, bw_final_state)) = bidirectional_dynamic_rnn(cell_with_dropout, cell_with_dropout, embedded_question, question_len, dtype='float', scope='encoded_question')
            encoded_question = tf.concat(2, [fw_outputs, bw_outputs])
            if config.share_lstm_weights:
                tf.get_variable_scope().reuse_variables()
                # [batch_size, max_num_sentences, max_sentence_size, 2*hidden_size]
                (fw_outputs, bw_outputs), _ = bidirectional_dynamic_rnn(cell, cell, embedded_passage, passage_len, dtype='float', scope='encoded_question')
                # [batch_size, max_num_sentences, max_sentence_size, 2*hidden_size]
                encoded_passage = tf.concat(3, [fw_outputs, bw_outputs])
            else:
                # [batch_size, max_num_sentences, max_sentence_size, 2*hidden_size]
                (fw_outputs, bw_outputs), _ = bidirectional_dynamic_rnn(cell, cell, embedded_passage, passage_len, dtype='float', scope='encoded_passage')
                # [batch_size, max_num_sentences, max_sentence_size, 2*hidden_size]
                encoded_passage = tf.concat(3, [fw_outputs, bw_outputs])
            self.tensor_dict['encoded_question'] = encoded_question
            self.tensor_dict['encoded_passage'] = encoded_passage

        with tf.variable_scope("main"):
            # The Attention Flow layer
            # TODO: NOT SURE WHAT TO CALL P0 here / NOT SURE WHAT IT REPRESENTS.
            if config.dynamic_att:
                p0 = encoded_passage
                encoded_question = tf.reshape(tf.tile(tf.expand_dims(encoded_question, 1), [1, max_num_sentences, 1, 1]), [batch_size * max_num_sentences, max_question_size, 2 * hidden_size])
                question_mask = tf.reshape(tf.tile(tf.expand_dims(self.question_mask, 1), [1, max_num_sentences, 1]), [batch_size * max_num_sentences, max_question_size])
                first_cell = AttentionCell(cell, encoded_question, mask=question_mask, mapper='sim',
                                           input_keep_prob=self.config.input_keep_prob, is_train=self.is_train)
            else:
                p0 = attention_layer(config, self.is_train, encoded_passage, encoded_question, h_mask=self.passage_mask, u_mask=self.question_mask, scope="p0", tensor_dict=self.tensor_dict)
                first_cell = cell_with_dropout

            # The modeling layer
            # The 0th (first) bidirectional encoder in the modeling layer
            # [batch_size, max_num_sentences, max_sentence_size, 2*hidden_size]
            (fw_output_biencoder0, bw_output_biencoder0), _ = bidirectional_dynamic_rnn(first_cell, first_cell, p0, passage_len, dtype='float', scope='model_layer_biencoder_0')
            modeled_passage_0 = tf.concat(3, [fw_output_biencoder0, bw_output_biencoder0])

            # The 1st (second) bidirectional encoder in the modeling layer
            # [batch_size, max_num_sentences, max_sentence_size, 2*hidden_size]
            (fw_output_biencoder1, bw_output_biencoder1), _ = bidirectional_dynamic_rnn(first_cell, first_cell, modeled_passage_0, passage_len, dtype='float', scope='model_layer_biencoder_1')
            # This encoding is used to get the start span index.
            modeled_passage_1 = tf.concat(3, [fw_output_biencoder1, bw_output_biencoder1])

            # The logits for the start span answer.
            logits = get_logits([modeled_passage_1, p0], hidden_size, True, wd=config.wd, input_keep_prob=config.input_keep_prob,
                                mask=self.passage_mask, is_train=self.is_train, func=config.answer_func, scope='logits1')

            # TODO: NOT QUITE SURE WHAT'S GOING ON HERE EITHER
            a1i = softsel(tf.reshape(modeled_passage_1, [batch_size, max_num_sentences * max_sentence_size, 2 * hidden_size]), tf.reshape(logits, [batch_size, max_num_sentences * max_sentence_size]))
            a1i = tf.tile(tf.expand_dims(tf.expand_dims(a1i, 1), 1), [1, max_num_sentences, max_sentence_size, 1])

            # [batch_size, max_num_sentences, max_sentence_size, 2*hidden_size]
            ((fw_end_encoder_output,
              bw_end_encoder_output), _) = bidirectional_dynamic_rnn(cell_with_dropout, cell_with_dropout, tf.concat(3, [p0, modeled_passage_1, a1i, modeled_passage_1 * a1i]),
                                                                     passage_len, dtype='float', scope='end_span_encoder')
            end_span_modeled_passage = tf.concat(3, [fw_end_encoder_output, bw_end_encoder_output])
            logits2 = get_logits([end_span_modeled_passage, p0], hidden_size, True, wd=config.wd, input_keep_prob=config.input_keep_prob,
                                 mask=self.passage_mask,
                                 is_train=self.is_train, func=config.answer_func, scope='logits2')

            flat_logits = tf.reshape(logits, [-1, max_num_sentences * max_sentence_size])
            # [-1, max_num_sentences*max_sentence_size]
            flat_yp = tf.nn.softmax(flat_logits)
            yp = tf.reshape(flat_yp, [-1, max_num_sentences, max_sentence_size])

            flat_logits2 = tf.reshape(logits2, [-1, max_num_sentences * max_sentence_size])
            flat_yp2 = tf.nn.softmax(flat_logits2)
            yp2 = tf.reshape(flat_yp2, [-1, max_num_sentences, max_sentence_size])

            self.tensor_dict['modeled_passage_1'] = modeled_passage_1
            self.tensor_dict['end_span_modeled_passage'] = end_span_modeled_passage

            self.logits = flat_logits
            self.logits2 = flat_logits2
            self.yp = yp
            self.yp2 = yp2

    def _build_loss(self):
        max_sentence_size = tf.shape(self.passage)[2]
        max_num_sentences = tf.shape(self.passage)[1]
        loss_mask = tf.reduce_max(tf.cast(self.question_mask, 'float'), 1)
        losses = tf.nn.softmax_cross_entropy_with_logits(
            self.logits, tf.cast(tf.reshape(self.y, [-1, max_num_sentences * max_sentence_size]), 'float'))
        ce_loss = tf.reduce_mean(loss_mask * losses)
        tf.add_to_collection('losses', ce_loss)
        ce_loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            self.logits2, tf.cast(tf.reshape(self.y2, [-1, max_num_sentences * max_sentence_size]), 'float')))
        tf.add_to_collection("losses", ce_loss2)

        self.loss = tf.add_n(tf.get_collection('losses', scope=self.scope), name='loss')
        tf.scalar_summary(self.loss.op.name, self.loss)
        tf.add_to_collection('ema/scalar', self.loss)

    def _build_ema(self):
        self.ema = tf.train.ExponentialMovingAverage(self.config.decay)
        ema = self.ema
        tensors = tf.get_collection("ema/scalar", scope=self.scope) + tf.get_collection("ema/vector", scope=self.scope)
        ema_op = ema.apply(tensors)
        for var in tf.get_collection("ema/scalar", scope=self.scope):
            ema_var = ema.average(var)
            tf.scalar_summary(ema_var.op.name, ema_var)
        for var in tf.get_collection("ema/vector", scope=self.scope):
            ema_var = ema.average(var)
            tf.histogram_summary(ema_var.op.name, ema_var)

        with tf.control_dependencies([ema_op]):
            self.loss = tf.identity(self.loss)

    def _build_var_ema(self):
        self.var_ema = tf.train.ExponentialMovingAverage(self.config.var_decay)
        ema = self.var_ema
        ema_op = ema.apply(tf.trainable_variables())
        with tf.control_dependencies([ema_op]):
            self.loss = tf.identity(self.loss)

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step

    def get_var_list(self):
        return self.var_list

    def get_feed_dict(self, batch, is_train, supervised=True):
        assert isinstance(batch, DataSet)
        config = self.config

        # N = config.batch_size
        # M = config.max_num_sents
        # JX = config.max_sent_size
        # JQ = config.max_ques_size
        # W = config.max_word_size
        batch_size = config.batch_size
        max_num_sentences = config.max_num_sents
        max_sentence_size = config.max_sent_size
        max_question_size = config.max_ques_size
        max_word_size = config.max_word_size

        feed_dict = {}

        if config.len_opt:
            """
            Note that this optimization results in variable GPU RAM usage (i.e. can cause OOM in the middle of training.)
            First test without len_opt and make sure no OOM, and use len_opt
            """
            if sum(len(sent) for para in batch.data['x'] for sent in para) == 0:
                new_max_sentence_size = 1
            else:
                new_max_sentence_size = max(len(sent) for para in batch.data['x'] for sent in para)
            max_sentence_size = min(max_sentence_size, new_max_sentence_size)

            if sum(len(ques) for ques in batch.data['q']) == 0:
                new_max_question_size = 1
            else:
                new_max_question_size = max(len(ques) for ques in batch.data['q'])
            max_question_size = min(max_question_size, new_max_question_size)

        if config.cpu_opt:
            if sum(len(para) for para in batch.data['x']) == 0:
                new_max_num_sentences = 1
            else:
                new_max_num_sentences = max(len(para) for para in batch.data['x'])
            max_num_sentences = min(max_num_sentences, new_max_num_sentences)

        passage = np.zeros([batch_size, max_num_sentences, max_sentence_size], dtype='int32')
        passage_characters = np.zeros([batch_size, max_num_sentences, max_sentence_size, max_word_size], dtype='int32')
        passage_mask = np.zeros([batch_size, max_num_sentences, max_sentence_size], dtype='bool')
        question = np.zeros([batch_size, max_question_size], dtype='int32')
        question_characters = np.zeros([batch_size, max_question_size, max_word_size], dtype='int32')
        question_mask = np.zeros([batch_size, max_question_size], dtype='bool')

        feed_dict[self.passage] = passage
        feed_dict[self.passage_mask] = passage_mask
        feed_dict[self.passage_characters] = passage_characters
        feed_dict[self.question] = question
        feed_dict[self.question_characters] = question_characters
        feed_dict[self.question_mask] = question_mask
        feed_dict[self.is_train] = is_train
        if config.use_glove_for_unk:
            feed_dict[self.new_emb_mat] = batch.shared['new_emb_mat']

        X = batch.data['x']
        CX = batch.data['cx']

        if supervised:
            y = np.zeros([batch_size, max_num_sentences, max_sentence_size], dtype='bool')
            y2 = np.zeros([batch_size, max_num_sentences, max_sentence_size], dtype='bool')
            feed_dict[self.y] = y
            feed_dict[self.y2] = y2

            for i, (xi, cxi, yi) in enumerate(zip(X, CX, batch.data['y'])):
                start_idx, stop_idx = random.choice(yi)
                j, k = start_idx
                j2, k2 = stop_idx
                if config.single:
                    X[i] = [xi[j]]
                    CX[i] = [cxi[j]]
                    j, j2 = 0, 0
                if config.squash:
                    offset = sum(map(len, xi[:j]))
                    j, k = 0, k + offset
                    offset = sum(map(len, xi[:j2]))
                    j2, k2 = 0, k2 + offset
                y[i, j, k] = True
                y2[i, j2, k2-1] = True

        def _get_word(word):
            d = batch.shared['word2idx']
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in d:
                    return d[each]
            if config.use_glove_for_unk:
                d2 = batch.shared['new_word2idx']
                for each in (word, word.lower(), word.capitalize(), word.upper()):
                    if each in d2:
                        return d2[each] + len(d)
            return 1

        def _get_char(char):
            d = batch.shared['char2idx']
            if char in d:
                return d[char]
            return 1

        for i, xi in enumerate(X):
            if self.config.squash:
                xi = [list(itertools.chain(*xi))]
            for j, xij in enumerate(xi):
                if j == config.max_num_sents:
                    break
                for k, xijk in enumerate(xij):
                    if k == config.max_sent_size:
                        break
                    each = _get_word(xijk)
                    assert isinstance(each, int), each
                    passage[i, j, k] = each
                    passage_mask[i, j, k] = True

        for i, cxi in enumerate(CX):
            if self.config.squash:
                cxi = [list(itertools.chain(*cxi))]
            for j, cxij in enumerate(cxi):
                if j == config.max_num_sents:
                    break
                for k, cxijk in enumerate(cxij):
                    if k == config.max_sent_size:
                        break
                    for l, cxijkl in enumerate(cxijk):
                        if l == config.max_word_size:
                            break
                        passage_characters[i, j, k, l] = _get_char(cxijkl)

        for i, qi in enumerate(batch.data['q']):
            for j, qij in enumerate(qi):
                question[i, j] = _get_word(qij)
                question_mask[i, j] = True

        for i, cqi in enumerate(batch.data['cq']):
            for j, cqij in enumerate(cqi):
                for k, cqijk in enumerate(cqij):
                    question_characters[i, j, k] = _get_char(cqijk)
                    if k + 1 == config.max_word_size:
                        break

        return feed_dict


def bi_attention(config, is_train, h, u, h_mask=None, u_mask=None, scope=None, tensor_dict=None):
    with tf.variable_scope(scope or "bi_attention"):
        JX = tf.shape(h)[2]
        M = tf.shape(h)[1]
        JQ = tf.shape(u)[1]
        h_aug = tf.tile(tf.expand_dims(h, 3), [1, 1, 1, JQ, 1])
        u_aug = tf.tile(tf.expand_dims(tf.expand_dims(u, 1), 1), [1, M, JX, 1, 1])
        if h_mask is None:
            hu_mask = None
        else:
            h_mask_aug = tf.tile(tf.expand_dims(h_mask, 3), [1, 1, 1, JQ])
            u_mask_aug = tf.tile(tf.expand_dims(tf.expand_dims(u_mask, 1), 1), [1, M, JX, 1])
            hu_mask = h_mask_aug & u_mask_aug

        u_logits = get_logits([h_aug, u_aug], None, True, wd=config.wd, mask=hu_mask,
                              is_train=is_train, func=config.logit_func, scope='u_logits')  # [N, M, JX, JQ]
        u_a = softsel(u_aug, u_logits)  # [N, M, JX, d]
        h_a = softsel(h, tf.reduce_max(u_logits, 3))  # [N, M, d]
        h_a = tf.tile(tf.expand_dims(h_a, 2), [1, 1, JX, 1])

        if tensor_dict is not None:
            a_u = tf.nn.softmax(u_logits)  # [N, M, JX, JQ]
            a_h = tf.nn.softmax(tf.reduce_max(u_logits, 3))
            tensor_dict['a_u'] = a_u
            tensor_dict['a_h'] = a_h
            variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=tf.get_variable_scope().name)
            for var in variables:
                tensor_dict[var.name] = var

        return u_a, h_a


def attention_layer(config, is_train, h, u, h_mask=None, u_mask=None, scope=None, tensor_dict=None):
    with tf.variable_scope(scope or "attention_layer"):
        JX = tf.shape(h)[2]
        M = tf.shape(h)[1]
        JQ = tf.shape(u)[1]
        if config.q2c_att or config.c2q_att:
            u_a, h_a = bi_attention(config, is_train, h, u, h_mask=h_mask, u_mask=u_mask, tensor_dict=tensor_dict)
        if not config.c2q_att:
            u_a = tf.tile(tf.expand_dims(tf.expand_dims(tf.reduce_mean(u, 1), 1), 1), [1, M, JX, 1])
        if config.q2c_att:
            p0 = tf.concat(3, [h, u_a, h * u_a, h * h_a])
        else:
            p0 = tf.concat(3, [h, u_a, h * u_a])
        return p0
