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
        word_size = config.max_word_size

        self.passage = tf.placeholder('int32', [batch_size, None, None], name='passage')
        self.passage_characters = tf.placeholder('int32', [batch_size, None, None, word_size], name='passage_characters')
        self.passage_mask = tf.placeholder('bool', [batch_size, None, None], name='passage_mask')

        self.question = tf.placeholder('int32', [batch_size, None], name='question')
        self.question_characters = tf.placeholder('int32', [batch_size, None, word_size], name='question_characters')
        self.question_mask = tf.placeholder('bool', [batch_size, None], name='question_mask')

        # These options shapes should be the same as the passage.
        self.options = tf.placeholder('int32', [batch_size, None, None], name='options')
        self.options_characters = tf.placeholder('int32', [batch_size, None, None, word_size], name='options_characters')
        self.options_mask = tf.placeholder('bool', [batch_size, None, None], name='options_mask')

        self.y = tf.placeholder('bool', [batch_size, None], name='y')
        self.is_train = tf.placeholder('bool', [], name='is_train')
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
        num_sentences = config.max_num_sents
        sentence_size = config.max_sent_size
        question_size = config.max_ques_size

        # TODO: add these options to the config
        num_options = config.max_num_options
        option_size = config.max_option_size

        word_vocab_size = config.word_vocab_size
        char_vocab_size = config.char_vocab_size
        hidden_size = config.hidden_size
        word_size = config.max_word_size

        question_size = tf.shape(self.question)[1]
        num_sentences = tf.shape(self.passage)[1]
        sentence_size = tf.shape(self.passage)[2]
        num_options = tf.shape(self.options)[1]
        option_size = tf.shape(self.options)[2]

        char_emb_size = config.char_emb_size
        word_emb_size = config.word_emb_size
        char_conv_out_size = config.char_out_size

        with tf.variable_scope("emb"):
            if config.use_char_emb:
                with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
                    char_emb_mat = tf.get_variable("char_emb_mat", shape=[char_vocab_size, char_emb_size], dtype='float')

                with tf.variable_scope("char"):
                    # [batch_size, num_sentences, sentence_size, word_size, char_emb_size]
                    passage_char_embeddings = tf.nn.embedding_lookup(char_emb_mat, self.passage_characters)
                    # [batch_size, question_size, word_size, char_emb_size]
                    question_char_embeddings = tf.nn.embedding_lookup(char_emb_mat, self.question_characters)
                    # [batch_size, num_options, option_size, word_size, char_emb_size]
                    options_char_embeddings = tf.nn.embedding_lookup(char_emb_mat, self.options_characters)

                    passage_char_embeddings = tf.reshape(passage_char_embeddings, [-1, sentence_size, word_size, char_emb_size])
                    question_char_embeddings = tf.reshape(question_char_embeddings, [-1, question_size, word_size, char_emb_size])
                    options_char_embeddings = tf.reshape(options_char_embeddings, [-1, option_size, word_size, char_emb_size])

                    filter_sizes = list(map(int, config.out_channel_dims.split(',')))
                    heights = list(map(int, config.filter_heights.split(',')))
                    assert sum(filter_sizes) == char_conv_out_size, (filter_sizes, char_conv_out_size)
                    with tf.variable_scope("conv"):
                        char_level_embedded_passage = multi_conv1d(passage_char_embeddings, filter_sizes, heights, "VALID",
                                                                   self.is_train, config.keep_prob, scope="char_level_embedded_passage")
                        if config.share_cnn_weights:
                            tf.get_variable_scope().reuse_variables()
                            char_level_embedded_question = multi_conv1d(question_char_embeddings, filter_sizes, heights, "VALID",
                                                                        self.is_train, config.keep_prob, scope="char_level_embedded_passage")
                            # TODO: Not sure if this reuse variables is necessary...
                            tf.get_variable_scope().reuse_variables()
                            char_level_embedded_options = multi_conv1d(options_char_embeddings, filter_sizes, heights, "VALID",
                                                                       self.is_train, config.keep_prob, scope="char_level_embedded_passage")
                        else:
                            char_level_embedded_question = multi_conv1d(question_char_embeddings, filter_sizes, heights, "VALID",
                                                                        self.is_train, config.keep_prob, scope="char_level_embedded_question")
                            char_level_embedded_options = multi_conv1d(options_char_embeddings, filter_sizes, heights, "VALID",
                                                                       self.is_train, config.keep_prob, scope="char_level_embedded_options")

                        char_level_embedded_passage = tf.reshape(char_level_embedded_passage,
                                                                 [-1, num_sentences, sentence_size, char_conv_out_size])
                        char_level_embedded_question = tf.reshape(char_level_embedded_question,
                                                                  [-1, question_size, char_conv_out_size])
                        char_level_embedded_options = tf.reshape(char_level_embedded_options,
                                                                 [-1, num_options, option_size, char_conv_out_size])

            if config.use_word_emb:
                with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
                    if config.mode == 'train':
                        word_emb_mat = tf.get_variable("word_emb_mat", dtype='float', shape=[word_vocab_size, word_emb_size],
                                                       initializer=get_initializer(config.emb_mat))
                    else:
                        word_emb_mat = tf.get_variable("word_emb_mat", shape=[word_vocab_size, word_emb_size], dtype='float')
                    if config.use_glove_for_unk:
                        word_emb_mat = tf.concat(0, [word_emb_mat, self.new_emb_mat])

                with tf.name_scope("word"):
                    # [batch_size, num_sentences, sentence_size, hidden_size]
                    word_level_embedded_passage = tf.nn.embedding_lookup(word_emb_mat, self.passage)
                    # [batch_size, question_size, hidden_size]
                    word_level_embedded_question = tf.nn.embedding_lookup(word_emb_mat, self.question)
                    # [batch_size, num_options, option_size, hidden_size]
                    word_level_embedded_options = tf.nn.embedding_lookup(word_emb_mat, self.options)

                    self.tensor_dict['word_level_embedded_passage'] = word_level_embedded_passage
                    self.tensor_dict['word_level_embedded_question'] = word_level_embedded_question
                    self.tensor_dict['word_level_embedded_options'] = word_level_embedded_options

                if config.use_char_emb:
                    # TODO: NOT SURE WHAT "di" indicates in the comment below.
                    # Maybe just a placeholder for embedding dimensionality
                    # [batch_size, num_sentences, sentence_size, di] (verified)
                    embedded_passage = tf.concat(3, [char_level_embedded_passage, word_level_embedded_passage])
                    # [batch_size, question_size, di] (verified)
                    embedded_question = tf.concat(2, [char_level_embedded_question, word_level_embedded_question])
                    # [batch_size, num_options, option_size, di] (verified)
                    embedded_options = tf.concat(3, [char_level_embedded_options, word_level_embedded_options])
                else:
                    embedded_passage = word_level_embedded_passage
                    embedded_question = word_level_embedded_question
                    embedded_options = word_level_embedded_options

        # highway network
        if config.highway:
            with tf.variable_scope("highway"):
                embedded_passage = highway_network(embedded_passage, config.highway_num_layers,
                                                   True, wd=config.wd, is_train=self.is_train)
                tf.get_variable_scope().reuse_variables()
                embedded_question = highway_network(embedded_question, config.highway_num_layers,
                                                    True, wd=config.wd, is_train=self.is_train)
                tf.get_variable_scope().reuse_variables()
                embedded_options = highway_network(embedded_options, config.highway_num_layers,
                                                   True, wd=config.wd, is_train=self.is_train)

        self.tensor_dict['embedded_passage'] = embedded_passage
        self.tensor_dict['embedded_question'] = embedded_question
        self.tensor_dict['embedded_options'] = embedded_options

        cell = BasicLSTMCell(hidden_size, state_is_tuple=True)
        cell_with_dropout = SwitchableDropoutWrapper(cell, self.is_train, input_keep_prob=config.input_keep_prob)
        # [batch_size, num_sentences]
        passage_len = tf.reduce_sum(tf.cast(self.passage_mask, 'int32'), 2)
        # [batch_size]
        question_len = tf.reduce_sum(tf.cast(self.question_mask, 'int32'), 1)
        # [batch_size, num_options]
        options_len = tf.reduce_sum(tf.cast(self.options_mask, 'int32'), 2)

        with tf.variable_scope("prepro"):
            # [batch_size, J, hidden_size], [batch_size, hidden_size]
            ((fw_outputs, bw_outputs),
             _) = bidirectional_dynamic_rnn(cell_with_dropout, cell_with_dropout, embedded_question,
                                            question_len, dtype='float', scope='encoded_question')
            encoded_question = tf.concat(2, [fw_outputs, bw_outputs])
            if config.share_lstm_weights:
                tf.get_variable_scope().reuse_variables()
                # [batch_size, num_sentences, sentence_size, 2*hidden_size] (verified)
                (fw_outputs, bw_outputs), _ = bidirectional_dynamic_rnn(cell, cell, embedded_passage,
                                                                        passage_len, dtype='float', scope='encoded_question')
                # [batch_size, num_sentences, sentence_size, 2*hidden_size] (verified)
                encoded_passage = tf.concat(3, [fw_outputs, bw_outputs])

                tf.get_variable_scope().reuse_variables()
                # [batch_size, num_options, option_size, 2*hidden_size] (verified)
                (fw_outputs, bw_outputs), _ = bidirectional_dynamic_rnn(cell, cell, embedded_options,
                                                                        options_len, dtype='float', scope='encoded_question')
                # [batch_size, num_options, option_size, 2*hidden_size] (verified)
                encoded_options = tf.concat(3, [fw_outputs, bw_outputs])
            else:
                # [batch_size, num_sentences, sentence_size, 2*hidden_size]
                (fw_outputs, bw_outputs), _ = bidirectional_dynamic_rnn(cell, cell, embedded_passage,
                                                                        passage_len, dtype='float', scope='encoded_passage')
                # [batch_size, num_sentences, sentence_size, 2*hidden_size]
                encoded_passage = tf.concat(3, [fw_outputs, bw_outputs])

                # [batch_size, num_options, option_size, 2*hidden_size]
                (fw_outputs, bw_outputs), _ = bidirectional_dynamic_rnn(cell, cell, embedded_options,
                                                                        options_len, dtype='float', scope='encoded_options')
                # [batch_size, num_options, option_size, 2*hidden_size]
                encoded_options = tf.concat(3, [fw_outputs, bw_outputs])

            self.tensor_dict['encoded_question'] = encoded_question
            self.tensor_dict['encoded_passage'] = encoded_passage
            self.tensor_dict['encoded_options'] = encoded_options

        with tf.variable_scope("main"):
            # The Attention Flow layer
            if config.dynamic_att:
                final_merged_passage = encoded_passage
                encoded_question = tf.reshape(tf.tile(tf.expand_dims(encoded_question, 1), [1, num_sentences, 1, 1]),
                                              [batch_size * num_sentences, question_size, 2 * hidden_size])
                question_mask = tf.reshape(tf.tile(tf.expand_dims(self.question_mask, 1), [1, num_sentences, 1]),
                                           [batch_size * num_sentences, question_size])
                first_cell = AttentionCell(cell, encoded_question, mask=question_mask, mapper='sim',
                                           input_keep_prob=self.config.input_keep_prob, is_train=self.is_train)
            else:
                final_merged_passage = attention_layer(config, self.is_train, encoded_passage, encoded_question,
                                                       h_mask=self.passage_mask, u_mask=self.question_mask,
                                                       scope="final_merged_passage", tensor_dict=self.tensor_dict)
                first_cell = cell_with_dropout

            # The modeling layer
            # The 0th (first) bidirectional encoder in the modeling layer
            # [batch_size, num_sentences, sentence_size, 2*hidden_size]
            (fw_output_biencoder0, bw_output_biencoder0), _ = bidirectional_dynamic_rnn(first_cell, first_cell, final_merged_passage,
                                                                                        passage_len, dtype='float', scope='model_layer_biencoder_0')
            modeled_passage_0 = tf.concat(3, [fw_output_biencoder0, bw_output_biencoder0])

            # The 1st (second) bidirectional encoder in the modeling layer
            # [batch_size, num_sentences, sentence_size, 2*hidden_size]
            (fw_output_biencoder1, bw_output_biencoder1), _ = bidirectional_dynamic_rnn(first_cell, first_cell, modeled_passage_0,
                                                                                        passage_len, dtype='float', scope='model_layer_biencoder_1')
            # This encoding is used to get the start span index.
            modeled_passage_1 = tf.concat(3, [fw_output_biencoder1, bw_output_biencoder1])

            # The logits for the start span answer.
            span_begin_logits = get_logits([modeled_passage_1, final_merged_passage], hidden_size, True,
                                           wd=config.wd, input_keep_prob=config.input_keep_prob,
                                           mask=self.passage_mask, is_train=self.is_train, func=config.answer_func, scope='logits1')

            passage_weighted_by_predicted_span = softsel(tf.reshape(modeled_passage_1, [batch_size, num_sentences * sentence_size, 2 * hidden_size]),
                                                         tf.reshape(span_begin_logits, [batch_size, num_sentences * sentence_size]))
            passage_weighted_by_predicted_span = tf.tile(tf.expand_dims(tf.expand_dims(passage_weighted_by_predicted_span, 1), 1),
                                                         [1, num_sentences, sentence_size, 1])

            # [batch_size, num_sentences, sentence_size, 2*hidden_size]
            ((fw_end_encoder_output,
              bw_end_encoder_output), _) = bidirectional_dynamic_rnn(cell_with_dropout, cell_with_dropout,
                                                                     tf.concat(3, [final_merged_passage, modeled_passage_1,
                                                                                   passage_weighted_by_predicted_span,
                                                                                   modeled_passage_1 * passage_weighted_by_predicted_span]),
                                                                     passage_len, dtype='float', scope='end_span_encoder')
            end_span_modeled_passage = tf.concat(3, [fw_end_encoder_output, bw_end_encoder_output])
            span_end_logits = get_logits([end_span_modeled_passage, final_merged_passage], hidden_size, True,
                                         wd=config.wd, input_keep_prob=config.input_keep_prob, mask=self.passage_mask,
                                         is_train=self.is_train, func=config.answer_func, scope='span_end_logits')

            flat_span_begin_logits = tf.reshape(span_begin_logits, [-1, num_sentences * sentence_size])
            # [-1, num_sentences*sentence_size]
            flat_span_begin = tf.nn.softmax(flat_span_begin_logits)

            flat_span_end_logits = tf.reshape(span_end_logits, [-1, num_sentences * sentence_size])
            flat_span_end = tf.nn.softmax(flat_span_end_logits)

            self.tensor_dict['modeled_passage_1'] = modeled_passage_1
            self.tensor_dict['end_span_modeled_passage'] = end_span_modeled_passage

            # Now, we take flat_span_begin and flat_span_end (or shape [-1, num_sentences*sentence_size])
            # and we compute a span envelope over the passage.
            after_span_begin = tf.cumsum(flat_span_begin, axis=-1)
            after_span_end = tf.cumsum(flat_span_end, axis=-1)
            before_span_end = 1.0 - after_span_end
            # shape: [-1, num_sentences*sentence_size]
            envelope = after_span_begin * before_span_end

            # Now we multiply the `encoded_passage` above by the envelope, reshaping the encoded_passage
            # to [-1, num_sentences*sentence_size, hidden_dim].
            flattened_encoded_passage = tf.reshape(encoded_passage,
                                                  [-1, num_sentences * sentence_size,
                                                   hidden_size * 2])
            # shape: [-1, num_sentences*sentence_size]
            flattened_passage_mask = tf.reshape(self.passage_mask,
                                                [-1, num_sentences * sentence_size])
            # shape: [-1, num_sentences*sentence_size, 2xhidden_dim]
            weighted_passage = tf.tile(tf.expand_dims(envelope, 2), [1, 1, hidden_size*2]) * flattened_encoded_passage
            # Now, we encode the weighted passage with a bag-of-words
            # shape: [-1, num_sentences*sentence_size]
            float_passage_mask = tf.cast(flattened_passage_mask, "float32")
            # expand the denominator to be same shape as numerator, then add epsilon to avoid
            # division by 0.
            # shape: [-1, num_sentences*sentence_size]
            weighted_passage_mask = float_passage_mask / (tf.reduce_sum(float_passage_mask,
                                                                        reduction_indices=1,
                                                                        keep_dims=True) + 10e-8)

            # expand mask to match dimensionality of weighted_passage
            # shape: [-1, num_sentences*sentence_size, 1]
            weighted_passage_mask = tf.expand_dims(weighted_passage_mask, -1)
            # shape: [-1, 2xhidden_dim]
            final_encoded_weighted_passage = tf.reduce_sum(weighted_passage * weighted_passage_mask,
                                                           reduction_indices=1)

            # We also encode the encoded options one more time with a bag-of-words
            # [-1, num_options, option_size]
            float_options_mask = tf.cast(self.options_mask, "float32")

            # [-1, num_options, option_size]
            weighted_options_mask = float_options_mask / (tf.reduce_sum(float_options_mask,
                                                                        reduction_indices=2,
                                                                        keep_dims=True) + 10e-8)
            # expand mask to match dimensionality of encoded_options
            # shape: [-1, num_options, option_size, 1]
            weighted_options_mask = tf.expand_dims(weighted_options_mask, -1)

            # shape: [-1, num_options, 2xhidden_dim]
            final_encoded_options = tf.reduce_sum(encoded_options * weighted_options_mask,
                                                  reduction_indices=2)

            # Now, we want to take the dot product of final_encoded_options and
            # final_encoded_weighted_passage to get a distribution over our options.

            # tile final_encoded_weighted_passage to be the same shape as the options
            # shape: [-1, num_options, 2xhidden_dim]
            tiled_final_encoded_weighted_passage = tf.tile(
                tf.expand_dims(final_encoded_weighted_passage, 1),
                [1, num_options, 1])

            # shape: [-1, num_options]
            answer_option_similarities = tf.reduce_sum(
                tiled_final_encoded_weighted_passage * final_encoded_options,
                reduction_indices=2)

            # the get_logits function takes the dot product between the final_encoded_options
            # and the tiled_final_encoded_weighted_passage. The size and bias
            # (second and third positional arguments) are unused, but the API requires them.
            # shape: [-1, num_options]
            answer_option_logits = get_logits([tiled_final_encoded_weighted_passage, final_encoded_options],
                                              hidden_size, True,
                                              mask=tf.cast(options_len, "bool"),
                                              func="dot", scope="option_logits")

            self.logits = answer_option_logits
            self.yp = tf.nn.softmax(answer_option_logits)

    def _build_loss(self):
        num_options = tf.shape(self.options)[1]
        loss_mask = tf.reduce_max(tf.cast(self.options_mask, 'float'), 1)
        losses = tf.nn.softmax_cross_entropy_with_logits(
            self.logits, tf.cast(tf.reshape(self.y, [-1, num_options]), 'float'))
        ce_loss = tf.reduce_mean(loss_mask * losses)
        tf.add_to_collection('losses', ce_loss)

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

        batch_size = config.batch_size
        num_sentences = config.max_num_sents
        sentence_size = config.max_sent_size
        num_options = config.max_num_options
        option_size = config.max_option_size
        question_size = config.max_ques_size
        word_size = config.max_word_size

        feed_dict = {}

        if config.len_opt:
            """
            Note that this optimization results in variable GPU RAM usage (i.e. can cause OOM in the middle of training.)
            First test without len_opt and make sure no OOM, and use len_opt
            """
            if sum(len(sent) for para in batch.data['x'] for sent in para) == 0:
                new_sentence_size = 1
            else:
                new_sentence_size = max(len(sent) for para in batch.data['x'] for sent in para)
            sentence_size = min(sentence_size, new_sentence_size)

            if sum(len(ques) for ques in batch.data['q']) == 0:
                new_question_size = 1
            else:
                new_question_size = max(len(ques) for ques in batch.data['q'])
            question_size = min(question_size, new_question_size)

        if config.cpu_opt:
            if sum(len(para) for para in batch.data['x']) == 0:
                new_num_sentences = 1
            else:
                new_num_sentences = max(len(para) for para in batch.data['x'])
            num_sentences = min(num_sentences, new_num_sentences)

        passage = np.zeros([batch_size, num_sentences, sentence_size], dtype='int32')
        passage_characters = np.zeros([batch_size, num_sentences, sentence_size, word_size], dtype='int32')
        passage_mask = np.zeros([batch_size, num_sentences, sentence_size], dtype='bool')
        question = np.zeros([batch_size, question_size], dtype='int32')
        question_characters = np.zeros([batch_size, question_size, word_size], dtype='int32')
        question_mask = np.zeros([batch_size, question_size], dtype='bool')
        options = np.zeros([batch_size, num_options, option_size], dtype='int32')
        options_characters = np.zeros([batch_size, num_options, option_size, word_size], dtype='int32')
        options_mask = np.zeros([batch_size, num_options, option_size], dtype='bool')

        feed_dict[self.passage] = passage
        feed_dict[self.passage_mask] = passage_mask
        feed_dict[self.passage_characters] = passage_characters

        feed_dict[self.question] = question
        feed_dict[self.question_characters] = question_characters
        feed_dict[self.question_mask] = question_mask

        feed_dict[self.options] = options
        feed_dict[self.options_characters] = options_characters
        feed_dict[self.options_mask] = options_mask

        feed_dict[self.is_train] = is_train
        if config.use_glove_for_unk:
            feed_dict[self.new_emb_mat] = batch.shared['new_emb_mat']

        X = batch.data['tokenized_passages']
        CX = batch.data['tokenized_passage_characters']

        # TODO: figure out what this does / if it's actually necessary in testing
        if supervised:
            y = np.zeros([batch_size, num_options], dtype='bool')
            feed_dict[self.y] = y
            for idx, label in enumerate(batch.data["labels"]):
                y[idx, label] = True

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

        # Indexes the passage words list and marks the mask values
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

        # Indexes the passage characters list.
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

        # Indexes the options words list and marks the mask values
        for i, xi in enumerate(batch.data['tokenized_options']):
            if self.config.squash:
                xi = [list(itertools.chain(*xi))]
            for j, xij in enumerate(xi):
                if j == num_options:
                    break
                for k, xijk in enumerate(xij):
                    if k == option_size:
                        break
                    each = _get_word(xijk)
                    assert isinstance(each, int), each
                    options[i, j, k] = each
                    options_mask[i, j, k] = True

        # Indexes the options characters list.
        for i, cxi in enumerate(batch.data['tokenized_option_characters']):
            if self.config.squash:
                cxi = [list(itertools.chain(*cxi))]
            for j, cxij in enumerate(cxi):
                if j == num_options:
                    break
                for k, cxijk in enumerate(cxij):
                    if k == option_size:
                        break
                    for l, cxijkl in enumerate(cxijk):
                        if l == config.max_word_size:
                            break
                        options_characters[i, j, k, l] = _get_char(cxijkl)

        # Indexes the question words list and marks the mask values
        for i, qi in enumerate(batch.data['tokenized_questions']):
            for j, qij in enumerate(qi):
                question[i, j] = _get_word(qij)
                question_mask[i, j] = True

        # Indexes the question characters list
        for i, cqi in enumerate(batch.data['tokenized_question_characters']):
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
