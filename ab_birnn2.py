import tensorflow as tf


class RNN:

    def __init__(self, sequence_length, num_classes,
                 text_vocab_size, text_embedding_size, pos_vocab_size, pos_embedding_size,cell_type,hidden_size,big_num,
                  l2_reg_lambda=0.0):

        print("sequence lengh=",sequence_length)
        print("num_classe",num_classes)
        print("text_vocab_size=",text_vocab_size)
        print("pos_vocab_size=", pos_vocab_size)
        print("text_embedding_size=", text_embedding_size)
        print("cell_type=", cell_type)
        print("hidden_size=", hidden_size)
        # Placeholders for input, output and dropout
        self.input_text = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_text')
        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')

        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        #self.input_text = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_text')
        self.input_pos1 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_pos1')
        self.input_pos2 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_pos2')


        l2_loss = tf.constant(0.0)
        text_length = self._length(self.input_text)

        self.total_shape = tf.placeholder(dtype=tf.int32, shape=[None ,sequence_length], name='total_shape')
        total_num = self.total_shape[-1]
        attention_w = tf.get_variable('attention_omega', [hidden_size, num_classes])
        sen_a = tf.get_variable('attention_A', [hidden_size])
        sen_r = tf.get_variable('query_r', [hidden_size, 1])
        relation_embedding = tf.get_variable('relation_embedding', [num_classes, hidden_size])
        sen_d = tf.get_variable('bias_d', [num_classes])

        sen_repre = []
        sen_alpha = []
        sen_s = []
        sen_out = []
        self.prob = []
        self.predictions=[]
        self.loss=[]

        # Embedding layer
        #with tf.device('/cpu:0'), tf.name_scope("text-embedding"):
            #self.W_text = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W_text")
            #self.embedded_chars = tf.nn.embedding_lookup(self.W_text, self.input_text)


        with tf.device('/cpu:0'), tf.name_scope("text-embedding"):
            self.W_text = tf.Variable(tf.random_uniform([text_vocab_size, text_embedding_size], -1.0, 1.0),
                                      name="W_text")
            self.text_embedded_chars = tf.nn.embedding_lookup(self.W_text, self.input_text)
            #self.text_embedded_chars_expanded = tf.expand_dims(self.text_embedded_chars, -1)
        with tf.device('/cpu:0'), tf.name_scope("position-embedding"):
            self.W_position = tf.Variable(tf.random_uniform([pos_vocab_size, pos_embedding_size], -1.0, 1.0),
                                          name="W_position")
            self.pos1_embedded_chars = tf.nn.embedding_lookup(self.W_position, self.input_pos1)
            #self.pos1_embedded_chars_expanded = tf.expand_dims(self.pos1_embedded_chars, -1)
            self.pos2_embedded_chars = tf.nn.embedding_lookup(self.W_position, self.input_pos2)
            #self.pos2_embedded_chars_expanded = tf.expand_dims(self.pos2_embedded_chars, -1)

        self.embedded_chars_expanded = tf.concat([self.text_embedded_chars,
                                                  self.pos1_embedded_chars,
                                                  self.pos2_embedded_chars],2)

        embedding_size = text_embedding_size + 2 * pos_embedding_size
        print("embeed real one",self.text_embedded_chars)
        print("embeed ", self.embedded_chars_expanded)
        # Bidirectional Recurrent Neural Network
        with tf.name_scope("rnn"):
            fw_cell = self._get_cell(hidden_size, cell_type)
            bw_cell=self._get_cell(hidden_size,cell_type)
            fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=0.5)
            bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=0.5)
            (all_outputs_fw,all_output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,cell_bw=bw_cell,
                                               inputs=self.embedded_chars_expanded,
                                               sequence_length=text_length,
                                               dtype=tf.float32)
            #all_outputs=tf.concat([all_outputs_fw, all_output_bw], axis=-1)
            all_outputs = all_outputs_fw+all_output_bw
            self.h_outputs = (self.flatten(all_outputs, text_length))

        #attention layer
        M=tf.tanh(self.h_outputs)
        #print("size of M=",M.shape)
        #print("size of w=", attention_w.shape)
        alpha=tf.nn.softmax(tf.matmul(M,attention_w))
        attention_r=tf.matmul(alpha,self.h_outputs)
        print("size of alpha=", alpha.shape)
        attention_r = tf.reshape(tf.matmul(tf.reshape(tf.nn.softmax(
            tf.reshape(tf.matmul(tf.reshape(tf.tanh(self.h_outputs), [total_num * sequence_length, hidden_size]), attention_w),
                       [total_num, sequence_length])), [total_num,1,  sequence_length]), self.h_outputs), [total_num, hidden_size])

        # sentence-level attention layer
        for i in range(big_num):

            sen_repre.append(tf.tanh(attention_r[self.total_shape[i]:self.total_shape[i + 1]]))
            batch_size = self.total_shape[i + 1] - self.total_shape[i]

            sen_alpha.append(
                tf.reshape(tf.nn.softmax(tf.reshape(tf.matmul(tf.multiply(sen_repre[i], sen_a), sen_r), [batch_size])),
                           [1, batch_size]))

            sen_s.append(tf.reshape(tf.matmul(sen_alpha[i], sen_repre[i]), [hidden_size, 1]))
            sen_out.append(tf.add(tf.reshape(tf.matmul(relation_embedding, sen_s[i]), [num_classes]), sen_d))

            self.prob.append(tf.nn.softmax(sen_out[i]))

            with tf.name_scope("output"):
                self.predictions.append(tf.argmax(self.prob[i], 0, name="predictions"))

            with tf.name_scope("loss"):
                self.loss.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(sen_out[i], self.input_y[i])))
                if i == 0:
                    self.total_loss = self.loss[i]
                else:
                    self.total_loss += self.loss[i]

            # tf.summary.scalar('loss',self.total_loss)
            # tf.scalar_summary(['loss'],[self.total_loss])
            with tf.name_scope("accuracy"):
                self.accuracy.append(tf.reduce_mean(
                tf.cast(tf.equal(self.predictions[i], tf.argmax(self.input_y[i], 0)), "float"),name="accuracy"))

                # tf.summary.scalar('loss',self.total_loss)
                tf.scalar_summary('loss', self.total_loss)
                # regularization
                self.l2_loss = tf.contrib.layers.apply_regularization(
                    regularizer=tf.contrib.layers.l2_regularizer(0.0001), weights_list=tf.trainable_variables())
                self.final_loss = self.total_loss + self.l2_loss
                tf.scalar_summary('l2_loss', self.l2_loss)
                tf.scalar_summary('final_loss', self.final_loss)


    @staticmethod
    def _get_cell(hidden_size, cell_type):


        if cell_type == "lstm":
            return tf.contrib.rnn.LSTMCell(hidden_size,state_is_tuple=False)
        else:
            print("ERROR: '" + cell_type + "' is a wrong cell type !!!")
            return None

    # Length of the sequence data
    @staticmethod
    def _length(seq):
        relevant = tf.sign(tf.abs(seq))
        length = tf.reduce_sum(relevant, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length


    @staticmethod
    def flatten(seq, length):
        batch_size = tf.shape(seq)[0]
        max_length = int(seq.get_shape()[1])
        input_size = int(seq.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(seq, [-1, input_size])
        return tf.gather(flat, index)