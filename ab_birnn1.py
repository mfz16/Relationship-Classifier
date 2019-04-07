import tensorflow as tf


class RNN:

    def __init__(self, sequence_length, num_classes,
                 text_vocab_size, text_embedding_size, pos_vocab_size, pos_embedding_size,cell_type,hidden_size,
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
        # BiDirectional Recurrent Neural Network
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
            output=tf.reshape(all_outputs,[-1,int(all_outputs.get_shape()[2])])
            self.h_outputs = (self.flatten(all_outputs, text_length))

        # Final scores and predictions
        with tf.name_scope("output"):
            #W = tf.get_variable("W", shape=[2*hidden_size, num_classes], initializer=tf.contrib.layers.xavier_initializer())
            W = tf.get_variable("W", shape=[hidden_size, num_classes], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.h_outputs, W, b, name="logits")
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")

    @staticmethod
    def _get_cell(hidden_size, cell_type):


        if cell_type == "lstm":
            return tf.contrib.rnn.LSTMCell(hidden_size,state_is_tuple=True)

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
        print("index is",index)
        flat = tf.reshape(seq, [-1, input_size])

        print("flat",flat.shape)
        #print("rrr",tf.gather(flat,index).shape)
        return tf.gather(flat, index)
        #return flat1