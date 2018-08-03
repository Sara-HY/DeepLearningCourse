import tensorflow as tf


class SentimentAnalysis():
    def __init__(self, embeddings, lstmUnits, maxLen, dimension, learning_rate, regularizing_rate, optimizer=tf.train.AdamOptimizer(), keep_probablity=0.5, num_layers=1, num_classes=5):
        self.embeddings = embeddings
        self.lstmUnits = lstmUnits
        self.maxLen = maxLen
        self.dimension = dimension
        self.optimizer = optimizer 
        self.learning_rate = learning_rate
        self.keep_probablity = keep_probablity
        self.regularizing_rate = regularizing_rate
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.regularizing_rate)
        self.num_classes = num_classes
        self.num_layers = num_layers

        self.x = tf.placeholder(tf.int32, shape=[None, self.maxLen], name='x')
        self.y = tf.placeholder(tf.int32, shape=[None], name='y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # Create autoencoder network
        self._create_network()

        # Loss function
        self._create_loss_optimizer()

        self._create_accuracy()

        # initializing the tensor flow variables
        init = tf.global_variables_initializer()

        # launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)


    def _create_network(self):
        self.word_embeddings = tf.nn.embedding_lookup(self.embeddings, self.x)
        self.word_embeddings = tf.cast(self.word_embeddings, tf.float32) 
        self.word_embeddings = tf.nn.dropout(self.word_embeddings, self.keep_prob)

        def get_a_cell(lstmUnits, keep_prob):
            lstm = tf.nn.rnn_cell.LSTMCell(lstmUnits, initializer=tf.truncated_normal_initializer(stddev=0.05))
            drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            return drop

        with tf.name_scope('lstm'):
            cell_fw = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(self.lstmUnits, self.keep_prob) for _ in range(self.num_layers)], state_is_tuple=True)
            cell_bw = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(self.lstmUnits, self.keep_prob) for _ in range(self.num_layers)], state_is_tuple=True)

        # initial_state = cell_fw.zero_state(batch_size, tf.float32)

        x, state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.word_embeddings, dtype=tf.float32)
        # x, state = tf.nn.dynamic_rnn(cell, self.word_embeddings, dtype=tf.float32)

        # lstmCell = tf.contrib.rnn.BasicLSTMCell(self.lstmUnits)
        # lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
        # self.value, _ = tf.nn.dynamic_rnn(lstmCell, self.word_embeddings, dtype=tf.float32)
        
        self.value = tf.concat(x, -1)
        self.value = tf.reshape(self.value, [-1, self.value.shape[1]*self.value.shape[2]])
        
        # self.value2 = tf.layers.dense(self.value, 256, 
        #     kernel_initializer=tf.truncated_normal_initializer(stddev=0.05),
        #     kernel_regularizer=self.regularizer,
        #     activation = tf.nn.relu,
        #     )

        # self.value2 = tf.nn.dropout(self.value2, self.keep_prob)

        self.logits = tf.layers.dense(self.value, self.num_classes, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.05),
            kernel_regularizer=self.regularizer)

        tf.add_to_collection('logits', self.logits)


    def _create_loss_optimizer(self):
        tensor_variable = tf.trainable_variables()
        reg_cost = self.regularizing_rate * tf.reduce_sum([ tf.nn.l2_loss(v) for v in tensor_variable ]) 
        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)) + reg_cost
        
        tf.add_to_collection('cost', self.cost)
        # grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tensor_variable), self.max_grad_norm)
    
        # regularaztion = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        
        self.optimizer.__setattr__("learning_rate", self.learning_rate)
        # self.optimizer.apply_gradients(zip(grads, tensor_variable))
        self.optimizer_cost = self.optimizer.minimize(self.cost)


    def _create_accuracy(self):
        correctPred = tf.equal(tf.cast(tf.argmax(self.logits,1), tf.int32), self.y)
        self.accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
        tf.add_to_collection('accuracy', self.accuracy)


     # training model based on mini-batch of input data, return cost of mini-batch.
    def partial_fit(self, X, Y):
        opt, cost, accuracy = self.sess.run((self.optimizer_cost, self.cost, self.accuracy), feed_dict={self.x: X, self.y: Y, self.keep_prob: self.keep_probablity})
        return cost, accuracy


    def partial_accuracy(self, X, Y):
        cost, accuracy = self.sess.run((self.cost, self.accuracy), feed_dict={self.x: X, self.y: Y, self.keep_prob: 1.0})     
        return cost, accuracy


    def save_model(self, model_path):
        saver = tf.train.Saver(max_to_keep = 1)
        saver.save(self.sess, model_path)

