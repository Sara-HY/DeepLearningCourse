import tensorflow as tf 
import numpy as np

# CNN  
class CNN():
	def __init__(self, image_width=200, image_height=200, optimizer=tf.train.AdadeltaOptimizer(), learning_rate = 0.1, keep_probablity=0.5, regularizing_rate = 0.1):
		self.optimizer = optimizer
		self.learning_rate = learning_rate
		self.keep_probablity = keep_probablity
		self.regularizing_rate = regularizing_rate
		self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.regularizing_rate)

		# print("optimizer: ", self.optimizer)

		# tf graph input
		self.x = tf.placeholder(tf.float32, shape=[None, image_width, image_height, 3], name='x')
		self.y = tf.placeholder(tf.int32, shape = [None, ], name='y')
		self.keep_prob = tf.placeholder(tf.float32)
		
		# Create the network
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
		# layer 1  conv 200*200*3 -- 200*200*32
		with tf.variable_scope(name_or_scope='conv1') as scope: 
			conv1_weight = tf.get_variable("weight", [3, 3, 3, 32], regularizer=self.regularizer, initializer=tf.truncated_normal_initializer(stddev=0.1))
			conv1_bias = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
			conv1 = tf.nn.bias_add(tf.nn.conv2d(input=self.x, filter=conv1_weight, strides=[1, 1, 1, 1], padding='SAME'), conv1_bias)
			bn1 = tf.layers.batch_normalization(inputs=conv1)
			relu1 = tf.nn.relu(bn1)

		# layer 2 pooling 200*200*32 -- 100*100*32
		with tf.variable_scope(name_or_scope='pool1') as scope:
			pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

		# layer 3 conv 100*100*32 -- 100*100*64
		with tf.variable_scope(name_or_scope='conv2') as scope:
			conv2_weight = tf.get_variable("weight", [3, 3, 32, 64], regularizer=self.regularizer, initializer=tf.truncated_normal_initializer(stddev=0.1))
			conv2_bias = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
			conv2 = tf.nn.bias_add(tf.nn.conv2d(input=pool1, filter=conv2_weight, strides=[1, 1, 1, 1], padding='SAME'), conv2_bias)
			bn2 = tf.layers.batch_normalization(inputs=conv2)
			relu2 = tf.nn.relu(bn2)

		# layer 4 pooling 100*100*64 -- 50*50*64
		with tf.variable_scope(name_or_scope='pool2') as scope:
			pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

		# layer 5 conv 50*50*64 -- 50*50*128
		with tf.variable_scope(name_or_scope='conv3') as scope:
			conv3_weight = tf.get_variable("weight", [3, 3, 64, 128], regularizer=self.regularizer, initializer=tf.truncated_normal_initializer(stddev=0.1))
			conv3_bias = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
			conv3 = tf.nn.bias_add(tf.nn.conv2d(input=pool2, filter=conv3_weight, strides=[1, 1, 1, 1], padding='SAME'), conv3_bias)
			bn3 = tf.layers.batch_normalization(inputs=conv3)
			relu3 = tf.nn.relu(bn3)

		# layer 6 pooling 50*50*128 -- 25*25*128
		with tf.variable_scope(name_or_scope='pool3') as scope:
			pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
			shape = pool3.shape[1]*pool3.shape[2]*pool3.shape[3]
			reshaped = tf.reshape(pool3, [-1, shape])
		
		# # layer 7 conv 
		# with tf.variable_scope(name_or_scope='conv4') as scope:
		# 	conv4_weight = tf.get_variable("weight", [3, 3, 128, 256], initializer=tf.truncated_normal_initializer(stddev=0.1))
		# 	conv4_bias = tf.get_variable("bias", [256], initializer=tf.constant_initializer(0.0))
		# 	conv4 = tf.nn.bias_add(tf.nn.conv2d(input=pool3, filter=conv4_weight, strides=[1, 1, 1, 1], padding='SAME'), conv4_bias)
		# 	bn4 = tf.layers.batch_normalization(inputs=conv4)
		# 	relu4 = tf.nn.relu(bn4)

		# # layer 8 pooling 
		# with tf.variable_scope(name_or_scope='pool4') as scope:
		# 	pool4 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
		# 	shape = pool4.shape[1]*pool4.shape[2]*pool4.shape[3]
		# 	reshaped = tf.reshape(pool4, [-1, shape])

		# layer 9 fc1  25*25*128 -- 1024
		with tf.variable_scope('fc1'):
			fc1_weights = tf.get_variable("weight", [shape, 1024], regularizer=self.regularizer, initializer=tf.truncated_normal_initializer(stddev=0.1))
			fc1_biases = tf.get_variable("bias", [1024], initializer=tf.constant_initializer(0.0))
			fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)

			fc1_drop = tf.nn.dropout(fc1, self.keep_prob)

		# # layer 10 fc2
		# with tf.variable_scope('fc2'):
		# 	fc2_weights = tf.get_variable("weight", [1024, 512], initializer=tf.truncated_normal_initializer(stddev=0.1))
		# 	fc2_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.0))
		# 	fc2 = tf.nn.relu(tf.matmul(fc1_drop, fc2_weights) + fc2_biases)

		# 	fc2_drop = tf.nn.dropout(fc2, self.keep_prob)

		# layer 10 fc3  1024-17
		with tf.variable_scope('fc3'):
			fc3_weights = tf.get_variable("weight", [1024, 17], regularizer=self.regularizer, initializer=tf.truncated_normal_initializer(stddev=0.1))
			fc3_biases = tf.get_variable("bias", [17], initializer=tf.constant_initializer(0.0)) 
			self.logits = tf.add(tf.matmul(fc1_drop, fc3_weights), fc3_biases, name='logits')

	def _create_loss_optimizer(self):
		# regularaztion_loss
		regularaztion = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
		self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)) + regularaztion
		self.optimizer.__setattr__("learning_rate", self.learning_rate)
		self.optimizer_cost = self.optimizer.minimize(self.cost)


	def _create_accuracy(self):
		correct_prediction = tf.equal(tf.cast(tf.argmax(self.logits, 1), tf.int32), self.y)    
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


	# training model based on mini-batch of input data, return cost of mini-batch.
	def partial_fit(self, X, Y):
		opt, cost, accuracy = self.sess.run((self.optimizer_cost, self.cost, self.accuracy), feed_dict={self.x: X, self.y: Y, self.keep_prob: self.keep_probablity})    
		return cost, accuracy


	def partial_accuracy(self, X, Y):
		cost, accuracy = self.sess.run((self.cost, self.accuracy), feed_dict={self.x: X, self.y: Y, self.keep_prob: 1.0})     
		return cost, accuracy


	def save_model(self, model_path):
		saver = tf.train.Saver()
		saver.save(self.sess, model_path)


	def load_model(self, model_path):
		saver = tf.train.import_meta_graph(model_path + '.meta')
		saver.restore(self.sess, tf.train.latest_checkpoint(model_path))

		graph = tf.get_default_graph()
		self.x = graph.get_tensor_by_name("x:0")
		self.logits = graph.get_tensor_by_name("logits:0")















