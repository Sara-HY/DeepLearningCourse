import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# initialize weights
def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)

# Autoencoder class
# Build a 2-layers autoencoder network.
class Autoencoder(object):
    """docstring for Autoencoder"""
    def __init__(self, network_architecture, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(), batch_size=100, learning_rate=0.001):

        self.network_architecture = network_architecture
        self.transfer = transfer_function
        self.optimizer = optimizer 
        self.learning_rate = learning_rate

        # tf graph input
        self.x = tf.placeholder(tf.float32, [None, network_architecture['n_input']], name='X')

        # Create autoencoder network
        self._create_network()

        # Loss function
        self._create_loss_optimizer()

        # initializing the tensor flow variables
        init = tf.global_variables_initializer()

        # launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)


    def _create_network(self):
        # initiazing autoencoder network weights and biases
        network_weights = self._initialize_weights(**self.network_architecture)
        self.network_weights = network_weights

        # building the encoder
        encoder_layer_1 = self.transfer(tf.add(tf.matmul(self.x, network_weights['Encoder_W1']), network_weights['Encoder_b1']))
        encoder_layer_2 = self.transfer(tf.add(tf.matmul(encoder_layer_1, network_weights['Encoder_W2']), network_weights['Encoder_b2']), name='x_encoder')

        # building the decoder
        decoder_layer_1 = self.transfer(tf.add(tf.matmul(encoder_layer_2, network_weights['Decoder_W1']), network_weights['Decoder_b1']))
        decoder_layer_2 = self.transfer(tf.add(tf.matmul(decoder_layer_1, network_weights['Decoder_W2']), network_weights['Decoder_b2']))

        self.x_encoder = encoder_layer_2
        self.x_decoder = decoder_layer_2


    def _initialize_weights(self, n_input, n_hidden_1, n_hidden_2):
        all_weights = dict()

        all_weights['Encoder_W1'] = tf.Variable(xavier_init(n_input, n_hidden_1))
        all_weights['Encoder_W2'] = tf.Variable(xavier_init(n_hidden_1, n_hidden_2))
        all_weights['Decoder_W1'] = tf.Variable(xavier_init(n_hidden_2, n_hidden_1))
        all_weights['Decoder_W2'] = tf.Variable(xavier_init(n_hidden_1, n_input))

        all_weights['Encoder_b1'] = tf.Variable(tf.zeros([n_hidden_1]))
        all_weights['Encoder_b2'] = tf.Variable(tf.zeros([n_hidden_2]))
        all_weights['Decoder_b1'] = tf.Variable(tf.zeros([n_hidden_1]))
        all_weights['Decoder_b2'] = tf.Variable(tf.zeros([n_input]))

        return all_weights


    def _create_loss_optimizer(self):
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.x, self.x_decoder), 2))
        self.optimizer.__setattr__("learning_rate", self.learning_rate)
        self.optimizer_cost = self.optimizer.minimize(self.cost)


    # training model based on mini-batch of input data, return cost of mini-batch.
    def partial_fit(self, X):
        opt, cost = self.sess.run((self.optimizer_cost, self.cost), feed_dict={self.x: X})
        return cost


    # transform data by mapping
    def transform(self, X):
        return self.sess.run(self.x_encoder, feed_dict={self.x: X})


    def reconstruct(self, X):
        return self.sess.run(self.x_decoder, feed_dict={self.x: X})
    
    # get the graph, and set all to const 
    def save_model_to_pb(self):
        graph_def = tf.get_default_graph().as_graph_def()

        output_graph_def = graph_util.convert_variables_to_constants(self.sess, graph_def, ['x_encoder'])
        
        with tf.gfile.GFile('data/MNIST_data/autocoder.pb', 'wb') as g:
            g.write(output_graph_def.SerializeToString())
   

# training the autoencoder
def train(mnist, n_samples, network_architecture, learning_rate=0.01, batch_size=100, training_epochs=10, display_step=5):
    autoencoder = Autoencoder(network_architecture, learning_rate=learning_rate, batch_size=batch_size)

    # training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(n_samples/batch_size)

        # loop over all batchs
        for i in range(total_batch):
            batch_xs, _ = mnist.train.next_batch(batch_size)

            # fit training
            cost = autoencoder.partial_fit(batch_xs)

            # computer loss
            avg_cost += cost / n_samples * batch_size

        if (epoch + 1) % display_step == 0:
            
            print("Epoch:", "%04d" % (epoch + 1), "cost: ", "{:.9f}".format(avg_cost))

    return autoencoder


# display the reconstruct results
def show_reconstruct(mnist, autoencoder):
	x_sample = mnist.test.next_batch(100)[0]
	x_reconstruct = autoencoder.reconstruct(x_sample)

	plt.figure(figsize=(8, 12))
	for i in range(5):

	    plt.subplot(5, 2, 2*i + 1)
	    plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
	    plt.title("Test input")
	    plt.colorbar()
	    plt.subplot(5, 2, 2*i + 2)
	    plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
	    plt.title("Reconstruction")
	    plt.colorbar()

	plt.show()
	plt.tight_layout()


def show_classifier(mnist, Autoencoder):
	x_sample, y_sample = mnist.test.next_batch(5000)
	label = autoencoder.transform(x_sample)
	plt.figure(figsize=(8, 6)) 
	plt.scatter(label[:, 0], label[:, 1], c=np.argmax(y_sample, 1))
	plt.colorbar()
	plt.grid() 
	plt.show()


# Load the autocoder network and add a full_connected softmax layer, the result locate at results/MNIST_2_layers_Autoencoder.png
# display the learning curve
def dispaly_results(title, train_cost, train_accuracy, test_cost, test_accuracy, step):
    training_iters = len(train_cost)
    # iters_steps
    iter_steps = [step *k for k in range(training_iters)]
    
    imh = plt.figure(1, figsize=(15, 14), dpi=160)

    imh.suptitle(title)
    plt.subplot(221)
    plt.semilogy(iter_steps, train_cost, '-g', label='Train Loss')
    plt.title('Train Loss ')
    plt.legend(loc='upper right')
    
    plt.subplot(222)
    plt.plot(iter_steps, train_accuracy, '-r', label='Train Accuracy')
    plt.title('Train Accuracy')
    plt.legend(loc='upper right')

    plt.subplot(223)
    plt.semilogy(iter_steps, test_cost, '-g', label='Test Loss')
    plt.title('Test Loss')
    plt.legend(loc='upper right')
    
    plt.subplot(224)
    plt.plot(iter_steps, test_accuracy, '-r', label='Test Accuracy')
    plt.title('Test Accuracy')
    plt.legend(loc='upper right')


    #plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    plot_file = "results/{}.png".format(title.replace(" ","_"))
    plt.savefig(plot_file)
    plt.show()


def classifier():
    n_hidden = 200

    # add a softmax layer
    # train_data
    x = tf.placeholder(tf.float32, shape=[None, n_hidden])
    _y = tf.placeholder(tf.float32, shape=[None, 10])

    # model wights
    W = tf.Variable(tf.random_normal(shape=[n_hidden, 10], dtype=tf.float32))
    b = tf.Variable(tf.random_normal(shape=[10], dtype=tf.float32))

    # cost
    logits = tf.add(tf.matmul(x, W), b)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=_y, logits=logits))
   
    # optimizer
    train_op = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)

    # accuracy
    correct_prediction = tf.equal(tf.argmax(_y, 1), tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    # initializing the tensor flow variables
    init = tf.global_variables_initializer()

    # launch the session
    sess = tf.InteractiveSession()
    sess.run(init) 

    # load model
    with gfile.FastGFile('data/MNIST_data/autocoder.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
            

    x_tensor, x_encoder_tensor = tf.import_graph_def(graph_def, return_elements=['X:0', 'x_encoder:0'])

    mnist = input_data.read_data_sets('data/MNIST_data', one_hot = True)
    n_samples = mnist.train.num_examples


    # set the model parameters
    training_epochs = 1000
    batch_size = 128
    display_step = 5

    train_loss = []
    train_accuracy = [] 
    test_loss = [] 
    test_accuracy = []
        
    # training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        avg_accuracy = 0
        total_batch = int(n_samples/batch_size)

        # loop over all batchs
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            train_features = sess.run(x_encoder_tensor, feed_dict={x_tensor: batch_xs})

            # fit training
            cost, _, acc = sess.run((cross_entropy, train_op, accuracy), feed_dict={x: train_features, _y:batch_ys})

            # computer loss
            avg_cost += cost / n_samples * batch_size
            avg_accuracy += acc /n_samples * batch_size
            
        if (epoch + 1) % display_step == 0:
            train_loss.append(avg_cost)
            train_accuracy.append(avg_accuracy)
                
            # computer test loss            
            test_features = sess.run(x_encoder_tensor, feed_dict={x_tensor: mnist.test.images})  
            test_labels = mnist.test.labels
            cost, _, acc = sess.run((cross_entropy, accuracy), feed_dict={x: test_features, _y: mnist.test.labels})

            test_loss.append(cost)
            test_accuracy.append(acc)
            
            print("Epoch:", "%04d" % (epoch + 1), "Train_cost: ", "{:.9f}".format(avg_cost), "Train_accuracy: ", "{:.3f}".format(avg_accuracy), "Test_cost: ", "{:.9f}".format(cost), "Test_accuracy: ", "{:.3f}".format(acc))

    dispaly_results("Mnist Autoencoder_2_layers", train_loss, train_accuracy, test_loss, test_accuracy, display_step)


if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)

    # loading data
    mnist = input_data.read_data_sets('data/MNIST_data', one_hot = True)
    n_samples = mnist.train.num_examples

    # set the model parameters
    learning_rate = 0.01
    batch_size = 100
    training_epochs = 500
    display_step = 5
    network_architecture = dict(n_input = 784, n_hidden_1 = 500, n_hidden_2 = 200)

    # training
    autoencoder = train(mnist, n_samples, network_architecture, learning_rate, batch_size, training_epochs, display_step)

    # save model
    autoencoder.save_model_to_pb()
    # show_reconstruct(mnist, autoencoder)
    # show_classifier(mnist, autoencoder)

    classifier()




