import tensorflow as tf 
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


# 5_layers FC neural network
# Build a 5-layers FC neural network, and adjust the active function, learning rate and optimizer algorithm. 
class Neural_network:
    def __init__(self, network_architecture, activation_function=tf.nn.softplus, 
                 optimizer=tf.train.AdadeltaOptimizer(), learning_rate=0.001):
        
        self.network_architecture = network_architecture
        self.active = activation_function
        self.optimizer = optimizer
        self.learning_rate = learning_rate

        # tf graph input
        self.x = tf.placeholder(tf.float32, [None, network_architecture['n_input']])
        self.y = tf.placeholder(tf.float32, [None, 10])
        
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
        network_weights = self._initialize_weight(**self.network_architecture)
        
        self.network_weights = network_weights
        
        #hidden_1
        hidden_1_output = self.active(tf.add(tf.matmul(self.x, network_weights['hidden_1_W']), network_weights['hidden_1_b']))
        
        #hidden_2
        hidden_2_output = self.active(tf.add(tf.matmul(hidden_1_output, network_weights['hidden_2_W']), network_weights['hidden_2_b']))
        
        #hidden_3
        hidden_3_output = self.active(tf.add(tf.matmul(hidden_2_output, network_weights['hidden_3_W']), network_weights['hidden_3_b']))
        
        #hidden_4
        hidden_4_output = self.active(tf.add(tf.matmul(hidden_3_output, network_weights['hidden_4_W']), network_weights['hidden_4_b']))
        
        #output
        self._y_logits = tf.add(tf.matmul(hidden_4_output, network_weights['hidden_5_W']), network_weights['hidden_5_b'])
        self._y = tf.nn.softmax(self._y_logits)
        
     
    def _initialize_weight(self, n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, n_hidden_5):
        all_weights = dict()
        
        all_weights['hidden_1_W'] = tf.Variable(tf.random_normal([n_input, n_hidden_1] ,dtype=tf.float32))
        all_weights['hidden_2_W'] = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2] ,dtype=tf.float32))
        all_weights['hidden_3_W'] = tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3] ,dtype=tf.float32))
        all_weights['hidden_4_W'] = tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4] ,dtype=tf.float32))
        all_weights['hidden_5_W'] = tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5] ,dtype=tf.float32))
        
        all_weights['hidden_1_b'] = tf.Variable(tf.random_normal([n_hidden_1] ,dtype=tf.float32))
        all_weights['hidden_2_b'] = tf.Variable(tf.random_normal([n_hidden_2] ,dtype=tf.float32))
        all_weights['hidden_3_b'] = tf.Variable(tf.random_normal([n_hidden_3] ,dtype=tf.float32))
        all_weights['hidden_4_b'] = tf.Variable(tf.random_normal([n_hidden_4] ,dtype=tf.float32))
        all_weights['hidden_5_b'] = tf.Variable(tf.random_normal([n_hidden_5] ,dtype=tf.float32))
        
        return all_weights
     

    def _create_loss_optimizer(self):
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self._y_logits, labels=self.y)) * 100
        self.optimizer.__setattr__("learning_rate", self.learning_rate)
        self.optimizer_cost = self.optimizer.minimize(self.cost)
        
        
    def _create_accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self._y, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    

    # training model based on mini-batch of input data, return cost of mini-batch.
    def partial_fit(self, X, Y):
        opt, cost, accuracy = self.sess.run((self.optimizer_cost, self.cost, self.accuracy), feed_dict={self.x: X, self.y: Y})    
        return cost, accuracy

    def partial_accuracy(self, X, Y):
        cost, accuracy = self.sess.run((self.cost, self.accuracy), feed_dict={self.x: X, self.y: Y})     
        return cost, accuracy
    
   
# training the neural network
def train(mnist, n_samples, network_architecture, learning_rate=0.01, batch_size=100, training_epochs=10, display_step=5):
    # optimizers = [tf.train.AdamOptimizer(learning_rate), tf.train.AdadeltaOptimizer(learning_rate), tf.train.AdagradOptimizer(learning_rate)]
    # Learning_rate = [0.1, 0.4, 0.7]
    Active_function = [tf.nn.softplus, tf.nn.relu, tf.sigmoid]
    
    train_loss = [[] for i in range(len(Active_function))]
    train_accuracy = [[] for i in range(len(Active_function))]
    
    test_loss = [[] for i in range(len(Active_function))]
    test_accuracy = [[] for i in range(len(Active_function))]
    
    for choose in range(len(Active_function)):
        neural_network = Neural_network(network_architecture, learning_rate=learning_rate, active_function=Active_function[choose])
    
        # training cycle
        for epoch in range(training_epochs):
            avg_cost = 0
            avg_accuracy = 0
            total_batch = int(n_samples/batch_size)

            # loop over all batchs
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)

                # fit training
                cost, accuracy = neural_network.partial_fit(batch_xs, batch_ys)

                # computer loss
                avg_cost += cost / n_samples * batch_size
                avg_accuracy += accuracy / n_samples * batch_size

            if (epoch + 1) % display_step == 0:
                train_loss[choose].append(avg_cost)
                train_accuracy[choose].append(avg_accuracy)

                # computer test loss            
                cost, accuracy = neural_network.partial_accuracy(mnist.test.images, mnist.test.labels)

                test_loss[choose].append(cost)
                test_accuracy[choose].append(accuracy)
                print("Epoch:", "%04d" % (epoch + 1), "Train_cost: ", "{:.9f}".format(avg_cost), "Train_accuracy: ", "{:.3f}".format(avg_accuracy), "Test_cost: ", "{:.9f}".format(cost), "Test_accuracy: ", "{:.3f}".format(accuracy))

    return train_loss, train_accuracy, test_loss, test_accuracy


# display the learning curve
def dispaly_results(title, train_cost, train_accuracy, test_cost, test_accuracy, step):
    training_iters = len(train_cost[0])
    # iters_steps
    iter_steps = [step *k for k in range(training_iters)]
    
    imh = plt.figure(1, figsize=(15, 14), dpi=160)
    # imh.tight_layout()
    # imh.subplots_adjust(top=0.88)

    # final_acc = test_accuracy[-1]
    # img_title = "{}, Test Accuracy={:.4f}".format(title, final_acc)
 
    imh.suptitle(title)
    plt.subplot(221)
    plt.semilogy(iter_steps, train_cost[0], '-g', label='Train Loss softplus')
    plt.semilogy(iter_steps, train_cost[1], '-r', label='Train Loss relu')
    plt.semilogy(iter_steps, train_cost[2], '-b', label='Train Loss sigmoid')
    plt.title('Train Loss ')
    plt.legend(loc='upper right')
    
    plt.subplot(222)
    plt.plot(iter_steps, train_accuracy[0], '-g', label='Train Accuracy softplus')
    plt.plot(iter_steps, train_accuracy[1], '-r', label='Train Accuracy relu')
    plt.plot(iter_steps, train_accuracy[2], '-b', label='Train Accuracy sigmoid')
    plt.title('Train Accuracy')
    plt.legend(loc='upper right')

    plt.subplot(223)
    plt.semilogy(iter_steps, test_cost[0], '-g', label='Test Loss softplus')
    plt.semilogy(iter_steps, test_cost[1], '-r', label='Test Loss relu')
    plt.semilogy(iter_steps, test_cost[2], '-b', label='Test Loss sigmoid')
    plt.title('Test Loss')
    plt.legend(loc='upper right')
    
    plt.subplot(224)
    plt.plot(iter_steps, test_accuracy[0], '-g', label='Test Accuracy softplus')
    plt.plot(iter_steps, test_accuracy[1], '-r', label='Test Accuracy relu')
    plt.plot(iter_steps, test_accuracy[2], '-b', label='Test Accuracy sigmoid')
    plt.title('Test Accuracy')
    plt.legend(loc='upper right')


    #plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    plot_file = "results/{}.png".format(title.replace(" ","_"))
    plt.savefig(plot_file)
    plt.show()
    


if __name__ == '__main__':
    # loading data
    mnist = input_data.read_data_sets('data/MNIST_data', one_hot = True)
    n_samples = mnist.train.num_examples
    
    # set the model parameters
    learning_rate = 0.5
    batch_size = 100
    training_epochs = 1000
    display_step = 5
    # examples_to_show = 10
    network_architecture = dict(n_input=784, n_hidden_1=500, n_hidden_2=200, n_hidden_3=100, n_hidden_4=50, n_hidden_5=10)
    
    # training
    train_loss, train_accuracy, test_loss, test_accuracy = train(mnist, n_samples, network_architecture, learning_rate, batch_size, training_epochs, display_step)

    dispaly_results("MNIST_5_layers Active_function", train_loss, train_accuracy, test_loss, test_accuracy, display_step)






