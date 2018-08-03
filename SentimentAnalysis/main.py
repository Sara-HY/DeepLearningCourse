from utils import *
from model import *


# get mini_batches of the data
def get_minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx: start_idx + batch_size]
		else:
			excerpt = slice(start_idx, start_idx + batch_size)

		yield np.array(inputs)[excerpt], np.array(targets)[excerpt]


# get the accuracy and cost of batches
def train_batch(model, n_samples, x, y, batch_size, mode, shuffle=True):
	avg_cost = 0
	avg_accuracy = 0

	for batch_xs, batch_ys in get_minibatches(x, y, batch_size, shuffle):
		if mode == "training":
			# fit training
			cost, accuracy = model.partial_fit(batch_xs, batch_ys)
		else:
			cost, accuracy = model.partial_accuracy(batch_xs, batch_ys)
        
        # computer loss
		avg_cost += cost / n_samples * batch_size
		avg_accuracy += accuracy / n_samples * batch_size

	return avg_cost, avg_accuracy


# training the autoencoder
def train(model, train_x, train_y, valid_x, valid_y, batch_size=100, training_epochs=10, display_step=1, shuffle=True):
    train_loss = []
    train_accuracy = []
    test_loss = []
    test_accuracy = []

    # training cycle
    best_accuracy = 0
    best_epoch = -1
    for epoch in range(training_epochs):
        train_avg_cost, train_avg_accuracy = train_batch(model, len(train_x), train_x, train_y, batch_size, mode='training', shuffle=True)
       
        # display
        if (epoch + 1) % display_step == 0:
            train_loss.append(train_avg_cost)
            train_accuracy.append(train_avg_accuracy)

            valid_avg_cost, valid_avg_accuracy = train_batch(model, len(valid_x), valid_x, valid_y, batch_size, mode='testing', shuffle=True)
            test_loss.append(valid_avg_cost)
            test_accuracy.append(valid_avg_accuracy)

            print("Epoch:", "%04d" % (epoch + 1), "Train_cost: ", "{:.9f}".format(train_avg_cost), "Train_accuracy: ", "{:.3f}".format(train_avg_accuracy), "Valid_cost: ", "{:.9f}".format(valid_avg_cost), "Valid_accuracy: ", "{:.3f}".format(valid_avg_accuracy))

            if(best_accuracy < valid_avg_accuracy):
            	model.save_model('data/SentimentAnalysis/model/model.ckpt')
            	best_accuracy = valid_avg_accuracy
            	best_epoch = epoch + 1
            	print("best_epoch", best_epoch)

    return train_loss, train_accuracy, test_loss, test_accuracy, best_epoch


# get the accuracy and cost of batches
def test_batch(sess, n_samples, x, y, batch_size, tensor_x, tensor_y, tensor_cost, tensor_accuracy, tensor_keep_prob, shuffle=True):
	avg_cost = 0
	avg_accuracy = 0

	for batch_xs, batch_ys in get_minibatches(x, y, batch_size, shuffle):
		cost, accuracy = sess.run((tensor_cost, tensor_accuracy), feed_dict={tensor_x: batch_xs, tensor_y: batch_ys, tensor_keep_prob: 1.0})     
       
		avg_cost += cost / n_samples * batch_size
		avg_accuracy += accuracy / n_samples * batch_size

	return avg_cost, avg_accuracy


def test(test_x_new, test_y_new, batch_size):
	saver = tf.train.import_meta_graph('data/SentimentAnalysis/model/model.ckpt' + '.meta')
	with tf.Session() as sess:
		saver.restore(sess, 'data/SentimentAnalysismodel/model.ckpt')
		graph = tf.get_default_graph()
		x = graph.get_tensor_by_name("x:0")
		y = graph.get_tensor_by_name("y:0")
		cost = tf.get_collection('cost')[0]
		accuracy = tf.get_collection('accuracy')[0]
		keep_prob = graph.get_tensor_by_name("keep_prob:0")
		
		test_avg_cost, test_avg_accuracy = test_batch(sess, len(test_x_new), test_x_new, test_y_new, batch_size, x, y, cost, accuracy, keep_prob, shuffle=True)

		print(test_avg_cost, test_avg_accuracy)


if __name__ == '__main__':

	batch_size = 32
	lstmUnits = 64
	numClasses = 5
	keep_probablity = 0.5
	regularizing_rate = 0.01
	# maxLen = 50
	dimension = 300
	learning_rate = 0.01
	training_epochs = 200
	display_step = 1
	num_layers = 3
	optimizer=tf.train.AdamOptimizer()

	train_x, train_y = read_data('data/SentimentAnalysis/train.txt')
	valid_x, valid_y = read_data('data/SentimentAnalysis/dev.txt')
	test_x, test_y = read_data('data/SentimentAnalysis/test.txt')

	# maxLen = max([len(x) for x in train_x] + [len(x) for x in valid_x] + [len(x) for x in test_x])
	maxLen = 50
	print(maxLen)

	words2ids, embeddings = load_embeddings("data/SentimentAnalysis/glove.6B.300d.txt", dimension)

	train_x_new, train_y_new = transform(train_x, train_y, words2ids, maxLen)
	valid_x_new, valid_y_new = transform(valid_x, valid_y, words2ids, maxLen)
	test_x_new, test_y_new = transform(test_x, test_y, words2ids, maxLen)

	model = SentimentAnalysis(embeddings, lstmUnits, maxLen, dimension, learning_rate, regularizing_rate, optimizer, keep_probablity, num_layers)

	train_loss, train_accuracy, test_loss, test_accuracy, epoch = train(model, train_x_new, train_y_new, valid_x_new, valid_y_new, batch_size, training_epochs, display_step)
	
	test(test_x_new, test_y_new, batch_size)
	
