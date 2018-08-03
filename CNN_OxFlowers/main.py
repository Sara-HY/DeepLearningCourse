from model import *
from data_process import *
import matplotlib.pyplot as plt


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

		yield inputs[excerpt], targets[excerpt]


# get the accuracy and cost of batches
def cnn_batch(n_samples, x, y, batch_size, mode, shuffle=True):
	avg_cost = 0
	avg_accuracy = 0

	for batch_xs, batch_ys in get_minibatches(x, y, batch_size, shuffle):
		if mode == "training":
			# fit training
			cost, accuracy = cnn.partial_fit(batch_xs, batch_ys)
		else:
			cost, accuracy = cnn.partial_accuracy(batch_xs, batch_ys)
        
        # computer loss
		avg_cost += cost / n_samples * batch_size
		avg_accuracy += accuracy / n_samples * batch_size

	return avg_cost, avg_accuracy


# training the neural network
def train(cnn, image_width, image_height, train_x, train_y, valid_x, valid_y, learning_rate=0.1, batch_size=68, training_epochs=10, display_step=5):
    train_loss = []
    train_accuracy = []
    test_loss = []
    test_accuracy = []
    
    # training cycle
    for epoch in range(training_epochs):

        train_avg_cost, train_avg_accuracy = cnn_batch(len(train_x), train_x, train_y, batch_size, mode='training', shuffle=True)
       
        # display
        if (epoch + 1) % display_step == 0:
            cnn.save_model("data/OxFlowers/model/")
            train_loss.append(train_avg_cost)
            train_accuracy.append(train_avg_accuracy)

            valid_avg_cost, valid_avg_accuracy = cnn_batch(len(valid_x), valid_x, valid_y, batch_size, mode='testing', shuffle=True)
            test_loss.append(valid_avg_cost)
            test_accuracy.append(valid_avg_accuracy)

            print("Epoch:", "%04d" % (epoch + 1), "Train_cost: ", "{:.9f}".format(train_avg_cost), "Train_accuracy: ", "{:.3f}".format(train_avg_accuracy), "Valid_cost: ", "{:.9f}".format(valid_avg_cost), "Valid_accuracy: ", "{:.3f}".format(valid_avg_accuracy))

    return train_loss, train_accuracy, test_loss, test_accuracy


# display the learning curve
def dispaly_results(title, train_cost, train_accuracy, test_cost, test_accuracy, step):
	training_iters = len(train_cost)
	# iters_steps
	iter_steps = [step *k for k in range(training_iters)]
    
	imh = plt.figure(1, figsize=(15, 14), dpi=160)
	# imh.tight_layout()
	# imh.subplots_adjust(top=0.88)

 	# final_acc = test_accuracy[-1]
	# img_title = "{}, Test Accuracy={:.4f}".format(title, final_acc)
 
	imh.suptitle(title)
	plt.subplot(221)
	plt.semilogy(iter_steps, train_cost, '-r', label='Train Loss')
	plt.title('Train Loss ')
	plt.legend(loc='upper right')

	plt.subplot(222)
	plt.plot(iter_steps, train_accuracy, '-g', label='Train Accuracy')
	plt.title('Train Accuracy')
	plt.legend(loc='upper right')

	plt.subplot(223)
	plt.semilogy(iter_steps, test_cost, '-r', label='Valid Loss')
	plt.title('Valid Loss')
	plt.legend(loc='upper right')

	plt.subplot(224)
	plt.plot(iter_steps, test_accuracy, '-g', label='Valid Accuracy')
	plt.title('Valid Accuracy')
	plt.legend(loc='upper right')

    #plt.tight_layout()
	plt.subplots_adjust(top=0.88)
    
	plot_file = "results/{}.png".format(title.replace(" ","_"))
	plt.savefig(plot_file)
	plt.show()


if __name__ == '__main__':
	image_width = 200
	image_height = 200

	train_x, train_y = load_data('data/OxFlowers/trainSet', image_width, image_height)
	train_aug_x, train_aug_y = load_data('data/OxFlowers/trainAugmentation', image_width, image_height)
	valid_x, valid_y = load_data('data/OxFlowers/validSet', image_width, image_height)
	
	# set the model parameters
	learning_rate = 0.01
	keep_probablity = 0.2
	regularizing_rate = 0.05
	batch_size = 128
	training_epochs = 5000
	display_step = 5
	print("AdamOptimizer", "learning_rate: ", learning_rate, " keep_probablity: ", keep_probablity, " regularizing_rate: ", regularizing_rate, " batch_size: ", batch_size, " training_epochs: ", training_epochs, " display_step: ", display_step)

	train_x = np.concatenate([train_x, train_aug_x])
	train_y = np.concatenate([train_y, train_aug_y])

	print(train_x.shape, train_y.shape)
	cnn = CNN(image_width, image_height, optimizer=tf.train.AdamOptimizer(), learning_rate=learning_rate, keep_probablity=keep_probablity, regularizing_rate = regularizing_rate)
	train_loss, train_accuracy, test_loss, test_accuracy = train(cnn, image_width, image_height, train_x, train_y, valid_x, valid_y, learning_rate, batch_size, training_epochs, display_step)
	dispaly_results("CNN_OxFlowers", train_loss, train_accuracy, test_loss, test_accuracy, display_step)
	
	
	test_x, test_y = load_data('data/OxFlowers/testSet', image_width, image_height)
	test_avg_cost, test_avg_accuracy = cnn_batch(len(test_x), test_x, test_y, batch_size, mode='testing', shuffle=True)
	print(test_avg_cost, test_avg_accuracy)

