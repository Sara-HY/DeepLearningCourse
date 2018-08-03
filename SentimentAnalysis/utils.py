import numpy as np
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
import tensorflow.contrib.keras as kr
import matplotlib.pyplot as plt


# datasetSentences.txt
def get_sentences(file_path): 
	sentences = []			# array of all sentences
	sentence_ids = {}       # dict of the sentences for the ids
	
	with open(file_path) as f:
		f.readline()
		for line in f:
			sentence = line.split('\t')[1].strip('\n')
			sentences.append(sentence)
			sentence_ids[sentence] = line.split("\t")[0]

	return sentences, sentence_ids


# dictionary.txt
def get_phrases(file_path):
	phrases = {}            # dict of the phrases for index

	with open(file_path) as f:
		for line in f:
			line = line.split('|')
			phrases[line[0]] = line[1].strip('\n')

	return phrases


# sentiment_labels.txt
def get_phrase_sentiments(file_path):
	phrase_sentiments = {}  # dict of the sentiment value by index

	with open(file_path) as f:
		f.readline()
		for line in f:
			line = line.split('|')
			phrase_sentiments[line[0]] = line[1].strip('\n')

	return phrase_sentiments


# datasetSplit.txt 
def get_datasets(file_path):
	sentenceSet = {}        # sentence_index, splitset_label

	with open(file_path) as f:
		f.readline()
		for line in f:
			line = line.split(',')
			sentenceSet[line[0]] = line[1].strip('\n')

	return sentenceSet


def preprocess(text):
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=re.MULTILINE | re.DOTALL)

    # text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "_url_")
    # text = re_sub(r"@\w+", "_user_")
    # text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "_smile_")
    # text = re_sub(r"{}{}p+".format(eyes, nose), "_lolface_")
    # text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "_sadface_")
    # text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "_neutralface_")
    # text = re_sub(r"/"," / ")
    # text = re_sub(r"<3","_heart_")
    # text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "_number_")
    # text = re_sub(r"([!?.]){2,}", r"\1 _repeat_")
    # text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 _elong_")

    return text.lower()


def read_data(file_path):
	X = []
	y = []
	with open(file_path) as f:
		for line in f:
			X.append(line.split('\t')[0])
			y.append(int(line.split('\t')[2].strip('\n')))

	X = [preprocess(x) for x in X]

	return X, y


def load_embeddings(embedding_path, emb_dim):
    words2ids = {}
    vectors = [np.zeros(emb_dim)]
    i = 1
    with open(embedding_path, "r") as f:
        for line in f:
            tokens = line.split(" ")
            word = tokens[0]
            if True:
                v = list(map(np.float32, tokens[1:]))
                vectors.append(v)
                words2ids[word] = i
                i = i + 1

    vectors = np.array(vectors)
    words2ids["_unknown_"] = 0

    return words2ids, vectors


def transform(X, y, words2ids, maxlen):
	stop_words = list(set(stopwords.words('english')))

	#  if not x in stop_words
	X_new = [[words2ids.get(x, words2ids["_unknown_"]) for x in word_tokenize(X[i])] for i in range(len(X))]

	X_new = kr.preprocessing.sequence.pad_sequences(np.array(X_new), value=0., maxlen=maxlen)

	y_new = y

	return X_new, y_new


# display the learning curve
def dispaly_results(title, train_cost, train_accuracy, valid_cost, valid_accuracy, step):
    training_iters = len(train_cost)
    # iters_steps
    iter_steps = [step *k for k in range(training_iters)]
    
    imh = plt.figure(1, figsize=(8, 6), dpi=160)

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
    plt.semilogy(iter_steps, valid_cost, '-g', label='Test Loss')
    plt.title('Valid Loss')
    plt.legend(loc='upper right')
    
    plt.subplot(224)
    plt.plot(iter_steps, valid_accuracy, '-r', label='Test Accuracy')
    plt.title('Valid Accuracy')
    plt.legend(loc='upper right')


    #plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    plot_file = "result/{}.png".format(title.replace(" ","_"))
    plt.savefig(plot_file)
    plt.show()
    

def get_results(file_path='run2.log'):
    train_loss = []
    train_accuracy = []
    valid_loss = []
    valid_accuracy = []

    pfile = open(file_path, 'r')
    data = pfile.read().split('\n')
    for line in data:
        index1 = line.find("Train_cost: ")
        index2 = line.find("Train_accuracy: ")
        index3 = line.find("Valid_cost: ")
        index4 = line.find("Valid_accuracy: ")
        if index1 != -1 and index2 != -1 and index3 != -1 and index4 != -1 :
            # print(line[index1 + 13: index1 + 19], line[index2 + 17: index2 + 23], line[index3 + 13: index3 + 18], line[index4 + 16: index4 + 22])
            train_loss.append(float(line[index1 + 13: index1 + 19]))
            train_accuracy.append(float(line[index2 + 17: index2 + 23]))
            valid_loss.append(float(line[index3 + 13: index3 + 18]))
            valid_accuracy.append(float(line[index4 + 16: index4 + 22]))


    # print(len(train_loss), '\n', train_loss, '\n', train_accuracy, '\n', valid_loss, '\n', valid_accuracy)
    dispaly_results("LSTM 3 Layers", train_loss[0:200], train_accuracy[0:200], valid_loss[0:200], valid_accuracy[0:200], 1)
    # dispaly_results("BidirectionalLSTM", train_loss[50:], train_accuracy[50:], valid_loss[50:], valid_accuracy[50:], 1)

# lstm layer=2 1.36623766913 0.447479981159
# lstm layer=3 1.3709502146  0.455016486105


# if __name__ == '__main__':
#     get_results()
	# sentences, sentence_ids = get_sentences('data/SentimentAnalysis/datasetSentences.txt')
	# phrases = get_phrases('data/SentimentAnalysis/dictionary.txt')
	# phrase_sentiments = get_phrase_sentiments('data/SentimentAnalysis/sentiment_labels.txt')
	# sentenceSet = get_datasets('data/SentimentAnalysis/datasetSplit.txt')

	# train_file = open('data/SentimentAnalysis/train.txt', 'w')
	# dev_file = open('data/SentimentAnalysis/dev.txt', 'w')
	# test_file = open('data/SentimentAnalysis/test.txt', 'w')

	# for sentence in sentences:
	# 	if sentence in phrases and sentence in sentence_ids:
	# 		type_set = sentenceSet[sentence_ids[sentence]]
	# 		label = phrase_sentiments[phrases[sentence]] 
			
	# 		sentiment_class = int(float(label) * 5.0)
	# 		if sentiment_class >= 5:
	# 			sentiment_class = 4
	# 		if type_set == '1':
	# 			train_file.write(sentence + '\t' + label + '\t' + str(sentiment_class) + '\n')
	# 		elif type_set == '2':
	# 			test_file.write(sentence + '\t' + label + '\t' + str(sentiment_class) + '\n')
	# 		elif type_set == '3':
	# 			dev_file.write(sentence + '\t' + label + '\t' + str(sentiment_class) + '\n')

	# train_file.close()
	# dev_file.close()
	# test_file.close()

	
