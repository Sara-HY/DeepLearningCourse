import scipy.io as scio
import shutil, os
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageOps, ImageFile
import numpy as np
import random
import threading, time


# split the raw data
def split_data():
	data = scio.loadmat("data/OxFlowers/datasplits.mat")
	# print(data['trn1'][0], data['val1'][0], data['tst1'][0])

	source_path = 'data/OxFlowers/image/'
	train_path = 'data/OxFlowers/trainSet/'
	valid_path = 'data/OxFlowers/validSet/'
	test_path = 'data/OxFlowers/testSet/'

	# load train set
	os.mkdir(train_path)
	for id in data['trn1'][0]:
		file_name = 'image_' + '{:0>4}'.format(str(id)) + '.jpg'
		shutil.copy(os.path.join(source_path, file_name), os.path.join(train_path, file_name))

	# load validation set
	os.mkdir(valid_path)
	for id in data['val1'][0]:
		file_name = 'image_' + '{:0>4}'.format(str(id)) + '.jpg'
		shutil.copy(os.path.join(source_path, file_name), os.path.join(valid_path, file_name))

	# load test set
	os.mkdir(test_path)
	for id in data['tst1'][0]:
		file_name = 'image_' + '{:0>4}'.format(str(id)) + '.jpg'
		shutil.copy(os.path.join(source_path, file_name), os.path.join(test_path, file_name))


# load image data and labels 
def load_data(file_path, ImageWidth, ImageHeight):
	images = []
	labels = []

	# load data and label
	for file_name in os.listdir(file_path):
		# image and normalize 
		img = Image.open(os.path.join(file_path, file_name))
		img = img.resize((ImageWidth, ImageHeight))
		img = np.asarray(img, dtype="float32")
		img /= 255 
		
		# label
		num = file_name.split('_')[1].split('.')[0]
		label = int((int(num) - 1) / 80)
		images.append(img)
		labels.append(label)

	return np.array(images), np.array(labels)


# image rotation
def random_flip(image, i):
	method = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]
	return image.transpose(method[i])


# image crop
def random_crop(image, i):
	image_width = image.size[0]
	image_height = image.size[1]
	crop_win_size = np.random.randint(0.6 * image_height, image_height)
	random_region = ((image_width - crop_win_size) >> 1, (image_height - crop_win_size) >> 1, (image_width + crop_win_size) >> 1, (image_height + crop_win_size) >> 1)

	return image.crop(random_region)


# image channel noise
def gaussianNoisy(channel, mean=0.2, sigma=0.3):
	for i in range(len(channel)):
		channel[i] += random.gauss(mean, sigma)
	return channel


# image noise
def random_gaussian(image, i, mean=0.2, sigma=0.3):
	img = np.asarray(image)
	img.flags.writeable = True
	width, height = img.shape[:2]
	img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
	img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
	img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
	
	img[:, :, 0] = img_r.reshape([width, height])
	img[:, :, 1] = img_g.reshape([width, height])
	img[:, :, 2] = img_b.reshape([width, height])

	return Image.fromarray(np.uint8(img))
    

# data augmentation opeartors
def image_ops(func, func_name, image, target_path, file_name, times=4):
	for i in range(0, times, 1):
		if func_name == "randomFlip" :
			name_id = i % 4
		else:
			name_id = i
		new_image = func(image, name_id)	
		if os.path.isdir(target_path) == 0:
			os.mkdir(target_path)
		new_image.save(os.path.join(target_path, func_name + str(name_id) + file_name))


# data augmentation
def data_augmentation(source_path, target_path):
	ops_list = {
		"randomFlip": random_flip,
		"randomCrop": random_crop,
		"randomGaussian": random_gaussian
	}

	for file_name in os.listdir(source_path):
		# flag = np.random.randint(0, 2)
		# if flag:
		image = Image.open(os.path.join(source_path, file_name))
		# threadImage = [0] * 5
		# _index = 0
		for ops_name in ops_list:
			# flag2 = np.random.randint(0, 2)
			# if flag2:
			# threadImage[_index] = threading.Thread(target=image_ops, args=(ops_list[ops_name], ops_name, image, target_path, file_name, 10))
			# threadImage[_index].start()
			# _index += 1
			# time.sleep(0.2)
			image_ops(ops_list[ops_name], ops_name, image, target_path, file_name, 10)


# if __name__ == '__main__':
# 	# split_data()
# 	data_augmentation('data/OxFlowers/trainSet', 'data/OxFlowers/trainAugmentation')



	


