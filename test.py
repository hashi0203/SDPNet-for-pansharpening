from __future__ import print_function

import time
import os
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import scipy.ndimage
from scipy.misc import imread, imsave
from skimage import transform, data
from glob import glob
import matplotlib.image as mpimg
import scipy.io as scio
import cv2
from pnet import PNet #_tradition
import time
import rasterio

from tensorflow.python import pywrap_tensorflow
from tqdm import tqdm

import config

# MODEL_SAVE_PATH = './models/5130.ckpt'
MODEL_SAVE_PATH = config.MODEL_SAVE_PATH
path1 = 'test_imgs/pan_org/'
path2 = 'test_imgs/ms_org/'
output_path = config.OUTPUT_PATH

os.mkdir(output_path)

def main():
	print('\nBegin to generate pictures ...\n')
	t=[]
	for i, p in tqdm(enumerate([255.0, config.dr])):
		file_name1 = path1 + str(i + 1) + '.png'
		file_name2 = path2 + str(i + 1) + '.tif'

		if i == 1:
			off_test = config.off_test
			pan = (imread(file_name1) + off_test) / p
			ms = (np.stack([rasterio.open(file_name2).read(c+1) for c in range(4)], axis=2) + off_test) / p
		else:
			off_test = 0
			pan = imread(file_name1) / p
			ms = imread(file_name2) / p
		print('file1:', file_name1, 'shape:', pan.shape)
		print('file2:', file_name2, 'shape:', ms.shape)

		h1, w1 = pan.shape
		pan = pan.reshape([1, h1, w1, 1])
		h2, w2, c = ms.shape
		ms = ms.reshape([1, h2, w2, 4])
		print(pan.shape, ms.shape)


		with tf.Graph().as_default(), tf.Session() as sess:
			MS = tf.placeholder(tf.float32, shape = (1, h2, w2, 4), name = 'MS')
			PAN = tf.placeholder(tf.float32, shape = (1, h1, w1, 1), name = 'PAN')
			Pnet = PNet('pnet')
			X = Pnet.transform(PAN = PAN, ms = MS)


			t_list = tf.trainable_variables()


			saver = tf.train.Saver(var_list = t_list)
			begin = time.time()
			sess.run(tf.global_variables_initializer())
			saver.restore(sess, MODEL_SAVE_PATH)

			output = sess.run(X, feed_dict = {PAN: pan, MS: ms})
			print(output.shape)
			output = np.where(output < 0, 0, output)

			if not os.path.exists(output_path):
				os.makedirs(output_path)
			scio.savemat(output_path + str(i + 1) + '.mat', {'i': output[0, :, :, :]})
			for j, c in enumerate(["red", "green", "blue", "nir"]):
				cv2.imwrite(output_path + str(i + 1) + '-' + c + '.tif', ((output[0, :, :, j] + off_test) * p).astype('uint' + str(8 * (i+1))))
				cv2.imwrite(output_path + c + '.tif', ((output[0, :, :, j] + off_test) * p).astype('uint' + str(8 * (i+1))))
			end=time.time()
			t.append(end-begin)
	print("Time: mean: %s,, std: %s" % (np.mean(t), np.std(t)))


if __name__ == '__main__':
	main()
