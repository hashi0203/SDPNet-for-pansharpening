from __future__ import print_function

import time
import os
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import scipy.ndimage
# from Net import Generator, WeightNet
from scipy.misc import imread, imsave
from skimage import transform, data
from glob import glob
import matplotlib.image as mpimg
import scipy.io as scio
import cv2
import rasterio
from pnet import PNet

from spat_ED import ED1
from spec_ED import ED2
from P2MSnet import pMS_ED
from MS2Pnet import pP_ED

from tensorflow.python import pywrap_tensorflow

import config
MS2P_MODEL_SAVEPATH = config.MS2P_MODEL_SAVEPATH
P2MS_MODEL_SAVEPATH = config.P2MS_MODEL_SAVEPATH
SPAT_MODEL_SAVEPATH = config.SPAT_MODEL_SAVEPATH
SPEC_MODEL_SAVEPATH = config.SPEC_MODEL_SAVEPATH

path1 = 'test_imgs/pan/'
path2 = 'test_imgs/ms/'
# output_path = 'features/'

dr = config.dr

def main():
	# print('\nBegin to generate pictures ...\n')
	"save features for examples"
	for i in range(100):
		# file_name1 = path1 + str(i + 1) + '.png'
		file_name1 = path1 + str(i + 1) + '.tif'
		file_name2 = path2 + str(i + 1) + '.tif'

		# pan = imread(file_name1) / dr
		# ms = imread(file_name2) / dr
		pan = rasterio.open(file_name1).read(1) / dr
		ms = np.stack([rasterio.open(file_name2).read(c+1) for c in range(4)], axis=2) / dr
		print('file1:', file_name1, 'shape:', pan.shape)
		print('file2:', file_name2, 'shape:', ms.shape)
		h1, w1 = pan.shape
		h2, w2, c = ms.shape

		with tf.Graph().as_default(), tf.Session() as sess:
			INPUT = tf.placeholder(tf.float32, shape = (1, h1, w1, 1), name = 'INPUT')
			with tf.device('/gpu:0'):
				spatnet = ED1('spatial_ED')
				OUTPUT = spatnet.transform(INPUT, is_training = False, reuse = False)
			spat_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'spatial_ED')

			MS = tf.placeholder(tf.float32, shape = (1, h1, w1, 4), name = 'MS')
			with tf.device('/gpu:0'):
				pPnet = pP_ED('pP_ED')
				MS_converted_PAN = pPnet.transform(I = MS, is_training = False, reuse = False)
			pP_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'pP_ED')

			t_list = tf.trainable_variables()
			sess.run(tf.global_variables_initializer())
			saver1 = tf.train.Saver(var_list = spat_list)
			saver1.restore(sess, SPAT_MODEL_SAVEPATH)
			saver2 = tf.train.Saver(var_list = pP_list)
			saver2.restore(sess, MS2P_MODEL_SAVEPATH)

			for c in range(4):
				if c == 0:
					ms_us = cv2.resize(ms[:, :, c], (h1, w1))
					ms_us = ms_us.reshape([1, h1, w1, 1])
				else:
					ms_upsampled = cv2.resize(ms[:, :, c], (h1, w1))
					ms_upsampled = ms_upsampled.reshape([1, h1, w1, 1])
					ms_us = np.concatenate([ms_us, ms_upsampled], axis = -1)
			cpan = sess.run(MS_converted_PAN, feed_dict = {MS: ms_us})
			pan = pan.reshape([1, h1, w1, 1])
			spat_features1 = sess.run(spatnet.features, feed_dict = {INPUT: pan})
			spat_features2 = sess.run(spatnet.features, feed_dict = {INPUT: cpan})

			diff = np.mean(np.abs(spat_features1 - spat_features2), axis = (1, 2))
			if i == 0:
				Diff = diff
			else:
				Diff = np.concatenate([Diff, diff], axis = 0)

	Diff = np.mean(Diff, axis=0)
	channel_sort = np.flip(np.argsort(Diff), axis=0)
	sorted_Diff = sorted(Diff, reverse=True)

	f = config.SPAT_DIFF_SAVEPATH
	for i in range(len(channel_sort)):
		if i==0:
			with open(f, "w") as file:
				file.write(str(channel_sort[i]) + "\n")
		else:
			with open(f, "a") as file:
				file.write(str(channel_sort[i]) + "\n")
	# scio.savemat('spat_diff.mat', {'D': Diff})



if __name__ == '__main__':
	main()