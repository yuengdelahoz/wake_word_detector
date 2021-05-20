#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 yuengdelahoz <yuengdelahoz@careaiLaptop>
#
# Distributed under terms of the MIT license.

"""

"""
import json
import numpy as np 
from .models import build_model
import time
import sys
import tensorflow as tf
import os

def create_folder(path):
	if not os.path.exists(path):
		os.makedirs(path)

def timing_val(func):
	def wrapper(*arg, **kw):
		t1 = time.time()
		f = func(*arg, **kw)
		t2 = time.time()
		print('It took {:.2f} secs to execute function {}'.format((t2 - t1), func.__name__))
		return f
	return wrapper

@timing_val
def _load_split(dataset, split_name):
	split = dataset[split_name]
	X_split = np.array(split['mfccs'])
	y_split = np.array(split['labels'])
	y_split = tf.one_hot(y_split,2)
	assert len(X_split) == len(y_split)
	return X_split, y_split

@timing_val
def _load_dataset():
	path = "/audio_files/dataset/json/dataset_split.json"
	with open(path,'r') as fp:
		dataset_split = json.load(fp)
		X_train,y_train = _load_split(dataset_split,'train')
		X_validation, y_validation =_load_split(dataset_split,'validation')
		X_test,y_test =_load_split(dataset_split,'test') 

	return X_train, X_validation, X_test, y_train, y_validation, y_test

@timing_val
def build_and_train():
	# get train, validation, and test splits
	X_train, X_validation, X_test, y_train, y_validation, y_test = _load_dataset()
	print(X_train.shape, y_train.shape)

	# create network
	input_shape = (X_train.shape[1],X_train.shape[2])
	model = build_model(input_shape)

	# compile model
	optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
	model.compile(optimizer=optimizer,
			loss='categorical_crossentropy',
			metrics=[
				tf.keras.metrics.CategoricalAccuracy(),
				tf.keras.metrics.Precision(),
				tf.keras.metrics.Recall()
				]
			)
	model.summary()
	callbacks = list()

	callbacks.append(tf.keras.callbacks.TensorBoard())

	checkpoint_filepath = 'checkpoints/chk-{epoch:02d}-{val_loss:.8f}.ckpt'
	checkpoint_dir = os.path.dirname(checkpoint_filepath)
	create_folder(checkpoint_dir)
	model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
		filepath=checkpoint_filepath,
		save_weights_only=True,
		monitor='val_loss',
		mode='min',
		save_best_only=True,
		verbose=1
		)

	callbacks.append(model_checkpoint_callback)

	# train model
	history = model.fit(X_train,y_train, 
			validation_data=(X_validation,y_validation),
			batch_size=128,
			epochs=30,
			callbacks = callbacks
			)
