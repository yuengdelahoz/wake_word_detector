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
from utils import timing_val
from glob import glob

def create_folder(path):
	if not os.path.exists(path):
		os.makedirs(path)

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
def export():
	model = build_model((299,13))
	checkpoints = glob('output/checkpoints/*.index')
	checkpoint_path = ''
	_max_epoch = 0
	for ch in checkpoints:
		epoch = int(ch.split('-')[1])
		if epoch > _max_epoch:
			_max_epoch = epoch
			checkpoint_path = ch

	model.load_weights('output/checkpoints/chk-47-0.02464273.ckpt')
	print('saving model in checkpoint',checkpoint_path)
	model.save('output/model')


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

	log_dir ="output/logs"
	create_folder(log_dir)
	callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir))

	checkpoint_filepath = 'output/checkpoints/chk-{epoch:02d}-{val_loss:.8f}.ckpt'
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
			epochs=50,
			callbacks = callbacks
			)
