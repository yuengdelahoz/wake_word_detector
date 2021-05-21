#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 yuengdelahoz <yuengdelahoz@careaiLaptop>
#
# Distributed under terms of the MIT license.

"""

"""
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Conv2D, MaxPooling2D, BatchNormalization, Flatten, InputLayer
from utils import create_folder
import os
from glob import glob

def model_01(input_shape):
	model = Sequential([
		LSTM(128,input_shape=input_shape),
		Dropout(0.3),
		Dense(2, activation='softmax')
		])
	return model

def model_02(input_shape):
	model = Sequential([
		LSTM(128,input_shape=input_shape),
		Dropout(0.3),
		Dense(64, activation='relu'),
		Dense(2, activation='softmax')
		])
	return model

def model_03(input_shape):
	model = Sequential([
		GRU(20,input_shape=input_shape),
		Dropout(0.3),
		Dense(1, activation='sigmoid')
		])
	return model

def model_04(input_shape):
	model = Sequential()
	model.add(InputLayer(input_shape=input_shape))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D((3, 3), strides=(2,2), padding='same'))
	model.add(BatchNormalization())
	model.add(Dropout(0.3))

	# model.add(Conv2D(64, (3, 3), activation='relu'))
	# model.add(MaxPooling2D((3, 3), strides=(2,2), padding='same'))
	# model.add(BatchNormalization())
	# model.add(Dropout(0.3))

	# model.add(Conv2D(32, (3, 3), activation='relu'))
	# model.add(MaxPooling2D((3, 3), strides=(2,2), padding='same'))
	# model.add(BatchNormalization())
	# model.add(Dropout(0.3))

	model.add(Flatten())
	model.add(Dense(64, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))

	return model

available_models = {
		1:model_01,
		2:model_02,
		3:model_03,
		4:model_04
		}

class TrainingModel:
	def __init__(self,  input_shape, model_index=1):
		self.model_index = model_index;
		self.model = available_models.get(model_index)(input_shape)
		self.model_index = model_index
		self.model.summary()
		input()

	def export(self):
		''' pick the best weights based on the validation loss '''
		checkpoints = glob('output/checkpoints/*.index')
		checkpoint_path = ''
		_max_epoch = 0
		for ch in checkpoints:
			epoch = int(ch.split('-')[1])
			if epoch > _max_epoch:
				_max_epoch = epoch
				checkpoint_path = ch

		checkpoint_path = checkpoint_path.replace('.index','')
		self.model.load_weights(checkpoint_path)
		self.model.save('output/model')
		print('saving model in checkpoint',checkpoint_path)
	
	def train(self, X_train, X_validation, X_test, y_train, y_validation, y_test):
		# compile model
		optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

		num_outputs = self.model.outputs[0].shape[-1]

		if num_outputs == 1:
			loss = 'binary_crossentropy' 
			metrics=[
				tf.keras.metrics.BinaryAccuracy(),
				tf.keras.metrics.Precision(),
				tf.keras.metrics.Recall()
				]
		else:
			loss='categorical_crossentropy',
			y_train = tf.one_hot(y_train ,num_outputs)
			y_validation =tf.one_hot(y_validation ,num_outputs)
			y_test = tf.one_hot(y_test ,num_outputs)

			metrics=[
				tf.keras.metrics.CategoricalAccuracy(),
				tf.keras.metrics.Precision(),
				tf.keras.metrics.Recall()
				]

		self.model.compile(optimizer=optimizer,
				loss=loss,
				metrics=metrics
				)

		# self.model.summary()
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
		self.model.fit(X_train,y_train, 
				validation_data=(X_validation,y_validation),
				batch_size=64,
				epochs=5,
				callbacks = callbacks
				)

		self.export() # load model with best weights and export
		self.model.evaluate(X_test,y_test, verbose=2)
