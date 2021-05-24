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
from .models import TrainingModel
import time
import sys
import tensorflow as tf
import os
from utils import timing_val
from glob import glob
import shutil

def create_folder(path):
	if not os.path.exists(path):
		os.makedirs(path)

@timing_val
def _load_split(dataset, split_name):
	split = dataset[split_name]
	X_split = np.array(split['mfccs'])
	y_split = np.array(split['labels'])
	assert len(X_split) == len(y_split)
	return X_split, y_split

@timing_val
def _load_dataset():
	# path = "/audio_files/dataset/json/dataset_split.json"
	path = "/audio_files/dataset/dataset.json"
	with open(path,'r') as fp:
		dataset_split = json.load(fp)
		X_train,y_train = _load_split(dataset_split,'train')
		X_test, y_test =_load_split(dataset_split,'test') 

	return X_train, X_test, y_train, y_test


@timing_val
def build_and_train():
	OUTPUT_FOLDER = 'output'
	if os.path.exists(OUTPUT_FOLDER):
		shutil.rmtree(OUTPUT_FOLDER)

	# sample = np.random.random((10,94,13))
	# labels = np.random.randint(0,2,size=(10))
	# X_train, X_test, y_train, y_test = sample, sample,labels,labels

	# get train, validation, and test splits
	X_train, X_test, y_train, y_test= _load_dataset()
	print(X_train.shape, y_train.shape)

	model_index = 3

	if model_index == 4:
		X_train, X_test = np.expand_dims(X_train,axis=-1), np.expand_dims(X_test,axis=-1)
		input_shape = (X_train.shape[1],X_train.shape[2],1)
	else:
		input_shape = (X_train.shape[1],X_train.shape[2])

	# create network
	trainig_model = TrainingModel(input_shape, model_index = model_index)
	trainig_model.train(X_train, X_test, y_train, y_test)

