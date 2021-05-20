#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 yuengdelahoz <yuengdelahoz@careaiLaptop>
#
# Distributed under terms of the MIT license.

"""

"""
# from python_speech_features import mfcc
import tensorflow as tf
from utils import  timing_val
import numpy as np

class Engine:
	def __init__(self):
		path = 'output/model'
		self.model = tf.keras.models.load_model(path)
		self.model.summary()

	@timing_val
	def predict(self,mfcc_feat): 
		_input = np.expand_dims(mfcc_feat,axis=0)
		pred = self.model.predict(_input)[0]
		idx = np.argmax(pred)
		confidence = pred[idx]
		return (idx,confidence)

